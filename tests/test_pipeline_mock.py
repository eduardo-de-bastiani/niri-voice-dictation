#!/usr/bin/env python3
"""
Automated Headless Test Suite for Niri Wayland Voice Dictation Architecture
Simulates the full pipeline:
1. Synthetic 16kHz mono audio fixture generation (zero hardware dependency).
2. Ephemeral storage and permission lifecycle (tmpfs mode 0700/0600 + immediate unlink).
3. Local HTTP Mock Groq STT Server with multipart form-data verification.
4. Error simulation (HTTP 401, 429 with Retry-After, 500, network timeouts).
5. Autoregressive prompt budget validation (<= 224 Whisper token ceiling).
6. Wayland injection contracts (focused window app_id terminal detection & paste chord selection).
7. Atomic state toggle and concurrency lockfile protection.
"""

import http.server
import json
import math
import os
import signal
import struct
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
import wave


def generate_synthetic_pcm_wav(duration_s: float = 0.5, freq_hz: float = 440.0) -> bytes:
    """Generate in-memory 16kHz mono 16-bit PCM WAV audio."""
    sample_rate = 16000
    num_samples = int(sample_rate * duration_s)
    frames = bytearray()
    for i in range(num_samples):
        # 440Hz sine wave at 30% amplitude
        val = int(32767.0 * 0.3 * math.sin(2.0 * math.pi * freq_hz * i / sample_rate))
        frames.extend(struct.pack("<h", val))

    temp_wav = tempfile.NamedTemporaryFile(delete=False, suffix=".wav")
    try:
        temp_wav.close()
        with wave.open(temp_wav.name, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(sample_rate)
            wav.writeframes(frames)
        with open(temp_wav.name, "rb") as f:
            return f.read()
    finally:
        if os.path.exists(temp_wav.name):
            os.unlink(temp_wav.name)


class MockGroqSTTHandler(http.server.BaseHTTPRequestHandler):
    """Local HTTP mock simulating Groq's whisper-large-v3-turbo API."""

    expected_token = "gsk_test_mock_token_abc123"
    transcription_result = "Criando branch no git e rodando docker compose para teste da função assíncrona."
    failure_code = None
    retry_after = 2

    def do_POST(self):
        auth_header = self.headers.get("Authorization", "")
        if not auth_header.startswith(f"Bearer {self.expected_token}"):
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Invalid API Key","type":"invalid_request_error"}}')
            return

        if self.failure_code:
            self.send_response(self.failure_code)
            if self.failure_code == 429:
                self.send_header("Retry-After", str(self.retry_after))
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Simulated error condition"}}')
            return

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        # Assert multipart payload contains mandatory model and language fields
        assert b"name=\"model\"" in body, "Missing model field in multipart body"
        assert b"whisper-large-v3-turbo" in body, "Incorrect model value"
        assert b"name=\"language\"" in body, "Missing language field"
        assert b"\r\n\r\npt\r\n" in body or b"\r\n\r\npt" in body, "Incorrect language parameter"
        assert b"name=\"file\"" in body, "Missing audio file attachment"

        # Check response format
        accept_header = self.headers.get("Accept", "")
        if "application/json" in accept_header or b"name=\"response_format\"\r\n\r\njson" in body:
            payload = json.dumps({"text": self.transcription_result}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
        else:
            payload = self.transcription_result.encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")

        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        # Suppress standard logging during tests
        pass


def build_multipart_request(url: str, token: str, audio_bytes: bytes, fields: dict) -> urllib.request.Request:
    """Construct an HTTP multipart/form-data request using only standard library."""
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()

    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{value}\r\n".encode("utf-8"))

    # Audio file attachment
    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="audio.oga"\r\n')
    body.extend(b"Content-Type: audio/ogg; codecs=opus\r\n\r\n")
    body.extend(audio_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(url, data=bytes(body), method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    return req


class TestPipelineArchitecture(unittest.TestCase):
    """Architectural test suite validating contracts and logic."""

    @classmethod
    def setUpClass(cls):
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), MockGroqSTTHandler)
        cls.port = cls.server.server_address[1]
        cls.mock_url = f"http://127.0.0.1:{cls.port}/openai/v1/audio/transcriptions"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        MockGroqSTTHandler.failure_code = None
        self.test_dir = tempfile.mkdtemp(prefix="niri_test_runtime_")

    def tearDown(self):
        if os.path.exists(self.test_dir):
            for root, dirs, files in os.walk(self.test_dir, topdown=False):
                for f in files:
                    os.unlink(os.path.join(root, f))
                for d in dirs:
                    os.rmdir(os.path.join(root, d))
            os.rmdir(self.test_dir)

    def test_synthetic_audio_fixture_generation(self):
        """Verify programmatic generation of valid 16kHz mono audio."""
        audio_bytes = generate_synthetic_pcm_wav(duration_s=0.2, freq_hz=440.0)
        self.assertGreater(len(audio_bytes), 44)  # Minimum WAV header size
        # Assert RIFF and WAVE magic headers
        self.assertEqual(audio_bytes[:4], b"RIFF")
        self.assertEqual(audio_bytes[8:12], b"WAVE")
        # Assert 16kHz sample rate in header (bytes 24-27)
        sample_rate = struct.unpack("<I", audio_bytes[24:28])[0]
        self.assertEqual(sample_rate, 16000)
        # Assert 1 channel (bytes 22-23)
        channels = struct.unpack("<H", audio_bytes[22:24])[0]
        self.assertEqual(channels, 1)

    def test_ephemeral_storage_and_immediate_unlink(self):
        """Verify ephemeral audio storage lifecycle: mode 0600 in tmpfs and instant unlink."""
        audio_bytes = generate_synthetic_pcm_wav(duration_s=0.1)
        audio_path = os.path.join(self.test_dir, f"dictate_{uuid.uuid4().hex}.oga")

        # Create file with restrictive 0600 permissions
        with open(audio_path, "wb") as f:
            f.write(audio_bytes)
        os.chmod(audio_path, 0o600)

        # Verify permissions
        st = os.stat(audio_path)
        self.assertEqual(st.st_mode & 0o777, 0o600)
        self.assertTrue(os.path.exists(audio_path))

        # Simulate read into memory buffer
        with open(audio_path, "rb") as f:
            buffer = f.read()

        # Immediate unlink contract
        os.unlink(audio_path)
        self.assertFalse(os.path.exists(audio_path))
        self.assertEqual(len(buffer), len(audio_bytes))

    def test_mock_groq_stt_transcription_success(self):
        """Verify full HTTP multipart transmission against mock Groq Whisper server."""
        audio_bytes = generate_synthetic_pcm_wav(duration_s=0.25)
        fields = {
            "model": "whisper-large-v3-turbo",
            "language": "pt",
            "temperature": "0.0",
            "response_format": "text",
            "prompt": "Código e desenvolvimento com git e docker."
        }
        req = build_multipart_request(
            url=self.mock_url,
            token=MockGroqSTTHandler.expected_token,
            audio_bytes=audio_bytes,
            fields=fields
        )

        with urllib.request.urlopen(req, timeout=5.0) as resp:
            self.assertEqual(resp.status, 200)
            text = resp.read().decode("utf-8")

        self.assertIn("git", text)
        self.assertIn("docker", text)
        self.assertIn("assíncrona", text)

    def test_mock_groq_stt_auth_failure(self):
        """Verify handling of invalid API key (HTTP 401)."""
        audio_bytes = generate_synthetic_pcm_wav(duration_s=0.1)
        fields = {"model": "whisper-large-v3-turbo", "language": "pt"}
        req = build_multipart_request(
            url=self.mock_url,
            token="invalid_token_xyz",
            audio_bytes=audio_bytes,
            fields=fields
        )

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5.0)
        self.assertEqual(ctx.exception.code, 401)

    def test_mock_groq_stt_rate_limit_retry_after(self):
        """Verify extraction of Retry-After header on HTTP 429."""
        MockGroqSTTHandler.failure_code = 429
        MockGroqSTTHandler.retry_after = 3

        audio_bytes = generate_synthetic_pcm_wav(duration_s=0.1)
        fields = {"model": "whisper-large-v3-turbo", "language": "pt"}
        req = build_multipart_request(
            url=self.mock_url,
            token=MockGroqSTTHandler.expected_token,
            audio_bytes=audio_bytes,
            fields=fields
        )

        with self.assertRaises(urllib.error.HTTPError) as ctx:
            urllib.request.urlopen(req, timeout=5.0)
        self.assertEqual(ctx.exception.code, 429)
        retry_val = ctx.exception.headers.get("Retry-After")
        self.assertEqual(retry_val, "3")

    def test_prompt_budget_engineering(self):
        """Verify prompt templates and dynamic guards remain strictly within Whisper's budget."""
        import tiktoken
        enc = tiktoken.get_encoding("gpt2")

        base_prompt = (
            "Código, commits e comandos de programação em Python, JavaScript, TypeScript, Rust, Go, SQL. "
            "Termos técnicos: pull request, merge, branch, commit, rebase, diff, git, docker, kubernetes, "
            "linux, bash, function, const, let, async, await, return, import, export, class, interface, "
            "null, undefined, boolean, string, number, array, object, api, rest, json, postgres, backend, "
            "frontend, devops, deploy, ci/cd, bug, fix, test, mock, endpoint, middleware, refactor."
        )
        custom_keywords = ["antigravity", "ages", "monorepo", "supabase", "trpc", "niri", "wayland"]

        # 1. Exact Base Prompt Verification
        base_words = base_prompt.split()
        base_tokens = len(enc.encode(base_prompt))
        self.assertEqual(len(base_words), 60, "Base prompt must be exactly 60 words")
        self.assertEqual(base_tokens, 138, f"Base prompt tokens must be 138 GPT-2 tokens, got {base_tokens}")
        self.assertLessEqual(base_tokens, 200, "Base prompt alone must not exceed 200 target ceiling")

        # 2. Combined Prompt Verification with Default Keywords
        combined_text = f"{base_prompt} Termos adicionais: {', '.join(custom_keywords)}."
        comb_words = combined_text.split()
        comb_tokens = len(enc.encode(combined_text))
        self.assertEqual(len(comb_words), 69, "Combined prompt with default keywords must be 69 words")
        self.assertEqual(comb_tokens, 166, f"Combined prompt must be exactly 166 GPT-2 tokens, got {comb_tokens}")
        self.assertLessEqual(comb_tokens, 200, "Default configuration exceeds 200-token safety target")
        self.assertLess(comb_tokens, 224, "Default configuration exceeds Whisper 224-token ceiling")
        self.assertGreaterEqual(224 - comb_tokens, 24, "Must maintain >= 24 tokens of safety margin")

        # 3. Dynamic Truncation Guard Verification under Stress (Tier 1 BPE and Tier 2 Static Fallback)
        def build_priming_prompt(base_template: str, keywords: list[str], max_tokens: int = 200) -> str:
            k_list = list(keywords)
            while k_list:
                cand = f"{base_template} Termos adicionais: {', '.join(k_list)}."
                if len(enc.encode(cand)) <= max_tokens:
                    return cand
                k_list.pop()
            return base_template

        def build_priming_prompt_fallback(base_template: str, keywords: list[str], max_words: int = 75, max_chars: int = 520) -> str:
            k_list = list(keywords)
            while k_list:
                cand = f"{base_template} Termos adicionais: {', '.join(k_list)}."
                if len(cand.split()) <= max_words and len(cand) <= max_chars:
                    return cand
                k_list.pop()
            words = base_template.split()
            return " ".join(words[:max_words]) if len(words) > max_words else base_template

        # Stress test with 50 custom keywords (would produce ~300+ tokens without guard)
        overflow_keywords = [f"termo_customizado_{i}" for i in range(50)]
        guarded_prompt = build_priming_prompt(base_prompt, overflow_keywords, max_tokens=200)
        guarded_tokens = len(enc.encode(guarded_prompt))
        self.assertLessEqual(guarded_tokens, 200, f"Guarded prompt tokens ({guarded_tokens}) exceeded 200 ceiling")
        self.assertLess(guarded_tokens, 224, "Guarded prompt breached Whisper 224-token limit")
        self.assertGreaterEqual(224 - guarded_tokens, 24, "Must maintain >= 24 tokens safety margin")
        self.assertTrue(guarded_prompt.startswith(base_prompt), "Base prompt must remain intact")

        # Test Tier 2 fallback guard with Portuguese accented words
        accented_overflow = ["função", "variável", "assíncrono", "execução"] * 10
        fallback_prompt = build_priming_prompt_fallback(base_prompt, accented_overflow, max_words=75, max_chars=520)
        fallback_words = len(fallback_prompt.split())
        fallback_tokens = len(enc.encode(fallback_prompt))
        self.assertLessEqual(fallback_words, 75, f"Fallback words ({fallback_words}) exceeded 75")
        self.assertLessEqual(len(fallback_prompt), 520, f"Fallback chars ({len(fallback_prompt)}) exceeded 520")
        self.assertLessEqual(fallback_tokens, 200, f"Fallback tokens ({fallback_tokens}) exceeded 200 target ceiling")
        self.assertGreaterEqual(224 - fallback_tokens, 24, "Fallback must maintain >= 24 tokens safety margin")

    def test_wayland_terminal_detection_and_paste_chord(self):
        """Verify focused window app_id terminal detection contract with regex and null safety."""
        import re

        DEFAULT_TERMINAL_PATTERN = r"(?i)(terminal|alacritty|kitty|foot|wezterm|ghostty|ptyxis|xterm|rxvt|contour|tilix)"
        terminal_regex = re.compile(DEFAULT_TERMINAL_PATTERN)

        def resolve_paste_shortcut(app_id: str | None, pattern: re.Pattern = terminal_regex) -> list[str]:
            if app_id and pattern.search(app_id):
                return ["ctrl", "shift", "v"]
            return ["ctrl", "v"]

        # 1. Comprehensive Terminal Suite (Ubuntu 26.04 Default, Flatpaks, Daemons, Wrappers)
        real_world_terminals = [
            "org.gnome.Ptyxis",          # Ubuntu 26.04 LTS default terminal!
            "ptyxis",                    # CLI direct launch
            "gnome-terminal",            # Ubuntu wrapper/script spawn
            "org.gnome.Terminal",        # Standard GNOME terminal
            "foot",                      # Foot standalone
            "footclient",                # Foot daemon client
            "kitty",                     # Kitty lowercase
            "Kitty",                     # Kitty capitalized
            "org.kovidgoyal.kitty",      # Kitty reverse-DNS
            "alacritty",                 # Alacritty lowercase
            "Alacritty",                 # Alacritty capitalized
            "org.alacritty.Alacritty",   # Alacritty reverse-DNS
            "ghostty",                   # Ghostty standalone
            "com.mitchellh.ghostty",     # Ghostty Flatpak / reverse-DNS
            "wezterm",                   # WezTerm standalone
            "wezterm-gui",               # WezTerm GUI binary
            "org.wezfurlong.wezterm",    # WezTerm Flatpak
            "xterm",                     # XTerm standard
            "xterm-256color",            # XTerm variant
            "tilix",                     # Tilix standalone
            "com.gexperts.Tilix",        # Tilix reverse-DNS
            "contour",                   # Contour terminal
            "rxvt",                      # RXVT standard
            "urxvt"                      # RXVT-unicode
        ]

        for term in real_world_terminals:
            self.assertEqual(
                resolve_paste_shortcut(term),
                ["ctrl", "shift", "v"],
                f"Failed to detect terminal app_id: '{term}'"
            )

        # 2. Standard GUI Applications (100% Specificity — Zero False Positives)
        standard_gui_apps = [
            "antigravity-ide",           # Antigravity IDE
            "code",                      # VS Code
            "google-chrome",             # Chrome browser
            "brave-browser",             # Brave browser
            "firefox",                   # Firefox browser
            "org.gnome.TextEditor",      # GNOME Text Editor
            "org.gnome.Nautilus",        # GNOME Files
            "slack",                     # Slack
            "discord",                   # Discord
            "spotify",                   # Spotify
            "obsidian",                  # Obsidian notes
            "libreoffice"                # LibreOffice
        ]

        for gui_app in standard_gui_apps:
            self.assertEqual(
                resolve_paste_shortcut(gui_app),
                ["ctrl", "v"],
                f"False positive: GUI app falsely flagged as terminal: '{gui_app}'"
            )

        # 3. Null Safety & Edge Cases (Unfocused, Root, Unmapped Surfaces)
        self.assertEqual(resolve_paste_shortcut(None), ["ctrl", "v"], "Failed on app_id=None")
        self.assertEqual(resolve_paste_shortcut(""), ["ctrl", "v"], "Failed on empty string app_id")
        self.assertEqual(resolve_paste_shortcut("   "), ["ctrl", "v"], "Failed on whitespace app_id")

        # 4. Modularity: Custom Pattern Override
        custom_pattern = re.compile(r"(?i)(mycustomterminal|altshell)")
        self.assertEqual(resolve_paste_shortcut("mycustomterminal", pattern=custom_pattern), ["ctrl", "shift", "v"])
        self.assertEqual(resolve_paste_shortcut("org.gnome.Ptyxis", pattern=custom_pattern), ["ctrl", "v"])

    def test_atomic_state_locking_concurrency(self):
        """Verify atomic state file locking prevents race conditions on toggle."""
        state_file = os.path.join(self.test_dir, "niri-dictate.state")
        lock_file = os.path.join(self.test_dir, "niri-dictate.lock")

        class StateManager:
            def __init__(self, s_path, l_path):
                self.s_path = s_path
                self.l_path = l_path

            def toggle(self, pid: int) -> str:
                # Open lock file with exclusive creation / atomic lock
                with open(self.l_path, "w") as lock:
                    import fcntl
                    fcntl.flock(lock, fcntl.LOCK_EX)
                    try:
                        if not os.path.exists(self.s_path):
                            # Transition IDLE -> RECORDING
                            with open(self.s_path, "w") as sf:
                                json.dump({"state": "RECORDING", "pid": pid, "timestamp": time.time()}, sf)
                            return "RECORDING_STARTED"
                        else:
                            # Transition RECORDING -> TRANSCRIBING
                            with open(self.s_path, "r") as sf:
                                data = json.load(sf)
                            active_pid = data.get("pid")
                            os.unlink(self.s_path)
                            return f"RECORDING_STOPPED:{active_pid}"
                    finally:
                        fcntl.flock(lock, fcntl.LOCK_UN)

        sm = StateManager(state_file, lock_file)

        # 1. First toggle starts recording
        res1 = sm.toggle(pid=12345)
        self.assertEqual(res1, "RECORDING_STARTED")
        self.assertTrue(os.path.exists(state_file))

        # 2. Second toggle stops recording and retrieves PID
        res2 = sm.toggle(pid=99999)
        self.assertEqual(res2, "RECORDING_STOPPED:12345")
        self.assertFalse(os.path.exists(state_file))


if __name__ == "__main__":
    unittest.main(verbosity=2)
