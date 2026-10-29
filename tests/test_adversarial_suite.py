#!/usr/bin/env python3
"""
Adversarial Stress Test Suite for Niri Voice Dictation Architecture
Empirical Challenger: challenger_m1_2
Validates:
1. Whisper Prompt Token Budget under tiktoken (gpt2 & cl100k_base).
2. Character Encoding & Accents via wl-copy and XKB keysym mapping.
3. Error Boundary Stress: HTTP 401, 429 (Retry-After), 500, 503, Connect/Read Timeouts, Malformed Bodies.
4. Terminal Detection Contract: Exact Match vs. Regex Pattern Matching.
"""

import ctypes
import http.server
import json
import os
import re
import socket
import struct
import subprocess
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
import uuid
import tiktoken


def build_multipart_request(url: str, token: str, audio_bytes: bytes, fields: dict) -> urllib.request.Request:
    """Construct an HTTP multipart/form-data request using urllib."""
    boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
    body = bytearray()
    for name, value in fields.items():
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
        body.extend(f"{value}\r\n".encode("utf-8"))

    body.extend(f"--{boundary}\r\n".encode("utf-8"))
    body.extend(b'Content-Disposition: form-data; name="file"; filename="audio.oga"\r\n')
    body.extend(b"Content-Type: audio/ogg; codecs=opus\r\n\r\n")
    body.extend(audio_bytes)
    body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

    req = urllib.request.Request(url, data=bytes(body), method="POST")
    req.add_header("Authorization", f"Bearer {token}")
    req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
    return req


class AdversarialMockGroqServer(http.server.BaseHTTPRequestHandler):
    """Adversarial HTTP Server capable of simulating extreme edge cases."""

    mode = "normal"  # "normal", "401", "429_int", "429_float", "429_missing", "500", "503", "slow_read", "malformed_utf8", "empty_body", "invalid_json", "html_error"
    delay_read_s = 0.0

    def do_POST(self):
        if self.mode == "401":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Invalid API Key","type":"invalid_request_error"}}')
            return

        if self.mode == "429_int":
            self.send_response(429)
            self.send_header("Retry-After", "5")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Rate limit reached","type":"tokens_exceeded"}}')
            return

        if self.mode == "429_float":
            self.send_response(429)
            self.send_header("Retry-After", "2.5")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Rate limit reached","type":"tokens_exceeded"}}')
            return

        if self.mode == "429_missing":
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Rate limit without Retry-After"}}')
            return

        if self.mode == "500":
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Internal server error"}}')
            return

        if self.mode == "503":
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Service Unavailable"}}')
            return

        if self.mode == "slow_read":
            time.sleep(self.delay_read_s)
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Completed after delay")
            return

        if self.mode == "malformed_utf8":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write(b"Transcription: \xff\xfe\x80 Invalid UTF-8")
            return

        if self.mode == "empty_body":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if self.mode == "invalid_json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{not-valid-json: [}")
            return

        if self.mode == "html_error":
            self.send_response(502)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><head><title>502 Bad Gateway</title></head><body>Cloudflare Error</body></html>")
            return

        # Normal response
        content_length = int(self.headers.get("Content-Length", 0))
        self.rfile.read(content_length)
        payload = b"Transcri\xc3\xa7\xc3\xa3o com acentua\xc3\xa7\xc3\xa3o v\xc3\xa1lida: fun\xc3\xa7\xc3\xa3o, vari\xc3\xa1vel, commit."
        self.send_response(200)
        self.send_header("Content-Type", "text/plain; charset=utf-8")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def log_message(self, format, *args):
        pass


class TestAdversarialChallenges(unittest.TestCase):
    """Empirical challenge suite verifying all 4 challenge scenarios."""

    @classmethod
    def setUpClass(cls):
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), AdversarialMockGroqServer)
        cls.port = cls.server.server_address[1]
        cls.mock_url = f"http://127.0.0.1:{cls.port}/audio/transcriptions"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        AdversarialMockGroqServer.mode = "normal"
        AdversarialMockGroqServer.delay_read_s = 0.0

    # =========================================================================
    # SCENARIO 1: Whisper Prompt Token Budget
    # =========================================================================
    def test_whisper_token_budget_gpt2_and_cl100k(self):
        """Stress test prompt token budget under GPT-2 BPE and CL100K tokenizers."""
        enc_gpt2 = tiktoken.get_encoding("gpt2")
        enc_cl100k = tiktoken.get_encoding("cl100k_base")

        base_prompt = (
            "Código, commits e comandos de programação em Python, JavaScript, TypeScript, Rust, Go, SQL. "
            "Termos técnicos: pull request, merge, branch, commit, rebase, diff, git, docker, kubernetes, "
            "linux, bash, function, const, let, async, await, return, import, export, class, interface, "
            "null, undefined, boolean, string, number, array, object, api, rest, json, postgres, backend, "
            "frontend, devops, deploy, ci/cd, bug, fix, test, mock, endpoint, middleware, refactor."
        )
        custom_keywords = ["antigravity", "ages", "monorepo", "supabase", "trpc", "niri", "wayland"]
        combined_prompt = f"{base_prompt} Termos adicionais: {', '.join(custom_keywords)}."

        # 1. Base prompt measurement
        tokens_gpt2_base = len(enc_gpt2.encode(base_prompt))
        tokens_cl100k_base = len(enc_cl100k.encode(base_prompt))

        # Base prompt is 138 GPT-2 tokens, 123 CL100K tokens (Worker claimed ~88 tokens based on 1.3 ratio)
        self.assertLess(tokens_gpt2_base, 224, "Base prompt exceeds 224 limit")
        self.assertGreater(tokens_gpt2_base, 100, "Base prompt actually requires >100 tokens")

        # 2. Combined prompt measurement
        tokens_gpt2_combined = len(enc_gpt2.encode(combined_prompt))
        tokens_cl100k_combined = len(enc_cl100k.encode(combined_prompt))
        self.assertLess(tokens_gpt2_combined, 224, "Combined prompt exceeds 224 limit")

        # 3. EMPIRICAL VULNERABILITY DEMONSTRATION: Naive 150-word cap
        # The worker's plan proposes: `if len(words) > 150: words = words[:150]`
        # Demonstrating that 150 Portuguese technical words severely violates the 224 token limit!
        sample_words = [
            "código", "função", "variável", "parâmetro", "retorno", "assíncrono", "execução",
            "ambiente", "configuração", "autenticação", "conexão", "requisição", "resposta",
            "exceção", "validação", "condição", "iteração", "integração", "produção", "repositório"
        ] * 8
        naive_capped_text = " ".join(sample_words[:150])
        naive_capped_tokens_gpt2 = len(enc_gpt2.encode(naive_capped_text))
        naive_capped_tokens_cl100k = len(enc_cl100k.encode(naive_capped_text))

        # 150 words produces 540 GPT2 tokens (>2.4x the 224 limit!)
        self.assertGreater(naive_capped_tokens_gpt2, 224, "Naive 150-word cap fails to guard 224 token limit!")
        self.assertGreater(naive_capped_tokens_cl100k, 224, "Naive 150-word cap fails in cl100k as well!")

        # 4. Safe Token Guard Oracle:
        def safe_token_guard(text: str, max_tokens: int = 224, tokenizer=enc_gpt2) -> str:
            tokens = tokenizer.encode(text)
            if len(tokens) <= max_tokens:
                return text
            return tokenizer.decode(tokens[:max_tokens])

        guarded = safe_token_guard(naive_capped_text, max_tokens=200)
        self.assertLessEqual(len(enc_gpt2.encode(guarded)), 200)

    # =========================================================================
    # SCENARIO 2: Character Encoding & Accents via wl-copy and XKB Keysyms
    # =========================================================================
    def test_character_encoding_wl_copy_and_xkb(self):
        """Stress test Portuguese accents and special code symbols."""
        accented_code_samples = [
            "ã é ç ô í õ á ó ú ê",
            'const regex = /^[a-z0-9_\\-\\.]+$/;\nif ($val === "ação") {\n\treturn \'coração\';\n}',
            'def calcular_média(valores: list[float]) -> float:\n    """Função que calcula a média."""\n    return sum(valores) / len(valores)',
            'export const CONFIG = {\n  host: "127.0.0.1",\n  path: "/api/v1/usuários",\n  key: "\\$SECRET_KEY"\n};'
        ]

        # 1. Test wl-copy stdin pipe persistence and exact wl-paste retrieval
        for sample in accented_code_samples:
            proc = subprocess.Popen(
                ["/usr/bin/wl-copy", "--type", "text/plain;charset=utf-8", "--trim-newline"],
                stdin=subprocess.PIPE
            )
            proc.communicate(input=sample.encode("utf-8"))
            self.assertEqual(proc.returncode, 0)

            pasted = subprocess.check_output(["/usr/bin/wl-paste", "--no-newline"]).decode("utf-8")
            self.assertEqual(pasted, sample.rstrip("\r\n"))

        # 2. Test XKB keysym resolution via libxkbcommon
        libxkb = ctypes.CDLL("libxkbcommon.so.0")
        libxkb.xkb_utf32_to_keysym.argtypes = [ctypes.c_uint32]
        libxkb.xkb_utf32_to_keysym.restype = ctypes.c_uint32
        libxkb.xkb_keysym_get_name.argtypes = [ctypes.c_uint32, ctypes.c_char_p, ctypes.c_size_t]
        libxkb.xkb_keysym_get_name.restype = ctypes.c_int

        name_buf = ctypes.create_string_buffer(64)
        expected_keysym_names = {
            "ã": "atilde",
            "é": "eacute",
            "ç": "ccedilla",
            "ô": "ocircumflex",
            "í": "iacute",
            "$": "dollar",
            "\\": "backslash",
            '"': "quotedbl",
            "\n": "Linefeed",
            "\t": "Tab"
        }

        for char, expected_name in expected_keysym_names.items():
            sym = libxkb.xkb_utf32_to_keysym(ord(char))
            self.assertNotEqual(sym, 0, f"Failed to map keysym for {repr(char)}")
            libxkb.xkb_keysym_get_name(sym, name_buf, 64)
            actual_name = name_buf.value.decode("utf-8")
            self.assertEqual(actual_name, expected_name)

    # =========================================================================
    # SCENARIO 3: Error Boundary Stress
    # =========================================================================
    def test_error_boundary_http_codes_and_resilience(self):
        """Test HTTP 401, 429, 500, 503, timeouts, and malformed bodies."""
        dummy_audio = b"RIFF....WAVEfmt ...."
        fields = {"model": "whisper-large-v3-turbo", "language": "pt"}

        # Helper client simulator
        def execute_groq_request(url, mode, delay=0.0, connect_timeout=2.0, read_timeout=8.0):
            AdversarialMockGroqServer.mode = mode
            AdversarialMockGroqServer.delay_read_s = delay
            req = build_multipart_request(url, "test_token", dummy_audio, fields)
            return urllib.request.urlopen(req, timeout=read_timeout)

        # 1. HTTP 401 Unauthorized
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            execute_groq_request(self.mock_url, "401")
        self.assertEqual(ctx.exception.code, 401)
        err_body = json.loads(ctx.exception.read().decode("utf-8"))
        self.assertEqual(err_body["error"]["message"], "Invalid API Key")

        # 2. HTTP 429 Rate Limit with integer Retry-After
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            execute_groq_request(self.mock_url, "429_int")
        self.assertEqual(ctx.exception.code, 429)
        self.assertEqual(ctx.exception.headers.get("Retry-After"), "5")

        # 3. HTTP 429 Rate Limit with float Retry-After
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            execute_groq_request(self.mock_url, "429_float")
        self.assertEqual(ctx.exception.code, 429)
        retry_val = float(ctx.exception.headers.get("Retry-After"))
        self.assertAlmostEqual(retry_val, 2.5)

        # 4. HTTP 429 Rate Limit with MISSING Retry-After header
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            execute_groq_request(self.mock_url, "429_missing")
        self.assertEqual(ctx.exception.code, 429)
        self.assertIsNone(ctx.exception.headers.get("Retry-After"))

        # 5. HTTP 500 Internal Server Error
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            execute_groq_request(self.mock_url, "500")
        self.assertEqual(ctx.exception.code, 500)

        # 6. HTTP 503 Service Unavailable
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            execute_groq_request(self.mock_url, "503")
        self.assertEqual(ctx.exception.code, 503)

        # 7. Connect Timeout (Connect to non-routable IP 10.255.255.1)
        start_t = time.time()
        with self.assertRaises(Exception) as ctx:
            urllib.request.urlopen("http://10.255.255.1:81/stt", timeout=1.0)
        elapsed = time.time() - start_t
        self.assertLess(elapsed, 2.5, f"Connect timeout took too long: {elapsed}s")

        # 8. Read Timeout (Server stalls for 2.0s with client timeout 0.5s)
        with self.assertRaises(socket.timeout):
            execute_groq_request(self.mock_url, "slow_read", delay=2.0, read_timeout=0.5)

        # 9. Malformed UTF-8 in response
        AdversarialMockGroqServer.mode = "malformed_utf8"
        req = build_multipart_request(self.mock_url, "test_token", dummy_audio, fields)
        resp = urllib.request.urlopen(req, timeout=2.0)
        raw_bytes = resp.read()
        # Verify strict UTF-8 decode catches bad bytes
        with self.assertRaises(UnicodeDecodeError):
            raw_bytes.decode("utf-8")
        # Verify surrogateescape or ignore handler provides resilient fallback
        safe_text = raw_bytes.decode("utf-8", errors="replace")
        self.assertIn("Invalid UTF-8", safe_text)

        # 10. Empty body response
        AdversarialMockGroqServer.mode = "empty_body"
        req = build_multipart_request(self.mock_url, "test_token", dummy_audio, fields)
        resp = urllib.request.urlopen(req, timeout=2.0)
        body = resp.read()
        self.assertEqual(len(body), 0)

        # 11. HTML error gateway page on HTTP 502
        with self.assertRaises(urllib.error.HTTPError) as ctx:
            execute_groq_request(self.mock_url, "html_error")
        self.assertEqual(ctx.exception.code, 502)
        html_payload = ctx.exception.read().decode("utf-8")
        self.assertIn("502 Bad Gateway", html_payload)

    # =========================================================================
    # SCENARIO 4: Terminal Detection Contract (Exact Match vs Regex)
    # =========================================================================
    def test_terminal_detection_contract(self):
        """Stress test terminal detection contract with regex and edge cases."""
        # Worker's exact match list from config.toml / test_pipeline_mock.py
        worker_terminal_apps = {
            "alacritty", "kitty", "foot", "org.gnome.Terminal",
            "wezterm", "xterm", "contour", "com.mitchellh.ghostty"
        }

        # Robust case-insensitive terminal pattern
        TERMINAL_REGEX = re.compile(r"(?i)(terminal|alacritty|kitty|foot|wezterm|ghostty|ptyxis|xterm|rxvt|contour|tilix)")

        # Real-world terminal app_id variations on Ubuntu Wayland
        real_world_terminals = [
            ("foot", True),
            ("footclient", True),
            ("kitty", True),
            ("alacritty", True),
            ("Alacritty", True),
            ("gnome-terminal", True),
            ("org.gnome.Terminal", True),
            ("org.gnome.Ptyxis", True),      # Default GNOME terminal on Ubuntu 26.04!
            ("ptyxis", True),
            ("ghostty", True),
            ("com.mitchellh.ghostty", True),
            ("wezterm", True),
            ("org.wezfurlong.wezterm", True),
            ("xterm", True),
            ("xterm-256color", True),
            ("tilix", True),
            ("contour", True)
        ]

        # Standard GUI apps that MUST NOT match terminal chords
        standard_gui_apps = [
            "antigravity-ide", "code", "google-chrome", "brave-browser",
            "firefox", "org.gnome.TextEditor", "org.gnome.Nautilus",
            "slack", "discord", "spotify", "obsidian", "libreoffice"
        ]

        # 1. Demonstrate where worker's exact match fails
        failed_by_worker = []
        for app_id, _ in real_world_terminals:
            if app_id not in worker_terminal_apps:
                failed_by_worker.append(app_id)

        # Worker fails on: ['footclient', 'Alacritty', 'gnome-terminal', 'org.gnome.Ptyxis', 'ptyxis', 'ghostty', 'org.wezfurlong.wezterm', 'xterm-256color', 'tilix']
        self.assertGreater(len(failed_by_worker), 5, f"Worker exact match failed on {len(failed_by_worker)} terminals!")

        # 2. Verify regex matching passes 100% of terminal variants
        for app_id, should_be_terminal in real_world_terminals:
            is_terminal = bool(TERMINAL_REGEX.search(app_id))
            self.assertEqual(is_terminal, should_be_terminal, f"Regex failed to identify terminal: {app_id}")

        # 3. Verify regex matching produces 0 false positives on GUI apps
        for gui_app in standard_gui_apps:
            is_terminal = bool(TERMINAL_REGEX.search(gui_app))
            self.assertFalse(is_terminal, f"Regex falsely flagged GUI app as terminal: {gui_app}")

        # 4. Verify None / Empty app_id handling (e.g. unmapped XWayland surface or root)
        def resolve_chord_safe(app_id: str | None) -> list[str]:
            if app_id and TERMINAL_REGEX.search(app_id):
                return ["ctrl", "shift", "v"]
            return ["ctrl", "v"]

        self.assertEqual(resolve_chord_safe(None), ["ctrl", "v"])
        self.assertEqual(resolve_chord_safe(""), ["ctrl", "v"])
        self.assertEqual(resolve_chord_safe("org.gnome.Ptyxis"), ["ctrl", "shift", "v"])
        self.assertEqual(resolve_chord_safe("antigravity-ide"), ["ctrl", "v"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
