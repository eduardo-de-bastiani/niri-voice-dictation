#!/usr/bin/env python3
"""
Direct Module Unit Tests for niri_dictate Package.
Tests real implementations of:
- config.py
- state.py
- audio.py
- stt.py
- prompt.py
- injector.py
- feedback.py
- cli.py
"""

from __future__ import annotations

import argparse
import http.server
import json
import os
import shutil
import tempfile
import threading
import time
import unittest
import sys
from pathlib import Path
from unittest.mock import MagicMock, patch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tiktoken

from niri_dictate.audio import AudioCaptureError, AudioCaptureManager, EmptyAudioError
from niri_dictate.cli import build_parser, handle_abort, handle_config_check, handle_status
from niri_dictate.config import (
    DEFAULT_PRIMING_PROMPT,
    AppConfig,
    AudioConfig,
    FeedbackConfig,
    GroqConfig,
    InjectionConfig,
    load_config,
)
from niri_dictate.feedback import VisualFeedbackManager
from niri_dictate.injector import WaylandInjector
from niri_dictate.prompt import PrimingPromptBuilder
from niri_dictate.state import HardenedStateManager
from niri_dictate.stt import (
    GroqAuthError,
    GroqNetworkError,
    GroqRateLimitError,
    GroqServerError,
    GroqSTTClient,
    GroqTimeoutError,
    GroqTranscriptionError,
    TimeoutHTTPConnection,
    TimeoutHTTPSConnection,
)


class MockGroqHttpHandler(http.server.BaseHTTPRequestHandler):
    """Local HTTP mock server handler for GroqSTTClient testing."""

    scenario: str = "text"
    request_count: int = 0
    received_headers: dict[str, str] = {}
    received_body: bytes = b""

    def do_POST(self):
        MockGroqHttpHandler.request_count += 1
        MockGroqHttpHandler.received_headers = {k: v for k, v in self.headers.items()}
        length = int(self.headers.get("Content-Length", 0))
        MockGroqHttpHandler.received_body = self.rfile.read(length)

        auth = self.headers.get("Authorization", "")
        if auth != "Bearer valid_key":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Invalid API Key","type":"invalid_request_error"}}')
            return

        if MockGroqHttpHandler.scenario == "text":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.end_headers()
            self.wfile.write("Transcribed texto em português".encode("utf-8"))

        elif MockGroqHttpHandler.scenario == "json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"text": "Transcribed via JSON"}).encode("utf-8"))

        elif MockGroqHttpHandler.scenario == "verbose_json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.end_headers()
            self.wfile.write(json.dumps({"text": "Transcribed verbose JSON", "duration": 1.5}).encode("utf-8"))

        elif MockGroqHttpHandler.scenario == "retry_429_success":
            if MockGroqHttpHandler.request_count == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0.05")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"Rate limit exceeded"}}')
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain; charset=utf-8")
                self.end_headers()
                self.wfile.write(b"Success after retry")

        elif MockGroqHttpHandler.scenario == "retry_429_fail":
            self.send_response(429)
            self.send_header("Retry-After", "3.0")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Rate limit quota exceeded"}}')

        elif MockGroqHttpHandler.scenario == "500_retry_success":
            if MockGroqHttpHandler.request_count == 1:
                self.send_response(500)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Internal Server Error")
            else:
                self.send_response(200)
                self.send_header("Content-Type", "text/plain")
                self.end_headers()
                self.wfile.write(b"Recovered from 500")

        elif MockGroqHttpHandler.scenario == "500_persistent":
            self.send_response(500)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Persistent Internal Server Error")

        elif MockGroqHttpHandler.scenario == "502":
            self.send_response(502)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Bad Gateway")

        elif MockGroqHttpHandler.scenario == "400":
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Invalid multipart payload"}}')

        elif MockGroqHttpHandler.scenario == "read_timeout":
            time.sleep(0.4)
            self.send_response(200)
            self.end_headers()

    def log_message(self, format, *args):
        pass


class TestNiriDictateModules(unittest.TestCase):
    """Direct unit tests for niri_dictate package modules."""

    @classmethod
    def setUpClass(cls):
        cls.stt_server = http.server.HTTPServer(("127.0.0.1", 0), MockGroqHttpHandler)
        cls.stt_port = cls.stt_server.server_address[1]
        cls.stt_server_thread = threading.Thread(target=cls.stt_server.serve_forever, daemon=True)
        cls.stt_server_thread.start()
        cls.stt_base_url = f"http://127.0.0.1:{cls.stt_port}/audio/transcriptions"

    @classmethod
    def tearDownClass(cls):
        cls.stt_server.shutdown()
        cls.stt_server.server_close()

    def reset_stt_handler(self, scenario: str = "text"):
        MockGroqHttpHandler.scenario = scenario
        MockGroqHttpHandler.request_count = 0
        MockGroqHttpHandler.received_headers = {}
        MockGroqHttpHandler.received_body = b""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="niri_mod_test_")
        self.addCleanup(lambda: shutil.rmtree(self.test_dir, ignore_errors=True))

    # --- Config Tests ---
    def test_config_load_defaults(self):
        """Verifies default configuration loads with correct types and values."""
        cfg = load_config(config_path=None)
        self.assertIsInstance(cfg, AppConfig)
        self.assertEqual(cfg.audio.sample_rate, 16000)
        self.assertEqual(cfg.audio.channels, 1)
        self.assertEqual(cfg.audio.format, "opus")
        self.assertEqual(cfg.groq.model, "whisper-large-v3-turbo")
        self.assertEqual(cfg.groq.language, "pt")
        self.assertEqual(cfg.injection.mode, "paste")
        self.assertEqual(cfg.injection.settle_delay_ms, 80)
        self.assertEqual(cfg.feedback.title_recording, "Recording audio...")
        self.assertEqual(cfg.feedback.title_transcribing, "Transcribing...")
        self.assertEqual(cfg.feedback.title_completed, "Completed")
        self.assertEqual(cfg.feedback.title_error, "Dictation Error")

    def test_config_deep_merge_overrides(self):
        """Verifies deep dictionary overrides override values properly."""
        overrides = {
            "audio": {"sample_rate": 48000},
            "groq": {"model": "whisper-large-v3"},
        }
        cfg = load_config(overrides=overrides)
        self.assertEqual(cfg.audio.sample_rate, 48000)
        self.assertEqual(cfg.groq.model, "whisper-large-v3")
        # Other audio settings preserved
        self.assertEqual(cfg.audio.channels, 1)

    def test_config_env_var_api_key_resolution(self):
        """Verifies environment variable resolution for Groq API key."""
        with patch.dict(os.environ, {"GROQ_API_KEY": "gsk_test_env_key_123"}):
            cfg = load_config()
            self.assertEqual(cfg.groq.api_key, "gsk_test_env_key_123")

    # --- Prompt Tests ---
    def test_priming_prompt_token_budget(self):
        """Verifies PrimingPromptBuilder adheres to exact token budget."""
        enc = tiktoken.get_encoding("gpt2")
        builder = PrimingPromptBuilder()
        prompt = builder.build_prompt()
        self.assertEqual(len(enc.encode(prompt)), 138)

        # With default custom keywords
        builder_with_keywords = PrimingPromptBuilder(
            custom_keywords=["antigravity", "ages", "monorepo", "supabase", "trpc", "niri", "wayland"]
        )
        prompt_with_keywords = builder_with_keywords.build_prompt()
        self.assertEqual(len(enc.encode(prompt_with_keywords)), 166)
        self.assertLessEqual(len(enc.encode(prompt_with_keywords)), 200)

    def test_priming_prompt_fallback_trimming(self):
        """Verifies Tier 2 fallback trimming when tokenization is unavailable."""
        builder = PrimingPromptBuilder(
            custom_keywords=["word" + str(i) for i in range(100)],
            max_words=75,
            max_chars=520,
        )
        with patch("tiktoken.get_encoding", side_effect=ImportError("No tiktoken")):
            prompt = builder.build_prompt()
            self.assertLessEqual(len(prompt.split()), 75)
            self.assertLessEqual(len(prompt), 520)

    # --- State Tests ---
    def test_hardened_state_manager_lifecycle(self):
        """Verifies HardenedStateManager transitions IDLE -> RECORDING -> TRANSCRIBING -> IDLE."""
        sm = HardenedStateManager(run_dir=self.test_dir)
        action1 = sm.handle_toggle(exit_on_contention=False)
        self.assertEqual(action1["action"], "START_RECORDING")
        lock_fd1 = action1["lock_fd"]
        sm.release_lock(lock_fd1)

        # Transition to recording
        rec_pid = os.getpid()
        sm.transition_to_recording(lock_fd=None, pid=rec_pid, audio_path=action1["audio_path"])

        # Check status
        with patch.object(sm, "is_pid_valid_recorder", return_value=True):
            status = sm.get_status()
            self.assertEqual(status["state"], "RECORDING")
            self.assertEqual(status["pid"], rec_pid)

            # Toggle while recording with valid recorder -> transitions to TRANSCRIBING
            action2 = sm.handle_toggle(exit_on_contention=False)
            self.assertEqual(action2["action"], "STOP_AND_TRANSCRIBE")
            self.assertEqual(action2["rec_pid"], rec_pid)
            lock_fd2 = action2["lock_fd"]
            sm.release_lock(lock_fd2)

        # Now in TRANSCRIBING: toggle while alive STT returns REJECT_CONCURRENT
        with patch.object(sm, "is_pid_alive", return_value=True):
            action3 = sm.handle_toggle(exit_on_contention=False)
            self.assertEqual(action3["action"], "REJECT_CONCURRENT")
            sm.release_lock(action3["lock_fd"])

        # Clear state
        sm.clear_state()
        self.assertEqual(sm.get_status()["state"], "IDLE")

    def test_state_manager_abort(self):
        """Verifies handle_abort terminates active recording and cleans up."""
        sm = HardenedStateManager(run_dir=self.test_dir)
        sm.transition_to_recording(lock_fd=None, pid=999999, audio_path=Path(self.test_dir) / "test.oga")
        res = sm.handle_abort()
        self.assertEqual(res["action"], "ABORTED")
        self.assertEqual(sm.get_status()["state"], "IDLE")

    # --- Audio Tests ---
    def test_audio_manager_invalid_pid(self):
        """Verifies stop_recording_and_read fails safely on invalid PID."""
        audio_cfg = AudioConfig()
        am = AudioCaptureManager(config=audio_cfg, runtime_dir=self.test_dir)
        with self.assertRaises(AudioCaptureError):
            am.stop_recording_and_read(pid=None, audio_path="dummy.oga")
        with self.assertRaises(AudioCaptureError):
            am.stop_recording_and_read(pid="not_int", audio_path="dummy.oga")

    def test_audio_manager_empty_audio_raises(self):
        """Verifies EmptyAudioError raised when recorded file has 0 bytes."""
        audio_cfg = AudioConfig()
        am = AudioCaptureManager(config=audio_cfg, runtime_dir=self.test_dir)
        empty_file = Path(self.test_dir) / "empty.oga"
        empty_file.touch()
        with patch("os.kill", return_value=None):
            with self.assertRaises(EmptyAudioError):
                am.stop_recording_and_read(pid=1234, audio_path=empty_file)

    # --- Injector Tests ---
    def test_wayland_injector_terminal_detection(self):
        """Verifies WaylandInjector terminal regex matching and resolution."""
        inj_cfg = InjectionConfig()
        injector = WaylandInjector(inj_cfg)

        # Terminals match regex
        self.assertTrue(injector.terminal_pattern.search("org.gnome.Ptyxis"))
        self.assertTrue(injector.terminal_pattern.search("alacritty"))
        self.assertTrue(injector.terminal_pattern.search("footclient"))
        self.assertTrue(injector.terminal_pattern.search("kitty"))
        self.assertTrue(injector.terminal_pattern.search("terminator"))
        self.assertTrue(injector.terminal_pattern.search("konsole"))
        self.assertTrue(injector.terminal_pattern.search("org.kde.konsole"))

        # GUI apps do not match regex
        self.assertFalse(injector.terminal_pattern.search("antigravity-ide"))
        self.assertFalse(injector.terminal_pattern.search("code"))
        self.assertFalse(injector.terminal_pattern.search("firefox"))

    def test_wayland_injector_fallback_mode_resolution(self):
        """Verifies graceful fallback to clipboard_only when wtype is missing."""
        inj_cfg = InjectionConfig(wtype_bin="/nonexistent/wtype")
        injector = WaylandInjector(inj_cfg)
        mode, is_fallback = injector.resolve_effective_mode()
        self.assertEqual(mode, "clipboard_only")
        self.assertTrue(is_fallback)

    # --- Feedback Tests ---
    def test_feedback_manager_silent_provider(self):
        """Verifies provider='none' produces no actions and returns None."""
        feed_cfg = FeedbackConfig(provider="none")
        fm = VisualFeedbackManager(feed_cfg)
        notif_id = fm.show_recording()
        self.assertIsNone(notif_id)
        # These should run without error
        fm.show_transcribing(1)
        fm.show_completed(1, typed=True)
        fm.show_error(1, message="test error")

    # --- CLI Subcommand Tests ---
    def test_cli_parser_subcommands(self):
        """Verifies CLI argument parser recognizes all 4 subcommands."""
        parser = build_parser()
        args_toggle = parser.parse_args(["toggle"])
        self.assertEqual(args_toggle.subcommand, "toggle")

        args_abort = parser.parse_args(["abort"])
        self.assertEqual(args_abort.subcommand, "abort")

        args_status = parser.parse_args(["status", "--json"])
        self.assertEqual(args_status.subcommand, "status")
        self.assertTrue(args_status.json)

        args_cfg = parser.parse_args(["config-check"])
        self.assertEqual(args_cfg.subcommand, "config-check")

    def test_cli_handle_status_idle(self):
        """Verifies handle_status reports IDLE state cleanly."""
        cfg = load_config(overrides={"general": {"runtime_dir": self.test_dir}})
        args = argparse.Namespace(json=True)
        ret = handle_status(args, cfg)
        self.assertEqual(ret, 0)

    # --- Groq STT Client Direct Unit Tests ---
    def test_stt_init_missing_api_key_raises(self):
        """Verifies GroqSTTClient raises GroqAuthError when API key is missing or empty."""
        cfg = GroqConfig()
        with self.assertRaises(GroqAuthError):
            GroqSTTClient(cfg, api_key="")
        with self.assertRaises(GroqAuthError):
            GroqSTTClient(cfg, api_key="   ")

    def test_stt_empty_audio_returns_empty_string(self):
        """Verifies transcribe(b'') returns empty string immediately without network request."""
        self.reset_stt_handler("text")
        cfg = GroqConfig(api_url=self.stt_base_url)
        client = GroqSTTClient(cfg, api_key="valid_key")
        res = client.transcribe(b"")
        self.assertEqual(res, "")
        self.assertEqual(MockGroqHttpHandler.request_count, 0)

    def test_stt_transcribe_success_text(self):
        """Verifies successful text transcription and validates headers and multipart body."""
        self.reset_stt_handler("text")
        cfg = GroqConfig(api_url=self.stt_base_url, response_format="text")
        client = GroqSTTClient(cfg, api_key="valid_key")
        result = client.transcribe(b"fake audio data", prompt="test prompt")

        self.assertEqual(result, "Transcribed texto em português")
        self.assertEqual(MockGroqHttpHandler.request_count, 1)
        self.assertEqual(
            MockGroqHttpHandler.received_headers.get("Authorization"), "Bearer valid_key"
        )
        self.assertIn("multipart/form-data; boundary=", MockGroqHttpHandler.received_headers.get("Content-Type", ""))
        self.assertIn(b'name="model"\r\n\r\nwhisper-large-v3-turbo', MockGroqHttpHandler.received_body)
        self.assertIn(b'name="prompt"\r\n\r\ntest prompt', MockGroqHttpHandler.received_body)
        self.assertIn(b'filename="audio.ogg"', MockGroqHttpHandler.received_body)

    def test_stt_transcribe_success_json(self):
        """Verifies successful transcription with JSON response parsing."""
        self.reset_stt_handler("json")
        cfg = GroqConfig(api_url=self.stt_base_url, response_format="json")
        client = GroqSTTClient(cfg, api_key="valid_key")
        result = client.transcribe(b"fake audio data")
        self.assertEqual(result, "Transcribed via JSON")

    def test_stt_transcribe_success_verbose_json(self):
        """Verifies successful transcription with verbose_json format parsing."""
        self.reset_stt_handler("verbose_json")
        cfg = GroqConfig(api_url=self.stt_base_url, response_format="verbose_json")
        client = GroqSTTClient(cfg, api_key="valid_key")
        result = client.transcribe(b"fake audio data")
        self.assertEqual(result, "Transcribed verbose JSON")

    def test_stt_transcribe_auth_failure_401(self):
        """Verifies HTTP 401 raises GroqAuthError."""
        self.reset_stt_handler("text")
        cfg = GroqConfig(api_url=self.stt_base_url)
        client = GroqSTTClient(cfg, api_key="wrong_key")
        with self.assertRaises(GroqAuthError) as ctx:
            client.transcribe(b"fake audio data")
        self.assertIn("Invalid or missing Groq API Key", str(ctx.exception))

    def test_stt_transcribe_rate_limit_retry_success_429(self):
        """Verifies HTTP 429 with short Retry-After (<=1.0s) retries automatically and succeeds."""
        self.reset_stt_handler("retry_429_success")
        cfg = GroqConfig(api_url=self.stt_base_url)
        client = GroqSTTClient(cfg, api_key="valid_key")
        result = client.transcribe(b"fake audio data")
        self.assertEqual(result, "Success after retry")
        self.assertEqual(MockGroqHttpHandler.request_count, 2)

    def test_stt_transcribe_rate_limit_exceeded_429(self):
        """Verifies HTTP 429 with long Retry-After (>1.0s) raises GroqRateLimitError immediately."""
        self.reset_stt_handler("retry_429_fail")
        cfg = GroqConfig(api_url=self.stt_base_url)
        client = GroqSTTClient(cfg, api_key="valid_key")
        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"fake audio data")
        self.assertEqual(ctx.exception.retry_after, 3.0)
        self.assertIn("please retry in 3 seconds", str(ctx.exception))
        self.assertEqual(MockGroqHttpHandler.request_count, 1)

    def test_stt_transcribe_server_error_500_retry_success(self):
        """Verifies transient HTTP 500 error retries once after 0.5s backoff and succeeds."""
        self.reset_stt_handler("500_retry_success")
        cfg = GroqConfig(api_url=self.stt_base_url)
        client = GroqSTTClient(cfg, api_key="valid_key")
        result = client.transcribe(b"fake audio data")
        self.assertEqual(result, "Recovered from 500")
        self.assertEqual(MockGroqHttpHandler.request_count, 2)

    def test_stt_transcribe_server_error_500_persistent(self):
        """Verifies persistent HTTP 500 error raises GroqServerError after retry."""
        self.reset_stt_handler("500_persistent")
        cfg = GroqConfig(api_url=self.stt_base_url)
        client = GroqSTTClient(cfg, api_key="valid_key")
        with self.assertRaises(GroqServerError) as ctx:
            client.transcribe(b"fake audio data")
        self.assertIn("HTTP 500", str(ctx.exception))
        self.assertEqual(MockGroqHttpHandler.request_count, 2)

    def test_stt_transcribe_gateway_error_502(self):
        """Verifies HTTP 502 Bad Gateway raises GroqServerError immediately without retry."""
        self.reset_stt_handler("502")
        cfg = GroqConfig(api_url=self.stt_base_url)
        client = GroqSTTClient(cfg, api_key="valid_key")
        with self.assertRaises(GroqServerError) as ctx:
            client.transcribe(b"fake audio data")
        self.assertIn("502 Bad Gateway", str(ctx.exception))
        self.assertEqual(MockGroqHttpHandler.request_count, 1)

    def test_stt_transcribe_client_error_400(self):
        """Verifies HTTP 400 Bad Request raises GroqTranscriptionError."""
        self.reset_stt_handler("400")
        cfg = GroqConfig(api_url=self.stt_base_url)
        client = GroqSTTClient(cfg, api_key="valid_key")
        with self.assertRaises(GroqTranscriptionError) as ctx:
            client.transcribe(b"fake audio data")
        self.assertIn("HTTP 400", str(ctx.exception))

    def test_stt_transcribe_read_timeout(self):
        """Verifies read timeout raises GroqTimeoutError."""
        self.reset_stt_handler("read_timeout")
        cfg = GroqConfig(api_url=self.stt_base_url, connect_timeout_seconds=0.2, read_timeout_seconds=0.1)
        client = GroqSTTClient(cfg, api_key="valid_key")
        with self.assertRaises(GroqTimeoutError) as ctx:
            client.transcribe(b"fake audio data")
        self.assertIn("timed out after 0.1s", str(ctx.exception))

    def test_stt_transcribe_connect_timeout(self):
        """Verifies connection timeout to unreachable endpoint raises GroqTimeoutError."""
        cfg = GroqConfig(api_url="http://10.255.255.1:80", connect_timeout_seconds=0.1, read_timeout_seconds=0.2)
        client = GroqSTTClient(cfg, api_key="valid_key")
        with self.assertRaises(GroqTimeoutError) as ctx:
            client.transcribe(b"fake audio data")
        self.assertIn("connection timed out after 0.1s", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
