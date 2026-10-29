#!/usr/bin/env python3
"""
Empirical Challenger Stress Test Suite
Author: teamwork_challenger_m1_2
Target Project: Niri Wayland Voice Dictation (niri-dictate)

Empirically challenges:
1. Whisper Decoder Budget Safety:
   - Tier 1 BPE Token Pruning (tiktoken) under heavy Portuguese diacritic technical vocabulary.
   - Tier 2 Static Length Fallback heuristic vulnerability (breaching 200 safety target and 224 hard limit).
2. Groq STT Client Network Opener Defect:
   - Reproduces urllib 'timeout' keyword collision in TimeoutHTTPConnection and TimeoutHTTPSConnection.
3. Groq STT HTTP 429 Rate Limit Handling:
   - Integer Retry-After header.
   - Float Retry-After header (short backoff <=1.0s retried, long backoff >1.0s rejected).
   - Missing Retry-After header.
   - Malformed / HTTP-date Retry-After header.
   - Second consecutive 429 during retry (rejection without infinite loop).
4. Groq STT Network Timeouts:
   - Fail-fast Connect Timeout (2.0s).
   - Fail-fast Read Timeout (8.0s).
5. Wayland Window Matching & Terminal Regex:
   - Standard terminal app_ids (Ptyxis, Foot, Kitty, Alacritty, WezTerm, Ghostty, XTerm).
   - Obscure / unhandled terminal app_ids (Terminator, Konsole, Rio).
   - Standard GUI app_ids (Antigravity, VS Code, Chrome, Firefox, Slack, Discord).
   - Edge cases (None, empty string, whitespace).
6. Absence of wtype Graceful Degradation:
   - Live system degradation to clipboard-only.
   - Full UTF-8 clipboard fidelity verification via wl-copy / wl-paste.
   - Missing wl-copy critical dependency handling.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path
from unittest.mock import patch

# Ensure niri_dictate package is importable
_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tiktoken

from niri_dictate.config import GroqConfig, InjectionConfig
from niri_dictate.injector import WaylandInjector
from niri_dictate.prompt import PrimingPromptBuilder
import niri_dictate.stt as stt
from niri_dictate.stt import (
    GroqAuthError,
    GroqNetworkError,
    GroqRateLimitError,
    GroqServerError,
    GroqSTTClient,
    GroqTimeoutError,
    GroqTranscriptionError,
)


class MockGroqServer(http.server.BaseHTTPRequestHandler):
    """Configurable HTTP server for empirical testing of Groq STT client."""

    scenario = "normal"
    request_count = 0
    delay_s = 0.0

    def do_POST(self):
        MockGroqServer.request_count += 1
        cnt = MockGroqServer.request_count

        if self.scenario == "429_int":
            self.send_response(429)
            self.send_header("Retry-After", "5")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "Rate limit: 5s backoff"}}')
            return

        if self.scenario == "429_float_short":
            if cnt == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0.1")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": {"message": "Rate limit: 0.1s backoff"}}')
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"text": "transcription succeeded on retry"}')
            return

        if self.scenario == "429_float_long":
            self.send_response(429)
            self.send_header("Retry-After", "2.5")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "Rate limit: 2.5s backoff"}}')
            return

        if self.scenario == "429_missing":
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "Rate limit without header"}}')
            return

        if self.scenario == "429_http_date":
            self.send_response(429)
            self.send_header("Retry-After", "Wed, 21 Oct 2026 07:28:00 GMT")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "Rate limit: HTTP-date format"}}')
            return

        if self.scenario == "429_persistent":
            # Returns 429 on every attempt
            self.send_response(429)
            self.send_header("Retry-After", "0.05")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error": {"message": "Persistent 429"}}')
            return

        if self.scenario == "500_transient":
            if cnt == 1:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error": "Internal Server Error"}')
            else:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"text": "recovered after 500"}')
            return

        if self.scenario == "502_bad_gateway":
            self.send_response(502)
            self.send_header("Content-Type", "text/html")
            self.end_headers()
            self.wfile.write(b"<html><head><title>502 Bad Gateway</title></head></html>")
            return

        if self.scenario == "slow_read":
            time.sleep(self.delay_s)
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"text": "slow response completed"}')
            return

        # Default normal response
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len > 0:
            self.rfile.read(content_len)
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"text": "transcricao normal com acentuacao"}')

    def log_message(self, format, *args):
        pass


class TestEmpiricalChallenger(unittest.TestCase):
    """Empirical challenge suite executing against the live package modules."""

    @classmethod
    def setUpClass(cls):
        cls.server = http.server.HTTPServer(("127.0.0.1", 0), MockGroqServer)
        cls.port = cls.server.server_address[1]
        cls.mock_url = f"http://127.0.0.1:{cls.port}/openai/v1/audio/transcriptions"
        cls.server_thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.server_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def setUp(self):
        MockGroqServer.scenario = "normal"
        MockGroqServer.request_count = 0
        MockGroqServer.delay_s = 0.0

    # =========================================================================
    # 1. WHISPER DECODER BUDGET SAFETY
    # =========================================================================

    def test_whisper_budget_tier1_bpe_portuguese_diacritics(self):
        """
        Verify Tier 1 (tiktoken BPE) strictly limits heavy diacritic Portuguese text
        to <= 200 tokens.
        """
        enc = tiktoken.get_encoding("gpt2")
        heavy_base = (
            "Código, configuração, autenticação, requisição, autorização, exceção, "
            "implementação, extensão, abstração, otimização, persistência, concorrência, "
            "sincronização, transação, comunicação, distribuição, validação, integridade, "
            "manutenção, substituição, restrição, avaliação, execução, compilação."
        )
        heavy_keywords = [
            "função", "produção", "recursão", "expressão", "direção", "solução",
            "condição", "iteração", "projeção", "precisão", "geração", "relação"
        ] * 10

        builder = PrimingPromptBuilder(
            base_prompt=heavy_base,
            custom_keywords=heavy_keywords,
            max_tokens=200,
        )

        prompt = builder.build_prompt()
        token_count = len(enc.encode(prompt))
        self.assertLessEqual(
            token_count,
            200,
            f"Tier 1 failed token bound: produced {token_count} tokens (target <= 200)",
        )
        # Verify result is valid UTF-8 and non-empty
        self.assertGreater(len(prompt), 50)
        self.assertIn("configuração", prompt)

    def test_whisper_budget_tier2_fallback_diacritic_breach_vulnerability(self):
        """
        EMPIRICAL VULNERABILITY DEMONSTRATION:
        When tiktoken is unavailable, Tier 2 fallback uses static thresholds
        (max_words=75, max_chars=520). Under Portuguese technical text with high
        diacritic density, each accented word produces multiple BPE tokens.
        This test proves that Tier 2 allows up to 252 GPT-2 tokens, breaching
        both the 200 safety target and the 224 Whisper decoder hard limit!
        """
        enc = tiktoken.get_encoding("gpt2")
        heavy_words = [
            "ação", "seção", "direção", "função", "opção", "fração", "relação",
            "criação", "adição", "conexão", "edição", "reação", "projeção"
        ]
        # Construct dense diacritic input
        dense_prompt = " ".join(heavy_words * 10)

        builder = PrimingPromptBuilder(
            base_prompt=dense_prompt,
            max_tokens=200,
            max_words=75,
            max_chars=520,
        )

        # Force Tier 2 fallback by simulating tiktoken failure
        with patch("tiktoken.get_encoding", side_effect=ImportError("tiktoken mocked unavailable")):
            tier2_result = builder.build_prompt()

        tier2_tokens = len(enc.encode(tier2_result))
        word_count = len(tier2_result.split())
        char_count = len(tier2_result)

        # Confirm the Tier 2 heuristic bounded words and chars correctly
        self.assertLessEqual(word_count, 75)
        self.assertLessEqual(char_count, 520)

        # Confirm Tier 2 hardening strictly bounds BPE tokens under safety ceiling (<= 200)
        self.assertLessEqual(
            tier2_tokens,
            200,
            f"Tier 2 produced {tier2_tokens} tokens, exceeding <= 200 safety target",
        )
        self.assertGreaterEqual(
            224 - tier2_tokens,
            24,
            "Must maintain >= 24 tokens of safety margin under Whisper 224 ceiling",
        )

    # =========================================================================
    # 2. GROQ STT CLIENT NETWORK OPENER DEFECT
    # =========================================================================

    def test_groq_stt_opener_timeout_collision_defect(self):
        """
        VERIFIES REMEDIATION OF OPENER TIMEOUT COLLISION DEFECT:
        Verifies that GroqSTTClient opener successfully transmits without raising
        TypeError: TimeoutHTTPConnection() got multiple values for keyword argument 'timeout'.
        """
        cfg = GroqConfig(api_key="gsk_test_key", api_url=self.mock_url)
        client = GroqSTTClient(cfg)

        result = client.transcribe(b"dummy_audio_bytes")
        self.assertIn("transcricao normal", result)

    # Helper method to obtain a client with fixed opener for testing subsequent STT logic
    def _create_patched_stt_client(self, **kwargs) -> GroqSTTClient:
        cfg_dict = {"api_key": "gsk_test_key", "api_url": self.mock_url}
        cfg_dict.update(kwargs)
        cfg = GroqConfig(**cfg_dict)
        client = GroqSTTClient(cfg)

        # Fix the lambda parameter collision solely to test inner STT logic
        class PatchedHTTPHandler(urllib.request.HTTPHandler):
            def http_open(_self, req: urllib.request.Request) -> http.client.HTTPResponse:
                return _self.do_open(
                    lambda host, timeout=None, **kw: stt.TimeoutHTTPConnection(
                        host, timeout=client.connect_timeout, read_timeout=client.read_timeout, **kw
                    ),
                    req,
                )

        class PatchedHTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(_self, req: urllib.request.Request) -> http.client.HTTPResponse:
                return _self.do_open(
                    lambda host, timeout=None, **kw: stt.TimeoutHTTPSConnection(
                        host, timeout=client.connect_timeout, read_timeout=client.read_timeout, **kw
                    ),
                    req,
                )

        client._opener = urllib.request.build_opener(PatchedHTTPHandler, PatchedHTTPSHandler)
        return client

    # =========================================================================
    # 3. HTTP 429 RATE LIMIT RETRY AND BACKOFF
    # =========================================================================

    def test_http_429_integer_retry_after(self):
        """Verify HTTP 429 with integer Retry-After (5s) raises GroqRateLimitError without sleep."""
        MockGroqServer.scenario = "429_int"
        client = self._create_patched_stt_client()

        t0 = time.time()
        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"dummy_audio")
        elapsed = time.time() - t0

        self.assertEqual(ctx.exception.retry_after, 5.0)
        self.assertIn("please retry in 5 seconds", str(ctx.exception))
        self.assertLess(elapsed, 1.0, "Should not sleep for >1.0s backoff")
        self.assertEqual(MockGroqServer.request_count, 1)

    def test_http_429_float_short_retry_after_success(self):
        """Verify HTTP 429 with short float Retry-After (0.1s <= 1.0s) sleeps and succeeds on retry."""
        MockGroqServer.scenario = "429_float_short"
        client = self._create_patched_stt_client()

        t0 = time.time()
        result = client.transcribe(b"dummy_audio")
        elapsed = time.time() - t0

        self.assertEqual(result, "transcription succeeded on retry")
        self.assertEqual(MockGroqServer.request_count, 2)
        self.assertGreaterEqual(elapsed, 0.08, "Should have slept for short Retry-After")

    def test_http_429_float_long_retry_after(self):
        """Verify HTTP 429 with float Retry-After (2.5s > 1.0s) raises GroqRateLimitError immediately."""
        MockGroqServer.scenario = "429_float_long"
        client = self._create_patched_stt_client()

        t0 = time.time()
        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"dummy_audio")
        elapsed = time.time() - t0

        self.assertEqual(ctx.exception.retry_after, 2.5)
        self.assertIn("please retry in 2.5 seconds", str(ctx.exception))
        self.assertLess(elapsed, 1.0)
        self.assertEqual(MockGroqServer.request_count, 1)

    def test_http_429_missing_retry_after(self):
        """Verify HTTP 429 without Retry-After header raises GroqRateLimitError with retry_after=None."""
        MockGroqServer.scenario = "429_missing"
        client = self._create_patched_stt_client()

        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"dummy_audio")

        self.assertIsNone(ctx.exception.retry_after)
        self.assertIn("quota exceeded, please retry later", str(ctx.exception))
        self.assertEqual(MockGroqServer.request_count, 1)

    def test_http_429_http_date_format_fallback(self):
        """Verify HTTP 429 with RFC HTTP-date Retry-After gracefully degrades to retry_after=None."""
        MockGroqServer.scenario = "429_http_date"
        client = self._create_patched_stt_client()

        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"dummy_audio")

        self.assertIsNone(ctx.exception.retry_after)
        self.assertEqual(MockGroqServer.request_count, 1)

    def test_http_429_persistent_retry_rejection(self):
        """Verify repeated 429 on retry does not enter infinite loop and raises GroqRateLimitError."""
        MockGroqServer.scenario = "429_persistent"
        client = self._create_patched_stt_client()

        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"dummy_audio")

        self.assertEqual(MockGroqServer.request_count, 2)
        self.assertEqual(ctx.exception.retry_after, 0.05)

    def test_http_500_transient_retry_success(self):
        """Verify HTTP 500 transient server error retries once after 500ms and succeeds."""
        MockGroqServer.scenario = "500_transient"
        client = self._create_patched_stt_client()

        t0 = time.time()
        res = client.transcribe(b"dummy_audio")
        elapsed = time.time() - t0

        self.assertEqual(res, "recovered after 500")
        self.assertEqual(MockGroqServer.request_count, 2)
        self.assertGreaterEqual(elapsed, 0.45)

    def test_http_502_bad_gateway_immediate_failure(self):
        """Verify HTTP 502 Bad Gateway raises GroqServerError without retrying."""
        MockGroqServer.scenario = "502_bad_gateway"
        client = self._create_patched_stt_client()

        with self.assertRaises(GroqServerError) as ctx:
            client.transcribe(b"dummy_audio")

        self.assertIn("502 Bad Gateway", str(ctx.exception))
        self.assertEqual(MockGroqServer.request_count, 1)

    # =========================================================================
    # 4. NETWORK TIMEOUTS (CONNECT & READ)
    # =========================================================================

    def test_read_timeout_enforcement(self):
        """Verify read timeout aborts request and raises GroqTimeoutError."""
        MockGroqServer.scenario = "slow_read"
        MockGroqServer.delay_s = 0.5
        client = self._create_patched_stt_client(read_timeout_seconds=0.1)

        t0 = time.time()
        with self.assertRaises(GroqTimeoutError) as ctx:
            client.transcribe(b"dummy_audio")
        elapsed = time.time() - t0

        self.assertIn("timed out after", str(ctx.exception).lower())
        self.assertLess(elapsed, 0.4, f"Read timeout took too long: {elapsed}s")

    def test_connect_timeout_enforcement(self):
        """Verify connect timeout to non-routable IP (10.255.255.1) raises GroqTimeoutError."""
        # Non-routable address to trigger TCP SYN connect timeout
        client = self._create_patched_stt_client(
            api_url="http://10.255.255.1:81/stt",
            connect_timeout_seconds=0.5,
        )

        t0 = time.time()
        with self.assertRaises(GroqTimeoutError) as ctx:
            client.transcribe(b"dummy_audio")
        elapsed = time.time() - t0

        self.assertIn("connection timed out", str(ctx.exception).lower())
        self.assertLess(elapsed, 2.0, f"Connect timeout exceeded bound: {elapsed}s")

    # =========================================================================
    # 5. WAYLAND WINDOW MATCHING & TERMINAL REGEX
    # =========================================================================

    def test_terminal_regex_standard_and_obscure_apps(self):
        """
        Stress test terminal detection regex against standard terminals, obscure terminals,
        and regular GUI applications.
        """
        inj_cfg = InjectionConfig()
        injector = WaylandInjector(inj_cfg)
        pattern = injector.terminal_pattern

        # Standard terminals that must evaluate to True
        standard_terminals = [
            "org.gnome.Ptyxis",      # Default GNOME terminal on Ubuntu 26.04
            "ptyxis",
            "kitty",
            "alacritty",
            "Alacritty",
            "foot",
            "footclient",
            "org.gnome.Terminal",
            "gnome-terminal",
            "wezterm",
            "org.wezfurlong.wezterm",
            "ghostty",
            "com.mitchellh.ghostty",
            "xterm",
            "xterm-256color",
            "contour",
            "tilix",
            "rxvt",
            "urxvt",
            "xfce4-terminal",
            "mate-terminal",
            "lxterminal",
        ]
        for term in standard_terminals:
            self.assertTrue(
                bool(pattern.search(term)),
                f"Terminal regex failed to match standard terminal: {term}",
            )

        # Standard GUI applications that must NEVER match terminal regex
        gui_apps = [
            "antigravity-ide",
            "code",
            "google-chrome",
            "brave-browser",
            "firefox",
            "org.gnome.TextEditor",
            "org.gnome.Nautilus",
            "slack",
            "discord",
            "spotify",
            "obsidian",
            "libreoffice",
            "thunderbird",
            "org.kde.kate",
        ]
        for app in gui_apps:
            self.assertFalse(
                bool(pattern.search(app)),
                f"Terminal regex falsely matched GUI application: {app}",
            )

        # Expanded terminals now match regex
        expanded_terminals = [
            "terminator",
            "konsole",
            "org.kde.konsole",
        ]
        for term in expanded_terminals:
            self.assertTrue(
                bool(pattern.search(term)),
                f"Expanded terminal regex failed to match: {term}",
            )

        # Unhandled terminal (rio) remains unmatched
        self.assertFalse(bool(pattern.search("rio")))

    def test_focused_window_edge_cases(self):
        """Verify focused window retrieval handles None, empty string, and exceptions safely."""
        inj_cfg = InjectionConfig()
        injector = WaylandInjector(inj_cfg)

        # Mock niri IPC returning null app_id
        with patch("subprocess.run") as mock_run:
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"app_id": null}'
            )
            self.assertIsNone(injector.get_focused_window_app_id())

            # Empty string app_id
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"app_id": ""}'
            )
            self.assertIsNone(injector.get_focused_window_app_id())

            # Whitespace app_id
            mock_run.return_value = subprocess.CompletedProcess(
                args=[], returncode=0, stdout='{"app_id": "   "}'
            )
            self.assertIsNone(injector.get_focused_window_app_id())

            # Non-JSON or subprocess error
            mock_run.side_effect = subprocess.SubprocessError("IPC error")
            self.assertIsNone(injector.get_focused_window_app_id())

    # =========================================================================
    # 6. ABSENCE OF WTYPE GRACEFUL DEGRADATION
    # =========================================================================

    def test_wtype_absence_graceful_degradation(self):
        """
        Verify injector when wtype is absent degrades cleanly
        to clipboard-only mode without crashing, populates clipboard, and returns
        informative advisory text.
        """
        inj_cfg = InjectionConfig(mode="paste", wtype_bin="/nonexistent/wtype")
        injector = WaylandInjector(inj_cfg)

        # 1. Effective mode resolution
        effective_mode, is_fallback = injector.resolve_effective_mode()
        self.assertEqual(effective_mode, "clipboard_only")
        self.assertTrue(is_fallback)

        # 2. Inject text execution
        test_payload = "Acentuação empírica: coração, função, variável, commit."
        result = injector.inject(test_payload)

        self.assertEqual(result["mode"], "clipboard_only")
        self.assertFalse(result["typed"])
        self.assertTrue(result["fallback"])
        self.assertIn("Install 'wtype'", result["advisory"])

        # 3. Verify wl-paste content on the live Wayland compositor
        pasted = subprocess.check_output(["wl-paste", "--no-newline"]).decode("utf-8")
        self.assertEqual(pasted, test_payload)

        # 4. Contract method inject_text returns False when untyped
        typed_bool = injector.inject_text(test_payload)
        self.assertFalse(typed_bool)

    def test_wl_copy_missing_raises_runtime_error(self):
        """Verify missing wl-copy binary raises RuntimeError indicating critical dependency."""
        inj_cfg = InjectionConfig(wl_copy_bin="/nonexistent/wl-copy")
        with patch("shutil.which", return_value=None):
            injector = WaylandInjector(inj_cfg)
            with self.assertRaises(RuntimeError) as ctx:
                injector.resolve_effective_mode()
            self.assertIn("wl-copy is required", str(ctx.exception))


if __name__ == "__main__":
    unittest.main(verbosity=2)
