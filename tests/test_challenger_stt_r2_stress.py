#!/usr/bin/env python3
"""
Empirical Challenger Round 2 Stress Test Suite for Groq STT Client.
Author: teamwork_challenger_r2_1
Target: niri_dictate/stt.py (GroqSTTClient)

Empirically tests and verifies:
1. Real execution against mock HTTP and HTTPS endpoints.
2. Complete absence of TypeError on connection creation across HTTP and HTTPS.
3. Dual timeouts: connect timeout (connect-bound) vs read timeout (read-bound).
4. HTTP 429 Retry-After backoff (integers, floats, missing header, date formats, persistent limits).
5. HTTP 500/503 retry behavior (transient vs persistent) and HTTP 502 HTML error safety.
6. UTF-8 safety (errors='replace') and empty body / whitespace handling.
7. Authorization, network error handling, and parameter edge cases.
"""

from __future__ import annotations

import http.server
import json
import os
import shutil
import socket
import ssl
import subprocess
import sys
import tempfile
import threading
import time
import unittest
import urllib.error
import urllib.request
from pathlib import Path

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from niri_dictate.config import GroqConfig
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


class EmpiricalMockGroqServer(http.server.BaseHTTPRequestHandler):
    """
    Stateful multi-scenario HTTP/HTTPS mock server for stress testing GroqSTTClient.
    """

    scenario: str = "success_text"
    request_count: int = 0
    received_headers: dict[str, str] = {}
    received_body: bytes = b""
    delay_s: float = 0.0

    def do_POST(self):
        EmpiricalMockGroqServer.request_count += 1
        cnt = EmpiricalMockGroqServer.request_count
        EmpiricalMockGroqServer.received_headers = {k: v for k, v in self.headers.items()}
        content_len = int(self.headers.get("Content-Length", 0))
        if content_len > 0:
            EmpiricalMockGroqServer.received_body = self.rfile.read(content_len)
        else:
            EmpiricalMockGroqServer.received_body = b""

        sc = EmpiricalMockGroqServer.scenario

        # -------------------------------------------------------------
        # Scenarios for Standard Execution and Formats
        # -------------------------------------------------------------
        if sc == "success_text":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain; charset=utf-8")
            self.send_header("Content-Length", str(len("Transcrição em português de teste".encode("utf-8"))))
            self.end_headers()
            self.wfile.write("Transcrição em português de teste".encode("utf-8"))
            return

        if sc == "success_json":
            body = json.dumps({"text": "Transcrição via JSON"}).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if sc == "success_verbose_json":
            body = json.dumps({
                "task": "transcribe",
                "language": "portuguese",
                "duration": 3.14,
                "text": "Transcrição detalhada verbose_json",
                "segments": [{"id": 0, "text": "Transcrição detalhada verbose_json"}]
            }).encode("utf-8")
            self.send_response(200)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        # -------------------------------------------------------------
        # Scenarios for Timeouts and Delays
        # -------------------------------------------------------------
        if sc == "slow_read":
            time.sleep(self.delay_s)
            body = b'{"text": "delayed response completed"}'
            try:
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            except (BrokenPipeError, ConnectionResetError):
                pass
            return

        # -------------------------------------------------------------
        # Scenarios for HTTP 429 Rate Limiting
        # -------------------------------------------------------------
        if sc == "429_int_0":
            if cnt == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"Rate limit exceeded"}}')
            else:
                body = b'{"text": "succeeded after retry 0s"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return

        if sc == "429_float_short":
            if cnt == 1:
                self.send_response(429)
                self.send_header("Retry-After", "0.05")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"Rate limit: 0.05s backoff"}}')
            else:
                body = b'{"text": "succeeded after retry 0.05s"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return

        if sc == "429_float_boundary_1_0":
            if cnt == 1:
                self.send_response(429)
                self.send_header("Retry-After", "1.0")
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":{"message":"Rate limit: 1.0s backoff"}}')
            else:
                body = b'{"text": "succeeded after boundary retry 1.0s"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return

        if sc == "429_float_long":
            self.send_response(429)
            self.send_header("Retry-After", "1.5")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Rate limit: 1.5s backoff"}}')
            return

        if sc == "429_int_long":
            self.send_response(429)
            self.send_header("Retry-After", "4")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Rate limit: 4s backoff"}}')
            return

        if sc == "429_missing_header":
            self.send_response(429)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Quota exceeded entirely"}}')
            return

        if sc == "429_malformed_string":
            self.send_response(429)
            self.send_header("Retry-After", "invalid_delta_seconds")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Invalid retry-after"}}')
            return

        if sc == "429_http_date":
            self.send_response(429)
            self.send_header("Retry-After", "Wed, 21 Oct 2026 07:28:00 GMT")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"HTTP date format"}}')
            return

        if sc == "429_persistent":
            self.send_response(429)
            self.send_header("Retry-After", "0.02")
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Persistent 429"}}')
            return

        # -------------------------------------------------------------
        # Scenarios for HTTP 500 / 503 Server Errors
        # -------------------------------------------------------------
        if sc == "500_transient":
            if cnt == 1:
                self.send_response(500)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"Internal Groq Server Error"}')
            else:
                body = b'{"text": "recovered after transient 500"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return

        if sc == "500_persistent":
            self.send_response(500)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"Persistent 500 Internal Server Error"}')
            return

        if sc == "503_transient":
            if cnt == 1:
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.end_headers()
                self.wfile.write(b'{"error":"Service Unavailable"}')
            else:
                body = b'{"text": "recovered after transient 503"}'
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
            return

        if sc == "503_persistent":
            self.send_response(503)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":"Persistent 503 Service Unavailable"}')
            return

        if sc == "502_bad_gateway_html":
            html = (
                b"<!DOCTYPE html><html><head><title>502 Bad Gateway</title></head>"
                b"<body><center><h1>502 Bad Gateway</h1></center><hr><center>cloudflare</center></body></html>"
            )
            self.send_response(502)
            self.send_header("Content-Type", "text/html; charset=UTF-8")
            self.send_header("Content-Length", str(len(html)))
            self.end_headers()
            self.wfile.write(html)
            return

        if sc == "401_auth_error":
            self.send_response(401)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Invalid API Key","type":"invalid_request_error"}}')
            return

        if sc == "400_bad_request":
            self.send_response(400)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"error":{"message":"Audio file too large or corrupted"}}')
            return

        # -------------------------------------------------------------
        # Scenarios for UTF-8 Safety and Empty/Corrupted Bodies
        # -------------------------------------------------------------
        if sc == "utf8_corrupted_json":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"text": "Corrompido \xff\xfe texto"}')
            return

        if sc == "utf8_corrupted_error_400":
            self.send_response(400)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"Erro grave \xff\xfe no processamento")
            return

        if sc == "empty_200_body":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", "0")
            self.end_headers()
            return

        if sc == "whitespace_200_body":
            self.send_response(200)
            self.send_header("Content-Type", "text/plain")
            self.end_headers()
            self.wfile.write(b"   \r\n\t  ")
            return

        if sc == "malformed_json_200":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b"{invalid: json, not: closed")
            return

        if sc == "json_null_text":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.end_headers()
            self.wfile.write(b'{"text": null}')
            return

        # Default fallback
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.end_headers()
        self.wfile.write(b'{"text": "default response"}')

    def log_message(self, format, *args):
        # Silence HTTP server logs during unit test execution
        pass


class QuietHTTPServer(http.server.HTTPServer):
    def handle_error(self, request, client_address):
        # Ignore broken pipes from intentional client-side disconnects / timeouts
        import sys
        exctype = sys.exc_info()[0]
        if exctype in (BrokenPipeError, ConnectionResetError):
            return
        super().handle_error(request, client_address)


class TestChallengerSTTR2Stress(unittest.TestCase):
    """
    Exhaustive empirical test suite stress-testing GroqSTTClient against production code.
    NO internal mocking or patching of client._opener is performed.
    """

    @classmethod
    def setUpClass(cls):
        # 1. Start HTTP Mock Server
        cls.http_server = QuietHTTPServer(("127.0.0.1", 0), EmpiricalMockGroqServer)
        cls.http_port = cls.http_server.server_address[1]
        cls.http_url = f"http://127.0.0.1:{cls.http_port}/openai/v1/audio/transcriptions"
        cls.http_thread = threading.Thread(target=cls.http_server.serve_forever, daemon=True)
        cls.http_thread.start()

        # 2. Start HTTPS Mock Server with temporary self-signed certificate
        cls.tmpdir = tempfile.mkdtemp()
        cls.certfile = os.path.join(cls.tmpdir, "cert.pem")
        subprocess.run(
            [
                "openssl", "req", "-x509", "-newkey", "rsa:2048",
                "-keyout", cls.certfile, "-out", cls.certfile,
                "-days", "1", "-nodes", "-subj", "/CN=localhost"
            ],
            check=True,
            capture_output=True,
        )

        cls.https_server = QuietHTTPServer(("localhost", 0), EmpiricalMockGroqServer)
        server_ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
        server_ctx.load_cert_chain(cls.certfile)
        cls.https_server.socket = server_ctx.wrap_socket(cls.https_server.socket, server_side=True)
        cls.https_port = cls.https_server.server_address[1]
        cls.https_url = f"https://localhost:{cls.https_port}/openai/v1/audio/transcriptions"
        cls.https_thread = threading.Thread(target=cls.https_server.serve_forever, daemon=True)
        cls.https_thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.http_server.shutdown()
        cls.http_server.server_close()
        cls.https_server.shutdown()
        cls.https_server.server_close()
        shutil.rmtree(cls.tmpdir, ignore_errors=True)

    def setUp(self):
        EmpiricalMockGroqServer.scenario = "success_text"
        EmpiricalMockGroqServer.request_count = 0
        EmpiricalMockGroqServer.received_headers = {}
        EmpiricalMockGroqServer.received_body = b""
        EmpiricalMockGroqServer.delay_s = 0.0

    # =========================================================================
    # 1. REAL EXECUTION & ZERO TYPEERROR ON CONNECTION CREATION
    # =========================================================================

    def test_http_connection_creation_zero_typeerror(self):
        """
        EMPIRICAL PROOF: Verifies that GroqSTTClient executing over HTTP connects
        without raising TypeError: got multiple values for keyword argument 'timeout'.
        """
        cfg = GroqConfig(api_url=self.http_url, connect_timeout_seconds=2.0, read_timeout_seconds=5.0)
        client = GroqSTTClient(cfg, api_key="gsk_live_test_key")

        # Execute transcription directly with UNPATCHED client._opener
        res = client.transcribe(b"dummy_opus_data")
        self.assertEqual(res, "Transcrição em português de teste")
        self.assertEqual(EmpiricalMockGroqServer.request_count, 1)

        # Verify Authorization header and Multipart headers
        self.assertEqual(EmpiricalMockGroqServer.received_headers.get("Authorization"), "Bearer gsk_live_test_key")
        self.assertIn("multipart/form-data; boundary=", EmpiricalMockGroqServer.received_headers.get("Content-Type", ""))
        self.assertIn(b'name="model"\r\n\r\nwhisper-large-v3-turbo', EmpiricalMockGroqServer.received_body)
        self.assertIn(b'filename="audio.ogg"', EmpiricalMockGroqServer.received_body)

    def test_https_connection_creation_zero_typeerror(self):
        """
        EMPIRICAL PROOF: Verifies that GroqSTTClient executing over HTTPS connects
        without raising TypeError: got multiple values for keyword argument 'timeout'
        and preserves TLS context.
        """
        cfg = GroqConfig(api_url=self.https_url, connect_timeout_seconds=2.0, read_timeout_seconds=5.0)
        client = GroqSTTClient(cfg, api_key="gsk_live_ssl_key")

        # Trust the temporary test certificate in the client's HTTPSHandler context
        for handler in client._opener.handlers:
            if hasattr(handler, "_context") and handler._context is not None:
                handler._context.load_verify_locations(cafile=self.certfile)

        res = client.transcribe(b"dummy_ssl_audio")
        self.assertEqual(res, "Transcrição em português de teste")
        self.assertEqual(EmpiricalMockGroqServer.request_count, 1)

    def test_direct_connection_classes_instantiation(self):
        """Directly instantiates TimeoutHTTPConnection and TimeoutHTTPSConnection with timeout kwargs."""
        http_conn = TimeoutHTTPConnection("127.0.0.1", timeout=2.5, read_timeout=8.5)
        self.assertEqual(http_conn.timeout, 2.5)
        self.assertEqual(http_conn.read_timeout, 8.5)

        https_conn = TimeoutHTTPSConnection("127.0.0.1", timeout=3.0, read_timeout=9.0)
        self.assertEqual(https_conn.timeout, 3.0)
        self.assertEqual(https_conn.read_timeout, 9.0)

    # =========================================================================
    # 2. RESPONSE FORMATS & PARSING
    # =========================================================================

    def test_response_format_json(self):
        """Verify parsing of application/json format."""
        EmpiricalMockGroqServer.scenario = "success_json"
        cfg = GroqConfig(api_url=self.http_url, response_format="json")
        client = GroqSTTClient(cfg, api_key="gsk_key")

        res = client.transcribe(b"audio")
        self.assertEqual(res, "Transcrição via JSON")

    def test_response_format_verbose_json(self):
        """Verify parsing of verbose_json format extracting 'text' field."""
        EmpiricalMockGroqServer.scenario = "success_verbose_json"
        cfg = GroqConfig(api_url=self.http_url, response_format="verbose_json")
        client = GroqSTTClient(cfg, api_key="gsk_key")

        res = client.transcribe(b"audio")
        self.assertEqual(res, "Transcrição detalhada verbose_json")

    # =========================================================================
    # 3. DUAL TIMEOUTS (CONNECT VS READ)
    # =========================================================================

    def test_connect_timeout_empirically_measured(self):
        """
        Verify connect timeout against non-routable IP terminates at connect_timeout threshold,
        NOT read_timeout threshold.
        """
        # Non-routable TEST-NET-2 IP
        cfg = GroqConfig(
            api_url="http://198.51.100.1:81/v1/stt",
            connect_timeout_seconds=0.25,
            read_timeout_seconds=5.0,
        )
        client = GroqSTTClient(cfg, api_key="gsk_key")

        t0 = time.time()
        with self.assertRaises(GroqTimeoutError) as ctx:
            client.transcribe(b"audio")
        elapsed = time.time() - t0

        self.assertIn("connection timed out after 0.25s", str(ctx.exception))
        # Elapsed time must be close to connect_timeout (0.25s), well under read_timeout (5.0s)
        self.assertLess(elapsed, 1.0, f"Connect timeout took {elapsed:.2f}s; expected ~0.25s")

    def test_read_timeout_empirically_measured(self):
        """
        Verify read timeout against slow-responding server terminates at read_timeout threshold,
        NOT connect_timeout threshold.
        """
        EmpiricalMockGroqServer.scenario = "slow_read"
        EmpiricalMockGroqServer.delay_s = 1.0

        cfg = GroqConfig(
            api_url=self.http_url,
            connect_timeout_seconds=5.0,
            read_timeout_seconds=0.2,
        )
        client = GroqSTTClient(cfg, api_key="gsk_key")

        t0 = time.time()
        with self.assertRaises(GroqTimeoutError) as ctx:
            client.transcribe(b"audio")
        elapsed = time.time() - t0

        self.assertIn("request timed out after 0.2s", str(ctx.exception))
        # Elapsed time must be close to read_timeout (0.2s), well under connect_timeout (5.0s)
        self.assertLess(elapsed, 0.8, f"Read timeout took {elapsed:.2f}s; expected ~0.2s")
        self.assertGreaterEqual(elapsed, 0.15)

    # =========================================================================
    # 4. HTTP 429 RETRY-AFTER BACKOFF (ALL VARIANTS)
    # =========================================================================

    def test_http_429_integer_0_retries_and_succeeds(self):
        """Verify HTTP 429 with 'Retry-After: 0' retries immediately and succeeds."""
        EmpiricalMockGroqServer.scenario = "429_int_0"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        res = client.transcribe(b"audio")
        self.assertEqual(res, "succeeded after retry 0s")
        self.assertEqual(EmpiricalMockGroqServer.request_count, 2)

    def test_http_429_float_short_retries_and_succeeds(self):
        """Verify HTTP 429 with 'Retry-After: 0.05' sleeps 50ms and succeeds on retry."""
        EmpiricalMockGroqServer.scenario = "429_float_short"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        t0 = time.time()
        res = client.transcribe(b"audio")
        elapsed = time.time() - t0

        self.assertEqual(res, "succeeded after retry 0.05s")
        self.assertEqual(EmpiricalMockGroqServer.request_count, 2)
        self.assertGreaterEqual(elapsed, 0.04)

    def test_http_429_boundary_1_0_retries_and_succeeds(self):
        """Verify HTTP 429 with exact boundary 'Retry-After: 1.0' retries and succeeds."""
        EmpiricalMockGroqServer.scenario = "429_float_boundary_1_0"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        t0 = time.time()
        res = client.transcribe(b"audio")
        elapsed = time.time() - t0

        self.assertEqual(res, "succeeded after boundary retry 1.0s")
        self.assertEqual(EmpiricalMockGroqServer.request_count, 2)
        self.assertGreaterEqual(elapsed, 0.95)

    def test_http_429_float_long_fails_immediately_without_sleeping(self):
        """Verify HTTP 429 with 'Retry-After: 1.5' (> 1.0s) raises GroqRateLimitError immediately."""
        EmpiricalMockGroqServer.scenario = "429_float_long"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        t0 = time.time()
        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"audio")
        elapsed = time.time() - t0

        self.assertEqual(ctx.exception.retry_after, 1.5)
        self.assertIn("please retry in 1.5 seconds", str(ctx.exception))
        self.assertEqual(EmpiricalMockGroqServer.request_count, 1)
        self.assertLess(elapsed, 0.2, "Must not sleep when retry_after > 1.0s")

    def test_http_429_integer_long_fails_immediately_without_sleeping(self):
        """Verify HTTP 429 with 'Retry-After: 4' raises GroqRateLimitError immediately."""
        EmpiricalMockGroqServer.scenario = "429_int_long"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        t0 = time.time()
        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"audio")
        elapsed = time.time() - t0

        self.assertEqual(ctx.exception.retry_after, 4.0)
        self.assertIn("please retry in 4 seconds", str(ctx.exception))
        self.assertEqual(EmpiricalMockGroqServer.request_count, 1)
        self.assertLess(elapsed, 0.2)

    def test_http_429_missing_header_raises_rate_limit_error(self):
        """Verify HTTP 429 with missing Retry-After header sets retry_after=None."""
        EmpiricalMockGroqServer.scenario = "429_missing_header"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"audio")

        self.assertIsNone(ctx.exception.retry_after)
        self.assertIn("quota exceeded, please retry later", str(ctx.exception))
        self.assertEqual(EmpiricalMockGroqServer.request_count, 1)

    def test_http_429_malformed_string_does_not_crash_with_value_error(self):
        """Verify HTTP 429 with non-numeric Retry-After string handles ValueError cleanly."""
        EmpiricalMockGroqServer.scenario = "429_malformed_string"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"audio")

        self.assertIsNone(ctx.exception.retry_after)
        self.assertEqual(EmpiricalMockGroqServer.request_count, 1)

    def test_http_429_http_date_fallback(self):
        """Verify HTTP 429 with HTTP-date Retry-After format degrades to retry_after=None."""
        EmpiricalMockGroqServer.scenario = "429_http_date"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"audio")

        self.assertIsNone(ctx.exception.retry_after)
        self.assertEqual(EmpiricalMockGroqServer.request_count, 1)

    def test_http_429_persistent_avoids_infinite_loop(self):
        """Verify consecutive 429 responses terminate on 2nd attempt without infinite recursion."""
        EmpiricalMockGroqServer.scenario = "429_persistent"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        with self.assertRaises(GroqRateLimitError) as ctx:
            client.transcribe(b"audio")

        self.assertEqual(ctx.exception.retry_after, 0.02)
        # Attempt 1 -> 429 (sleep 0.02s) -> Attempt 2 -> 429 (raises GroqRateLimitError)
        self.assertEqual(EmpiricalMockGroqServer.request_count, 2)

    # =========================================================================
    # 5. HTTP 500 / 503 RETRY & HTTP 502 HTML SAFETY
    # =========================================================================

    def test_http_500_transient_retries_after_500ms(self):
        """Verify transient HTTP 500 retries once after 500ms and succeeds."""
        EmpiricalMockGroqServer.scenario = "500_transient"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        t0 = time.time()
        res = client.transcribe(b"audio")
        elapsed = time.time() - t0

        self.assertEqual(res, "recovered after transient 500")
        self.assertEqual(EmpiricalMockGroqServer.request_count, 2)
        self.assertGreaterEqual(elapsed, 0.45, "Must sleep 500ms on 500 retry")

    def test_http_500_persistent_fails_after_retry(self):
        """Verify persistent HTTP 500 retries once, then raises GroqServerError."""
        EmpiricalMockGroqServer.scenario = "500_persistent"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        t0 = time.time()
        with self.assertRaises(GroqServerError) as ctx:
            client.transcribe(b"audio")
        elapsed = time.time() - t0

        self.assertIn("Groq server error (HTTP 500)", str(ctx.exception))
        self.assertEqual(EmpiricalMockGroqServer.request_count, 2)
        self.assertGreaterEqual(elapsed, 0.45)

    def test_http_503_transient_retries_and_succeeds(self):
        """Verify transient HTTP 503 retries once after 500ms and succeeds."""
        EmpiricalMockGroqServer.scenario = "503_transient"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        t0 = time.time()
        res = client.transcribe(b"audio")
        elapsed = time.time() - t0

        self.assertEqual(res, "recovered after transient 503")
        self.assertEqual(EmpiricalMockGroqServer.request_count, 2)
        self.assertGreaterEqual(elapsed, 0.45)

    def test_http_503_persistent_fails_after_retry(self):
        """Verify persistent HTTP 503 retries once, then raises GroqServerError."""
        EmpiricalMockGroqServer.scenario = "503_persistent"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        t0 = time.time()
        with self.assertRaises(GroqServerError) as ctx:
            client.transcribe(b"audio")
        elapsed = time.time() - t0

        self.assertIn("Groq server error (HTTP 503)", str(ctx.exception))
        self.assertEqual(EmpiricalMockGroqServer.request_count, 2)
        self.assertGreaterEqual(elapsed, 0.45)

    def test_http_502_bad_gateway_html_fails_fast_without_retry(self):
        """
        Verify HTTP 502 Bad Gateway with raw Cloudflare HTML body raises GroqServerError
        immediately without wasting retry cycles or dumping unreadable HTML.
        """
        EmpiricalMockGroqServer.scenario = "502_bad_gateway_html"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        t0 = time.time()
        with self.assertRaises(GroqServerError) as ctx:
            client.transcribe(b"audio")
        elapsed = time.time() - t0

        self.assertEqual(str(ctx.exception), "Groq gateway error: 502 Bad Gateway")
        self.assertEqual(EmpiricalMockGroqServer.request_count, 1, "502 must not trigger 500ms retry loop")
        self.assertLess(elapsed, 0.2)

    def test_http_401_unauthorized_fails_immediately(self):
        """Verify HTTP 401 raises GroqAuthError without retrying."""
        EmpiricalMockGroqServer.scenario = "401_auth_error"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_invalid_key")

        with self.assertRaises(GroqAuthError):
            client.transcribe(b"audio")
        self.assertEqual(EmpiricalMockGroqServer.request_count, 1)

    def test_http_400_bad_request_fails_immediately(self):
        """Verify HTTP 400 raises GroqTranscriptionError without retrying."""
        EmpiricalMockGroqServer.scenario = "400_bad_request"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        with self.assertRaises(GroqTranscriptionError) as ctx:
            client.transcribe(b"audio")
        self.assertIn("HTTP 400", str(ctx.exception))
        self.assertEqual(EmpiricalMockGroqServer.request_count, 1)

    # =========================================================================
    # 6. UTF-8 SAFETY & EMPTY / CORRUPTED BODY HANDLING
    # =========================================================================

    def test_utf8_corrupted_json_response_decoded_safely(self):
        """Verify invalid UTF-8 byte sequences are safely replaced via errors='replace'."""
        EmpiricalMockGroqServer.scenario = "utf8_corrupted_json"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        res = client.transcribe(b"audio")
        self.assertIn("Corrompido \ufffd\ufffd texto", res)

    def test_utf8_corrupted_error_response_decoded_safely(self):
        """Verify invalid UTF-8 byte sequences in HTTP error responses do not crash."""
        EmpiricalMockGroqServer.scenario = "utf8_corrupted_error_400"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        with self.assertRaises(GroqTranscriptionError) as ctx:
            client.transcribe(b"audio")
        self.assertIn("\ufffd\ufffd", str(ctx.exception))

    def test_empty_200_body_returns_empty_string(self):
        """Verify 0-byte 200 OK response returns empty string without JSONDecodeError."""
        EmpiricalMockGroqServer.scenario = "empty_200_body"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        res = client.transcribe(b"audio")
        self.assertEqual(res, "")

    def test_whitespace_200_body_returns_empty_string(self):
        """Verify whitespace-only 200 OK response returns empty string."""
        EmpiricalMockGroqServer.scenario = "whitespace_200_body"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        res = client.transcribe(b"audio")
        self.assertEqual(res, "")

    def test_malformed_json_200_falls_back_to_raw_text(self):
        """Verify unparseable JSON in 200 OK falls back to raw string."""
        EmpiricalMockGroqServer.scenario = "malformed_json_200"
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        res = client.transcribe(b"audio")
        self.assertEqual(res, "{invalid: json, not: closed")

    def test_empty_audio_bytes_short_circuits_without_network(self):
        """Verify calling transcribe(b'') returns empty string instantly without HTTP request."""
        cfg = GroqConfig(api_url=self.http_url)
        client = GroqSTTClient(cfg, api_key="gsk_key")

        res = client.transcribe(b"")
        self.assertEqual(res, "")
        self.assertEqual(EmpiricalMockGroqServer.request_count, 0)

    # =========================================================================
    # 7. CLIENT INITIALIZATION & CREDENTIAL RESOLUTION
    # =========================================================================

    def test_missing_api_key_raises_groq_auth_error(self):
        """Verify missing, empty, or whitespace API key raises GroqAuthError."""
        cfg = GroqConfig(api_key="")
        with self.assertRaises(GroqAuthError):
            GroqSTTClient(cfg, api_key=None)

        with self.assertRaises(GroqAuthError):
            GroqSTTClient(cfg, api_key="   ")

    def test_dns_resolution_failure_raises_groq_network_error(self):
        """Verify DNS lookup failure raises GroqNetworkError."""
        cfg = GroqConfig(api_url="http://nonexistent.groq.cloud.invalid.example:80/v1/stt")
        client = GroqSTTClient(cfg, api_key="gsk_key")

        with self.assertRaises(GroqNetworkError) as ctx:
            client.transcribe(b"audio")
        self.assertIn("Network error connecting to Groq STT", str(ctx.exception))


if __name__ == "__main__":
    unittest.main()
