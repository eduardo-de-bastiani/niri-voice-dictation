"""
Groq Cloud STT Client for Niri Voice Dictation.
Implements HTTPS multipart form-data upload to whisper-large-v3-turbo,
fail-fast network timeouts, HTTP 429 Retry-After backoff, and robust error handling.
"""

from __future__ import annotations

import http.client
import json
import logging
import socket
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import GroqConfig

logger = logging.getLogger("niri-dictate.stt")


class GroqError(Exception):
    """Base exception for all Groq STT client failures."""
    pass


class GroqAuthError(GroqError):
    """Raised when API credentials are missing, empty, or rejected (HTTP 401)."""
    pass


class GroqRateLimitError(GroqError):
    """Raised when HTTP 429 rate limit is encountered and cannot be immediately retried."""

    def __init__(self, message: str, retry_after: float | None = None):
        super().__init__(message)
        self.retry_after = retry_after


class GroqServerError(GroqError):
    """Raised on persistent HTTP 500, 502, or 503 server/gateway errors."""
    pass


class GroqNetworkError(GroqError):
    """Raised on DNS failure, connection refusal, or network unreachable."""
    pass


class GroqTimeoutError(GroqNetworkError):
    """Raised when connection or read timeout thresholds are exceeded."""
    pass


class GroqTranscriptionError(GroqError):
    """Raised when the transcription payload is malformed or rejected."""
    pass


class TimeoutHTTPConnection(http.client.HTTPConnection):
    """HTTPConnection enforcing separate connect and read timeouts."""

    def __init__(self, *args: Any, read_timeout: float = 8.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read_timeout = read_timeout

    def connect(self) -> None:
        super().connect()
        if self.sock:
            self.sock.settimeout(self.read_timeout)


class TimeoutHTTPSConnection(http.client.HTTPSConnection):
    """HTTPSConnection enforcing separate connect and read timeouts."""

    def __init__(self, *args: Any, read_timeout: float = 8.0, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.read_timeout = read_timeout

    def connect(self) -> None:
        super().connect()
        if self.sock:
            self.sock.settimeout(self.read_timeout)


class GroqSTTClient:
    """Client for Groq's whisper-large-v3-turbo STT service using Python standard library."""

    def __init__(self, config: GroqConfig, api_key: str | None = None) -> None:
        self.config = config
        self.api_key = (api_key or config.api_key or "").strip()
        if not self.api_key:
            raise GroqAuthError("Missing Groq API Key ($GROQ_API_KEY or config.toml)")

        self.connect_timeout = float(config.connect_timeout_seconds)
        self.read_timeout = float(config.read_timeout_seconds)

        # Build custom opener with dual-timeout connections
        class CustomHTTPHandler(urllib.request.HTTPHandler):
            def http_open(_self, req: urllib.request.Request) -> http.client.HTTPResponse:
                def _factory(host: str, timeout: Any = None, **kw: Any) -> TimeoutHTTPConnection:
                    return TimeoutHTTPConnection(
                        host, timeout=self.connect_timeout, read_timeout=self.read_timeout, **kw
                    )

                return _self.do_open(_factory, req)

        class CustomHTTPSHandler(urllib.request.HTTPSHandler):
            def https_open(_self, req: urllib.request.Request) -> http.client.HTTPResponse:
                def _factory(host: str, timeout: Any = None, **kw: Any) -> TimeoutHTTPSConnection:
                    return TimeoutHTTPSConnection(
                        host, timeout=self.connect_timeout, read_timeout=self.read_timeout, **kw
                    )

                return _self.do_open(_factory, req, context=_self._context)

        self._opener = urllib.request.build_opener(CustomHTTPHandler, CustomHTTPSHandler)

    def _build_multipart_request(
        self, audio_bytes: bytes, prompt: str
    ) -> tuple[urllib.request.Request, str]:
        """Constructs an RFC-compliant multipart/form-data request."""
        boundary = f"----WebKitFormBoundary{uuid.uuid4().hex}"
        body = bytearray()

        fields: dict[str, str | None] = {
            "model": self.config.model,
            "language": self.config.language,
            "temperature": str(self.config.temperature),
            "response_format": self.config.response_format,
            "prompt": prompt if prompt else None,
        }

        for name, value in fields.items():
            if value is not None:
                body.extend(f"--{boundary}\r\n".encode("utf-8"))
                body.extend(f'Content-Disposition: form-data; name="{name}"\r\n\r\n'.encode("utf-8"))
                body.extend(f"{value}\r\n".encode("utf-8"))

        # Audio file attachment (Groq requires .ogg/.opus/.wav extension in multipart filename)
        body.extend(f"--{boundary}\r\n".encode("utf-8"))
        body.extend(b'Content-Disposition: form-data; name="file"; filename="audio.ogg"\r\n')
        body.extend(b"Content-Type: audio/ogg\r\n\r\n")
        body.extend(audio_bytes)
        body.extend(f"\r\n--{boundary}--\r\n".encode("utf-8"))

        req = urllib.request.Request(self.config.api_url, data=bytes(body), method="POST")
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", f"multipart/form-data; boundary={boundary}")
        req.add_header("Accept", "text/plain, application/json")
        req.add_header("User-Agent", "niri-dictate/1.0 (Linux; Wayland)")
        return req, boundary

    def transcribe(self, audio_bytes: bytes, prompt: str = "") -> str:
        """
        Transcribes audio bytes via Groq Cloud STT API.
        """
        if not audio_bytes:
            logger.debug("Transcribe called with empty audio bytes; returning empty string.")
            return ""

        req, _ = self._build_multipart_request(audio_bytes, prompt)

        def _execute(allow_retry: bool = True) -> str:
            try:
                with self._opener.open(req) as resp:
                    raw_bytes: bytes = resp.read()
                    # Resilient decoding: replacement characters prevent UnicodeDecodeError crashes
                    decoded_text = raw_bytes.decode("utf-8", errors="replace").strip()
                    if not decoded_text:
                        return ""

                    # JSON response handling
                    if (
                        self.config.response_format in ("json", "verbose_json")
                        or resp.headers.get_content_type() == "application/json"
                    ):
                        try:
                            data = json.loads(decoded_text)
                            if isinstance(data, dict):
                                return str(data.get("text", "")).strip()
                        except json.JSONDecodeError:
                            logger.warning(
                                "Failed to parse JSON response from Groq; returning raw text."
                            )
                            return decoded_text

                    return decoded_text

            except urllib.error.HTTPError as err:
                status = err.code
                err_body = err.read().decode("utf-8", errors="replace")

                # HTTP 401: Unauthorized
                if status == 401:
                    logger.error("Groq API authentication failed (HTTP 401): %s", err_body)
                    raise GroqAuthError("Invalid or missing Groq API Key ($GROQ_API_KEY)") from err

                # HTTP 429: Rate Limit
                if status == 429:
                    retry_header = err.headers.get("Retry-After")
                    retry_after: float | None = None
                    if retry_header:
                        try:
                            retry_after = float(retry_header)
                        except ValueError:
                            pass

                    logger.warning(
                        "Groq HTTP 429 rate limit encountered. Retry-After=%s", retry_after
                    )
                    if allow_retry and retry_after is not None and retry_after <= 1.0:
                        logger.info("Sleeping %ss for short rate-limit backoff...", retry_after)
                        time.sleep(retry_after)
                        return _execute(allow_retry=False)

                    msg = (
                        f"Groq Rate Limit: please retry in {retry_after:g} seconds"
                        if retry_after is not None
                        else "Groq Rate Limit: quota exceeded, please retry later"
                    )
                    raise GroqRateLimitError(msg, retry_after=retry_after) from err

                # HTTP 500 / 503: Transient Server Errors
                if status in (500, 503):
                    logger.warning(
                        "Groq server error (HTTP %d). Retrying once after 500ms...", status
                    )
                    if allow_retry:
                        time.sleep(0.5)
                        return _execute(allow_retry=False)
                    raise GroqServerError(f"Groq server error (HTTP {status}): {err_body}") from err

                # HTTP 502: Bad Gateway
                if status == 502:
                    logger.error("Groq gateway error (HTTP 502 Bad Gateway).")
                    raise GroqServerError("Groq gateway error: 502 Bad Gateway") from err

                # Other HTTP client/server errors
                logger.error("Groq API returned HTTP %d: %s", status, err_body)
                raise GroqTranscriptionError(
                    f"Groq API error (HTTP {status}): {err_body}"
                ) from err

            except (socket.timeout, TimeoutError) as err:
                logger.error(
                    "Groq STT request timed out (read timeout %ss): %s", self.read_timeout, err
                )
                raise GroqTimeoutError(
                    f"Groq STT request timed out after {self.read_timeout}s"
                ) from err

            except urllib.error.URLError as err:
                if isinstance(err.reason, (socket.timeout, TimeoutError)):
                    logger.error(
                        "Groq STT connection timed out (%ss): %s", self.connect_timeout, err
                    )
                    raise GroqTimeoutError(
                        f"Groq STT connection timed out after {self.connect_timeout}s"
                    ) from err
                logger.error("Failed to connect to Groq STT endpoint: %s", err)
                raise GroqNetworkError(f"Network error connecting to Groq STT: {err.reason}") from err

        return _execute()
