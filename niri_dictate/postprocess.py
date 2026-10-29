"""
LLM Transcript Post-Processing Module for Niri Voice Dictation.
Transforms raw spontaneous Whisper speech-to-text output into clean, natural text
using Groq's ultra-fast LLM inference (e.g., openai/gpt-oss-20b) while strictly
preserving 100% of spoken information without summarizing or answering dictated prompts.
"""

from __future__ import annotations

import http.client
import json
import logging
import re
import socket
import time
import urllib.error
import urllib.request
from typing import TYPE_CHECKING, Any

from .stt import TimeoutHTTPConnection, TimeoutHTTPSConnection

if TYPE_CHECKING:
    from .config import PostProcessConfig

logger = logging.getLogger("niri-dictate.postprocess")

_TAG_WRAPPER_RE = re.compile(
    r"^\s*<transcricao_bruta>\s*(.*?)\s*</transcricao_bruta>\s*$",
    re.DOTALL | re.IGNORECASE,
)


class PostProcessError(Exception):
    """Raised when LLM transcript post-processing fails."""


class GroqPostProcessor:
    """Refines raw speech transcriptions via Groq Chat Completions API."""

    def __init__(self, config: PostProcessConfig, api_key: str) -> None:
        self.config = config
        self.api_key = (api_key or "").strip()
        if not self.api_key:
            raise PostProcessError("Missing Groq API Key for post-processing.")

        self.connect_timeout = float(config.connect_timeout_seconds)
        self.read_timeout = float(config.read_timeout_seconds)

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

    def _build_payload(self, raw_text: str) -> dict[str, Any]:
        """Constructs the JSON request payload configured for Groq chat completions."""
        wrapped_input = f"<transcricao_bruta>\n{raw_text.strip()}\n</transcricao_bruta>"
        payload: dict[str, Any] = {
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": self.config.system_prompt},
                {"role": "user", "content": wrapped_input},
            ],
            "temperature": self.config.temperature,
            "stream": False,
        }
        if self.config.reasoning_effort:
            payload["reasoning_effort"] = self.config.reasoning_effort
        if "gpt-oss" in self.config.model:
            payload["include_reasoning"] = bool(self.config.include_reasoning)
        return payload

    @staticmethod
    def _clean_output(content: str) -> str:
        """Strips accidental delimiter echoes while keeping exact refined text."""
        cleaned = content.strip()
        match = _TAG_WRAPPER_RE.match(cleaned)
        if match:
            cleaned = match.group(1).strip()
        return cleaned

    def refine(self, raw_text: str) -> str:
        """
        Sends raw transcription to Groq LLM for disfluency/self-correction cleanup.
        Returns cleaned text, or raises PostProcessError on API/network failure.
        """
        stripped = raw_text.strip()
        if not stripped:
            return ""

        payload_bytes = json.dumps(self._build_payload(stripped)).encode("utf-8")
        req = urllib.request.Request(
            self.config.api_url,
            data=payload_bytes,
            method="POST",
        )
        req.add_header("Authorization", f"Bearer {self.api_key}")
        req.add_header("Content-Type", "application/json")
        req.add_header("Accept", "application/json")
        req.add_header("User-Agent", "niri-dictate/1.0 (Linux; Wayland)")

        def _execute(allow_retry: bool = True) -> str:
            try:
                with self._opener.open(req) as resp:
                    body_str = resp.read().decode("utf-8", errors="replace")
                    data = json.loads(body_str)
                    choices = data.get("choices") or []
                    if not choices:
                        raise PostProcessError("Groq chat completion returned empty choices.")
                    message = choices[0].get("message") or {}
                    content = message.get("content")
                    if not isinstance(content, str) or not content.strip():
                        raise PostProcessError("Groq chat completion returned empty content.")
                    return self._clean_output(content)

            except urllib.error.HTTPError as err:
                status = err.code
                err_body = err.read().decode("utf-8", errors="replace")
                if status in (429, 500, 502, 503) and allow_retry:
                    retry_wait = 0.5
                    retry_hdr = err.headers.get("Retry-After")
                    if retry_hdr:
                        try:
                            parsed = float(retry_hdr)
                            if parsed <= 1.5:
                                retry_wait = parsed
                        except ValueError:
                            pass
                    logger.warning("Post-process HTTP %d; retrying once in %ss...", status, retry_wait)
                    time.sleep(retry_wait)
                    return _execute(allow_retry=False)
                raise PostProcessError(f"Groq LLM HTTP {status}: {err_body}") from err

            except (socket.timeout, TimeoutError, urllib.error.URLError, json.JSONDecodeError) as err:
                raise PostProcessError(f"Groq LLM post-processing failed: {err}") from err

        return _execute()

