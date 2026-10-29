"""
Unit & Integration Tests for LLM Transcript Post-Processing (GroqPostProcessor)
and Quickshell Floating Dictation Pill (VisualFeedbackManager provider='pill').
"""

from __future__ import annotations

import argparse
import http.server
import json
import socketserver
import tempfile
import threading
import unittest
from pathlib import Path
from typing import Any
from unittest.mock import MagicMock, patch

from niri_dictate.cli import handle_abort, handle_toggle
from niri_dictate.config import (
    AppConfig,
    AudioConfig,
    CompositorConfig,
    FeedbackConfig,
    GeneralConfig,
    GroqConfig,
    InjectionConfig,
    PostProcessConfig,
)
from niri_dictate.feedback import VisualFeedbackManager
from niri_dictate.postprocess import GroqPostProcessor, PostProcessError
from niri_dictate.state import HardenedStateManager


class _MockLLMServer:
    """Ephemeral local HTTP server simulating Groq /openai/v1/chat/completions."""

    def __init__(self) -> None:
        self.requests: list[dict[str, Any]] = []
        self.responses: list[tuple[int, dict[str, Any] | str]] = []
        self._lock = threading.Lock()

        outer = self

        class Handler(http.server.BaseHTTPRequestHandler):
            def do_POST(self) -> None:  # noqa: N802
                length = int(self.headers.get("Content-Length", "0"))
                raw_body = self.rfile.read(length).decode("utf-8")
                parsed_body = json.loads(raw_body) if raw_body else {}

                with outer._lock:
                    outer.requests.append(
                        {
                            "path": self.path,
                            "headers": {k.lower(): v for k, v in self.headers.items()},
                            "body": parsed_body,
                        }
                    )
                    if outer.responses:
                        status, payload = outer.responses.pop(0)
                    else:
                        status, payload = (
                            200,
                            {
                                "choices": [
                                    {"message": {"role": "assistant", "content": "Texto limpo final."}}
                                ]
                            },
                        )

                self.send_response(status)
                self.send_header("Content-Type", "application/json")
                if status == 429:
                    self.send_header("Retry-After", "0.05")
                self.end_headers()
                body_bytes = (
                    json.dumps(payload).encode("utf-8")
                    if isinstance(payload, dict)
                    else payload.encode("utf-8")
                )
                self.wfile.write(body_bytes)

            def log_message(self, format: str, *args: object) -> None:  # noqa: A002
                return

        self._httpd = socketserver.TCPServer(("127.0.0.1", 0), Handler)
        self.port = self._httpd.server_address[1]
        self.url = f"http://127.0.0.1:{self.port}/openai/v1/chat/completions"
        self._thread = threading.Thread(target=self._httpd.serve_forever, daemon=True)
        self._thread.start()

    def close(self) -> None:
        self._httpd.shutdown()
        self._httpd.server_close()


class TestGroqPostProcessor(unittest.TestCase):
    def setUp(self) -> None:
        self.server = _MockLLMServer()

    def tearDown(self) -> None:
        self.server.close()

    def _make_config(self) -> PostProcessConfig:
        return PostProcessConfig(
            enabled=True,
            api_url=self.server.url,
            model="openai/gpt-oss-20b",
            temperature=0.0,
            reasoning_effort="low",
            include_reasoning=False,
            connect_timeout_seconds=2.0,
            read_timeout_seconds=4.0,
            system_prompt="System prompt de teste.",
        )

    def test_payload_headers_and_tag_wrapping(self) -> None:
        self.server.responses.append(
            (
                200,
                {
                    "choices": [
                        {
                            "message": {
                                "role": "assistant",
                                "content": "<transcricao_bruta>\nPrecisamos corrigir o endpoint da API.\n</transcricao_bruta>",
                            }
                        }
                    ]
                },
            )
        )
        proc = GroqPostProcessor(self._make_config(), api_key="gsk_test_dummy_key")
        out = proc.refine("É, tipo assim, precisamos corrigir o endpoint... quer dizer, o endpoint da API.")

        self.assertEqual(out, "Precisamos corrigir o endpoint da API.")
        self.assertEqual(len(self.server.requests), 1)

        req = self.server.requests[0]
        self.assertEqual(req["headers"].get("authorization"), "Bearer gsk_test_dummy_key")
        self.assertIn("niri-dictate/1.0", req["headers"].get("user-agent", ""))

        body = req["body"]
        self.assertEqual(body["model"], "openai/gpt-oss-20b")
        self.assertEqual(body["temperature"], 0.0)
        self.assertEqual(body["reasoning_effort"], "low")
        self.assertIs(body["include_reasoning"], False)
        self.assertIs(body["stream"], False)
        self.assertEqual(body["messages"][0]["role"], "system")
        self.assertIn("<transcricao_bruta>", body["messages"][1]["content"])

    def test_single_retry_on_transient_500_and_429(self) -> None:
        self.server.responses.append((429, {"error": "rate_limit"}))
        self.server.responses.append(
            (200, {"choices": [{"message": {"content": "Resultado após retry."}}]})
        )
        proc = GroqPostProcessor(self._make_config(), api_key="gsk_test_dummy_key")
        out = proc.refine("fala original")
        self.assertEqual(out, "Resultado após retry.")
        self.assertEqual(len(self.server.requests), 2)

    def test_persistent_error_raises_postprocess_error(self) -> None:
        self.server.responses.append((500, {"error": "server_down_1"}))
        self.server.responses.append((500, {"error": "server_down_2"}))
        proc = GroqPostProcessor(self._make_config(), api_key="gsk_test_dummy_key")
        with self.assertRaises(PostProcessError):
            proc.refine("fala original")


class TestPillFeedbackAndCLIFallback(unittest.TestCase):
    def setUp(self) -> None:
        self.tmp_dir = tempfile.TemporaryDirectory()
        self.run_dir = Path(self.tmp_dir.name)

    def tearDown(self) -> None:
        self.tmp_dir.cleanup()

    def _build_app_config(self, pp_enabled: bool = True, provider: str = "pill") -> AppConfig:
        return AppConfig(
            general=GeneralConfig(runtime_dir=self.run_dir),
            compositor=CompositorConfig(),
            audio=AudioConfig(source="@DEFAULT_AUDIO_SOURCE@"),
            groq=GroqConfig(api_key="gsk_test_key"),
            injection=InjectionConfig(),
            feedback=FeedbackConfig(provider=provider, quickshell_bin="/usr/bin/qs", pill_bottom_margin=38),
            postprocess=PostProcessConfig(enabled=pp_enabled),
        )

    def test_pill_ui_state_lifecycle(self) -> None:
        cfg = self._build_app_config(provider="pill")
        fm = VisualFeedbackManager(cfg.feedback, runtime_dir=self.run_dir, audio_source="@DEFAULT_AUDIO_SOURCE@")

        with (
            patch.object(VisualFeedbackManager, "_get_focused_output", return_value="eDP-1"),
            patch.object(fm, "_qs_bin", "/usr/bin/qs"),
            patch("niri_dictate.feedback.subprocess.Popen") as mock_popen,
        ):
            notif_id = fm.show_recording()
            self.assertEqual(notif_id, 0)
            mock_popen.assert_called_once()
            spawn_args, spawn_kwargs = mock_popen.call_args
            self.assertEqual(spawn_args[0][0], "/usr/bin/qs")
            self.assertEqual(spawn_args[0][1], "-p")
            self.assertTrue(spawn_args[0][2].endswith("DictationPill.qml"))
            self.assertEqual(spawn_kwargs["env"]["NIRI_DICTATE_UI_STATE"], str(fm.ui_state_file))
            self.assertEqual(spawn_kwargs["env"]["NIRI_DICTATE_BOTTOM_MARGIN"], "38")

            state_rec = json.loads(fm.ui_state_file.read_text(encoding="utf-8"))
            self.assertEqual(state_rec["state"], "RECORDING")
            self.assertEqual(state_rec["output"], "eDP-1")

            fm.show_transcribing(notif_id=0)
            self.assertEqual(json.loads(fm.ui_state_file.read_text(encoding="utf-8"))["state"], "TRANSCRIBING")

            fm.show_refining(notif_id=0)
            self.assertEqual(json.loads(fm.ui_state_file.read_text(encoding="utf-8"))["state"], "REFINING")

            fm.show_completed(notif_id=0, typed=True)
            self.assertEqual(json.loads(fm.ui_state_file.read_text(encoding="utf-8"))["state"], "COMPLETED")

            fm.dismiss_pill()
            self.assertEqual(json.loads(fm.ui_state_file.read_text(encoding="utf-8"))["state"], "ABORTED")

    def test_cli_postprocess_success_and_graceful_fallback(self) -> None:
        cfg = self._build_app_config(pp_enabled=True, provider="pill")
        args = argparse.Namespace(key=None, config=None, verbose=False, model=None, subcommand="toggle")

        sm = HardenedStateManager(run_dir=self.run_dir)
        audio_path = self.run_dir / "rec_test.oga"
        audio_path.write_bytes(b"OggS" * 60)

        # Case 1: Post-processor succeeds -> Refined text injected
        sm._write_state_atomic(
            {
                "state": "RECORDING",
                "pid": 99991,
                "audio_file": str(audio_path),
                "notification_id": 0,
                "timestamp": 1000.0,
                "started_at": 1000.0,
            }
        )
        mock_audio = MagicMock()
        mock_audio.stop_recording_and_read.return_value = b"OggS" * 100
        mock_stt = MagicMock()
        mock_stt.transcribe.return_value = "hum então vamos criar a branch feat quer dizer fix login"
        mock_pp = MagicMock()
        mock_pp.refine.return_value = "Vamos criar a branch fix-login."
        mock_fb = MagicMock()
        mock_inj = MagicMock()
        mock_inj.inject.return_value = {"typed": True, "advisory": None}

        with (
            patch.object(HardenedStateManager, "is_pid_valid_recorder", return_value=True),
            patch("niri_dictate.cli.AudioCaptureManager", return_value=mock_audio),
            patch("niri_dictate.cli.GroqSTTClient", return_value=mock_stt),
            patch("niri_dictate.cli.GroqPostProcessor", return_value=mock_pp),
            patch("niri_dictate.cli.VisualFeedbackManager", return_value=mock_fb),
            patch("niri_dictate.cli.WaylandInjector", return_value=mock_inj),
        ):
            rc = handle_toggle(args, cfg)
            self.assertEqual(rc, 0)
            mock_fb.show_refining.assert_called_once_with(notif_id=0)
            mock_inj.inject.assert_called_once_with("Vamos criar a branch fix-login.")

        # Case 2: Post-processor fails -> Raw Whisper text injected (NEVER loses user's speech)
        sm._write_state_atomic(
            {
                "state": "RECORDING",
                "pid": 99992,
                "audio_file": str(audio_path),
                "notification_id": 0,
                "timestamp": 1000.0,
                "started_at": 1000.0,
            }
        )
        mock_pp_fail = MagicMock()
        mock_pp_fail.refine.side_effect = PostProcessError("HTTP 503 Service Unavailable")
        mock_inj_fallback = MagicMock()
        mock_inj_fallback.inject.return_value = {"typed": True, "advisory": None}

        with (
            patch.object(HardenedStateManager, "is_pid_valid_recorder", return_value=True),
            patch("niri_dictate.cli.AudioCaptureManager", return_value=mock_audio),
            patch("niri_dictate.cli.GroqSTTClient", return_value=mock_stt),
            patch("niri_dictate.cli.GroqPostProcessor", return_value=mock_pp_fail),
            patch("niri_dictate.cli.VisualFeedbackManager", return_value=mock_fb),
            patch("niri_dictate.cli.WaylandInjector", return_value=mock_inj_fallback),
        ):
            rc = handle_toggle(args, cfg)
            self.assertEqual(rc, 0)
            mock_inj_fallback.inject.assert_called_once_with(
                "hum então vamos criar a branch feat quer dizer fix login"
            )

    def test_abort_with_pill_dismisses_cleanly_without_redundant_toast(self) -> None:
        cfg = self._build_app_config(pp_enabled=True, provider="pill")
        args = argparse.Namespace(key=None, config=None, verbose=False, model=None, subcommand="abort")
        mock_fb = MagicMock()

        with patch("niri_dictate.cli.VisualFeedbackManager", return_value=mock_fb):
            rc = handle_abort(args, cfg)
            self.assertEqual(rc, 0)
            mock_fb.dismiss_pill.assert_called_once()
            mock_fb.show_toast.assert_not_called()


if __name__ == "__main__":
    unittest.main(verbosity=2)

