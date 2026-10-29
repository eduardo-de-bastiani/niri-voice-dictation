"""
Wayland Desktop Visual Feedback Manager for Niri Voice Dictation.
Supports:
- "pill": Wispr Flow-style floating bottom-center Quickshell overlay (WlrLayer.Overlay + live waveform)
- "dms": Dank Material Shell Toast IPC + Freedesktop notifications (notify-send -r)
- "notify-send": Standard Freedesktop notifications
- "none": Silent operation
Guarantees zero keyboard focus interruption (WlrKeyboardFocus.None) across all providers.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import FeedbackConfig

logger = logging.getLogger("niri-dictate.feedback")


class VisualFeedbackManager:
    """
    Manages non-focus-stealing visual indicators across operational states:
    State A: RECORDING (live microphone waveform pill or persistent notification)
    State B1: TRANSCRIBING (animated processing wave pill or in-place notification update)
    State B2: REFINING (LLM post-processing wave state)
    State C1: COMPLETED (smooth pill fade-out or auto-dismiss notification)
    State C2: ERROR (error flash + critical alert)
    """

    def __init__(
        self,
        config: FeedbackConfig,
        runtime_dir: Path | str | None = None,
        audio_source: str = "",
    ) -> None:
        self.config = config
        self.audio_source = audio_source
        self._notify_send_bin = shutil.which("notify-send")
        self._dms_bin = shutil.which("dms")
        self._qs_bin = self._resolve_qs(getattr(config, "quickshell_bin", "/usr/bin/qs"))
        self._project_root = Path(__file__).resolve().parent.parent
        self._pill_qml = self._project_root / "ui" / "DictationPill.qml"
        self._cli_script = self._project_root / "scripts" / "niri-dictate"

        if runtime_dir is not None:
            self.runtime_dir = Path(runtime_dir)
        else:
            xdg = os.environ.get("XDG_RUNTIME_DIR")
            self.runtime_dir = (
                Path(xdg) / "niri-dictate"
                if xdg
                else Path(tempfile.gettempdir()) / f"niri-dictate-{os.getuid()}"
            )
        self.ui_state_file = self.runtime_dir / "ui_state.json"

    @staticmethod
    def _resolve_qs(configured: str) -> str | None:
        if configured and shutil.which(configured):
            return configured
        return shutil.which("qs") or shutil.which("quickshell")

    @property
    def _use_pill(self) -> bool:
        return (
            self.config.provider == "pill"
            and self._qs_bin is not None
            and self._pill_qml.is_file()
        )

    @staticmethod
    def _get_focused_output() -> str:
        """Queries Niri IPC for the currently focused monitor name (e.g., eDP-1, HDMI-A-1)."""
        try:
            res = subprocess.run(
                ["niri", "msg", "--json", "focused-output"],
                capture_output=True,
                text=True,
                timeout=0.5,
            )
            if res.returncode == 0 and res.stdout.strip():
                data = json.loads(res.stdout)
                if isinstance(data, dict) and isinstance(data.get("name"), str):
                    return data["name"]
        except Exception:
            pass
        return ""

    def _write_ui_state(self, state: str, extra: dict[str, Any] | None = None) -> None:
        """Atomically writes UI state JSON for Quickshell FileView watcher."""
        try:
            self.runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
            payload: dict[str, Any] = {
                "state": state,
                "ts": time.time(),
            }
            if extra:
                payload.update(extra)
            tmp_path = self.runtime_dir / f".ui_state.{os.getpid()}.tmp"
            tmp_path.write_text(json.dumps(payload), encoding="utf-8")
            os.replace(tmp_path, self.ui_state_file)
        except Exception as exc:
            logger.debug("Failed to write ui_state.json: %s", exc)

    def _spawn_pill_overlay(self) -> None:
        """Spawns detached Quickshell DictationPill.qml overlay if not already active."""
        if not self._qs_bin:
            return
        env = dict(os.environ)
        env["NIRI_DICTATE_UI_STATE"] = str(self.ui_state_file)
        env["NIRI_DICTATE_ROOT"] = str(self._project_root)
        env["NIRI_DICTATE_BIN"] = str(self._cli_script)
        env["NIRI_DICTATE_PYTHON"] = sys.executable or "python3"
        env["NIRI_DICTATE_SOURCE"] = self.audio_source
        env["NIRI_DICTATE_BOTTOM_MARGIN"] = str(getattr(self.config, "pill_bottom_margin", 38))

        try:
            subprocess.Popen(
                [self._qs_bin, "-p", str(self._pill_qml)],
                env=env,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except Exception as exc:
            logger.warning("Failed to launch Quickshell DictationPill: %s", exc)

    def show_recording(self) -> int | None:
        """Displays State A (Recording Active)."""
        if self.config.provider == "none":
            return None

        if self._use_pill:
            output_name = self._get_focused_output()
            self._write_ui_state("RECORDING", {"output": output_name})
            self._spawn_pill_overlay()
            return 0

        notif_id = None
        if self._notify_send_bin:
            cmd = [
                self._notify_send_bin,
                "-p",
                "-t", "0",
                "-a", self.config.app_name,
                "-i", self.config.icon_recording,
                "-e",
                self.config.title_recording,
                self.config.body_recording,
            ]
            try:
                res = subprocess.run(cmd, capture_output=True, text=True, timeout=2.0)
                if res.returncode == 0:
                    out = res.stdout.strip()
                    if out.isdigit():
                        notif_id = int(out)
            except Exception as exc:
                logger.debug("notify-send show_recording failed: %s", exc)

        if self.config.provider in ("dms", "all") and self.config.toast_recording:
            self.show_toast(self.config.toast_recording, level="info")

        return notif_id

    def show_transcribing(self, notif_id: int | None = None) -> None:
        """Displays State B1 (Transcribing in Progress)."""
        if self.config.provider == "none":
            return

        if self._use_pill:
            self._write_ui_state("TRANSCRIBING")
            return

        if not self._notify_send_bin:
            return

        cmd = [
            self._notify_send_bin,
            "-t", "0",
            "-a", self.config.app_name,
            "-i", self.config.icon_transcribing,
            "-e",
        ]
        if notif_id is not None and notif_id > 0:
            cmd.extend(["-r", str(notif_id)])
        cmd.extend([self.config.title_transcribing, self.config.body_transcribing])

        try:
            subprocess.run(cmd, capture_output=True, timeout=2.0)
        except Exception as exc:
            logger.debug("notify-send show_transcribing failed: %s", exc)

    def show_refining(self, notif_id: int | None = None) -> None:
        """Displays State B2 (LLM Post-Processing / Refining)."""
        if self.config.provider == "none":
            return

        if self._use_pill:
            self._write_ui_state("REFINING")
            return

        if not self._notify_send_bin:
            return

        cmd = [
            self._notify_send_bin,
            "-t", "0",
            "-a", self.config.app_name,
            "-i", self.config.icon_transcribing,
            "-e",
        ]
        if notif_id is not None and notif_id > 0:
            cmd.extend(["-r", str(notif_id)])
        cmd.extend(["Refining text...", "Cleaning transcript via Groq LLM..."])

        try:
            subprocess.run(cmd, capture_output=True, timeout=2.0)
        except Exception as exc:
            logger.debug("notify-send show_refining failed: %s", exc)

    def show_completed(
        self,
        notif_id: int | None = None,
        typed: bool = True,
        advisory: str | None = None,
    ) -> None:
        """Displays State C1 (Completed / Text Injected)."""
        if self._use_pill:
            self._write_ui_state("COMPLETED")
            if advisory and not typed:
                self.show_toast("Text copied to clipboard (Ctrl+V to paste)", level="info")
            return

        if self.config.provider != "none" and self._notify_send_bin:
            cmd = [
                self._notify_send_bin,
                "-t", str(self.config.notification_timeout_ms),
                "-a", self.config.app_name,
                "-i", self.config.icon_success,
                "-e",
            ]
            if notif_id is not None and notif_id > 0:
                cmd.extend(["-r", str(notif_id)])
            cmd.extend([self.config.title_completed, self.config.body_completed])

            try:
                subprocess.run(cmd, capture_output=True, timeout=2.0)
            except Exception as exc:
                logger.debug("notify-send show_completed failed: %s", exc)

        if advisory:
            self.show_toast(advisory, level="info")
        elif not typed and self.config.toast_fallback_wtype:
            self.show_toast(self.config.toast_fallback_wtype, level="info")

    def show_error(self, notif_id: int | None = None, message: str = "") -> None:
        """Displays State C2 (Dictation Error)."""
        if self._use_pill:
            self._write_ui_state("ERROR", {"message": message})
            toast_text = f"{self.config.title_error}: {message}" if message else self.config.title_error
            self.show_toast(toast_text, level="error")
            return

        body = f"{self.config.body_error_prefix}{message}" if message else self.config.title_error
        if self.config.provider != "none" and self._notify_send_bin:
            cmd = [
                self._notify_send_bin,
                "-t", "5000",
                "-u", "critical",
                "-a", self.config.app_name,
                "-i", self.config.icon_error,
                "-e",
            ]
            if notif_id is not None and notif_id > 0:
                cmd.extend(["-r", str(notif_id)])
            cmd.extend([self.config.title_error, body])

            try:
                subprocess.run(cmd, capture_output=True, timeout=2.0)
            except Exception as exc:
                logger.debug("notify-send show_error failed: %s", exc)

        if self.config.provider in ("dms", "all"):
            toast_text = (
                f"{self.config.title_error}: {message}" if message else self.config.title_error
            )
            self.show_toast(toast_text, level="error")

    def dismiss_pill(self) -> None:
        """Dismisses any active Quickshell pill cleanly (used on abort)."""
        if self._use_pill or self.ui_state_file.exists():
            self._write_ui_state("ABORTED")

    def show_toast(self, message: str, level: str = "info") -> None:
        """Emits a top-edge overlay toast card via DMS IPC."""
        if self.config.provider not in ("dms", "pill", "all") or not self._dms_bin:
            return

        cmd = [self._dms_bin, "ipc", "call", "toast", level, message]
        try:
            subprocess.run(cmd, capture_output=True, timeout=1.0)
        except Exception as exc:
            logger.debug("dms toast failed: %s", exc)
