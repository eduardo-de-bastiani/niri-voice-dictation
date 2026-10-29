"""
Dual-Mode Wayland Output Injector for Niri Voice Dictation.
Implements UTF-8 clipboard copy via wl-copy, active window IPC inspection via niri,
case-insensitive terminal emulator regex detection, wtype keystroke simulation with
an 80ms modifier settle delay, and graceful degradation to clipboard-only mode.
"""

from __future__ import annotations

import json
import logging
import re
import shutil
import subprocess
import time
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .config import InjectionConfig

logger = logging.getLogger("niri-dictate.injector")


class WaylandInjector:
    """Wayland-native output injector with dynamic binary resolution (Invariant 4)."""

    DEFAULT_TERMINAL_PATTERN = (
        r"(?i)(terminal|alacritty|kitty|foot|wezterm|ghostty|ptyxis|xterm|rxvt|contour|tilix|terminator|konsole)"
    )

    def __init__(self, config: InjectionConfig) -> None:
        self.config = config
        self.configured_mode = config.mode
        self.settle_delay_s = float(config.settle_delay_ms) / 1000.0
        self.type_delay_ms = int(config.type_delay_ms)
        self.mime_type = config.mime_type
        self.trim_newline = bool(config.trim_newline)

        # Compile terminal detection regex
        pattern_str = config.terminal_pattern or self.DEFAULT_TERMINAL_PATTERN
        self.terminal_pattern = re.compile(pattern_str)

        # Dynamic binary resolution
        self.wl_copy_bin = self._resolve_bin(config.wl_copy_bin, "wl-copy")
        self.wtype_bin = self._resolve_bin(config.wtype_bin, "wtype")

    @staticmethod
    def _resolve_bin(configured_path: str, fallback_name: str) -> str | None:
        """Resolves binary by explicit path, falling back to PATH lookup only for default/empty paths."""
        if configured_path:
            resolved = shutil.which(configured_path)
            if resolved:
                return resolved
            if configured_path != f"/usr/bin/{fallback_name}" and "/" in configured_path:
                return None
        return shutil.which(fallback_name)

    def resolve_effective_mode(self) -> tuple[str, bool]:
        """
        Dynamically resolves effective mode based on binary availability (Invariant 4).

        Returns:
            Tuple of (effective_mode, is_degraded_fallback).
        """
        if not self.wl_copy_bin:
            raise RuntimeError("Critical dependency missing: /usr/bin/wl-copy is required.")

        if self.configured_mode in ("paste", "type"):
            if not self.wtype_bin:
                logger.warning("wtype binary not found on host. Degrading to 'clipboard_only'.")
                return ("clipboard_only", True)
            return (self.configured_mode, False)

        return ("clipboard_only", False)

    def get_focused_window_app_id(self) -> str | None:
        """
        Queries Niri IPC for focused window app_id.
        Tolerates unmapped surfaces, root workspaces, and empty IPC outputs.
        """
        try:
            res = subprocess.run(
                ["niri", "msg", "--json", "focused-window"],
                capture_output=True,
                text=True,
                check=True,
                timeout=1.0,
            )
            data = json.loads(res.stdout)
            if data and isinstance(data, dict):
                app_id = data.get("app_id")
                return app_id.strip() if isinstance(app_id, str) and app_id.strip() else None
        except Exception as e:
            logger.debug("Could not retrieve focused window from Niri IPC: %s", e)
        return None

    def copy_to_clipboard(self, text: str) -> None:
        """Copies text to Wayland clipboard via wl-copy standard input."""
        if not self.wl_copy_bin:
            raise RuntimeError("wl-copy binary not available.")

        cmd = [self.wl_copy_bin, "--type", self.mime_type]
        if self.trim_newline:
            cmd.append("--trim-newline")

        subprocess.run(
            cmd,
            input=text,
            text=True,
            check=True,
            timeout=2.0,
        )

    def inject_paste(self, is_terminal: bool) -> None:
        """Simulates atomic paste chord via wtype after enforcing the modifier settle delay."""
        if not self.wtype_bin:
            raise RuntimeError("wtype binary is required for paste simulation.")

        # 80ms modifier settle delay: allows physical Mod/Alt keys to release
        time.sleep(self.settle_delay_s)

        if is_terminal:
            # Terminal Paste Chord: Ctrl+Shift+V
            subprocess.run(
                [
                    self.wtype_bin,
                    "-M", "ctrl",
                    "-M", "shift",
                    "-k", "v",
                    "-m", "shift",
                    "-m", "ctrl",
                ],
                check=True,
                timeout=2.0,
            )
        else:
            # Standard GUI Paste Chord: Ctrl+V
            subprocess.run(
                [self.wtype_bin, "-M", "ctrl", "-k", "v", "-m", "ctrl"],
                check=True,
                timeout=2.0,
            )

    def inject_type(self, text: str) -> None:
        """Simulates direct typing character-by-character via wtype."""
        if not self.wtype_bin:
            raise RuntimeError("wtype binary is required for direct typing.")

        time.sleep(self.settle_delay_s)
        cmd = [self.wtype_bin]
        if self.type_delay_ms > 0:
            cmd.extend(["-d", str(self.type_delay_ms)])
        cmd.extend(["--", text])
        subprocess.run(cmd, check=True, timeout=5.0)

    def inject(self, text: str) -> dict[str, Any]:
        """
        Delivers text according to dynamically resolved mode.
        Returns execution telemetry including fallback advisory status.
        """
        effective_mode, is_fallback = self.resolve_effective_mode()

        # 1. Always populate Wayland system clipboard with full UTF-8 fidelity
        self.copy_to_clipboard(text)

        # 2. Execute virtual keystroke simulation if wtype is available
        if effective_mode == "paste":
            app_id = self.get_focused_window_app_id()
            is_terminal = bool(app_id and self.terminal_pattern.search(app_id))
            self.inject_paste(is_terminal=is_terminal)
            return {
                "mode": "paste",
                "typed": True,
                "is_terminal": is_terminal,
                "fallback": False,
                "advisory": None,
            }

        if effective_mode == "type":
            self.inject_type(text)
            return {
                "mode": "type",
                "typed": True,
                "is_terminal": False,
                "fallback": False,
                "advisory": None,
            }

        # Effective mode: clipboard_only
        advisory = (
            "Text copied to clipboard (Ctrl+V to paste). Install 'wtype' for automatic typing (sudo apt install wtype)."
            if is_fallback
            else "Text copied to clipboard (Ctrl+V to paste)"
        )
        return {
            "mode": "clipboard_only",
            "typed": False,
            "is_terminal": False,
            "fallback": is_fallback,
            "advisory": advisory,
        }

    def inject_text(self, text: str) -> bool:
        """
        Contract method specified in PROJECT.md.
        Delivers text and returns True if typed, False if clipboard-only.
        """
        result = self.inject(text)
        return bool(result.get("typed", False))
