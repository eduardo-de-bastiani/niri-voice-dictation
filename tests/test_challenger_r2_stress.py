#!/usr/bin/env python3
"""
Empirical Stress Test Suite for Niri Voice Dictation (dictation_challenger_r2_2)
Empirically stress-tests:
1. State recovery from corrupted/non-dict JSON (`[]`, `"string"`, `null`, `123`, `true`),
   null/invalid `transcribing_since`, negative/zero PIDs, unreadable `/proc/<pid>/comm`,
   and dual parameter `active_path`/`active_file` in sweep_orphaned_audio.
2. Dense Portuguese technical text token estimation under Tier 2 static fallback heuristic,
   verifying GPT-2 tokens strictly stay <= 200 with >= 24 tokens headroom under 224.
3. WaylandInjector terminal regex matching for terminator, konsole, org.kde.konsole vs GUI apps.
4. CLI injection exception safety: verifying state is cleared and error shown when injector.inject() raises.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import signal
import subprocess
import sys
import tempfile
import time
import unittest
from pathlib import Path
from unittest.mock import MagicMock, patch

_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

import tiktoken

from niri_dictate.audio import AudioCaptureManager
from niri_dictate.cli import handle_toggle
from niri_dictate.config import (
    AppConfig,
    AudioConfig,
    CompositorConfig,
    FeedbackConfig,
    GeneralConfig,
    GroqConfig,
    InjectionConfig,
)
from niri_dictate.feedback import VisualFeedbackManager
from niri_dictate.injector import WaylandInjector
from niri_dictate.prompt import PrimingPromptBuilder
from niri_dictate.state import HardenedStateManager


class TestStateRecoveryStress(unittest.TestCase):
    """Empirical stress tests for HardenedStateManager robustness and recovery."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="niri_stress_state_")
        self.state_file = Path(self.test_dir) / "niri-dictate.state"
        self.lock_file = Path(self.test_dir) / "niri-dictate.lock"
        self.sm = HardenedStateManager(run_dir=self.test_dir, timeout_threshold=30.0)

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_corrupted_json_recovery_non_dict_payloads(self):
        """Verify corrupted/non-dict JSON payloads ([] , 'string', null, 123, true, false) are recovered."""
        non_dict_payloads = [
            b"[]",
            b'"just a string"',
            b"null",
            b"12345",
            b"true",
            b"false",
            b'{"incomplete": ',
            b"{bad json}",
            b"",
        ]

        for payload in non_dict_payloads:
            with self.subTest(payload=payload):
                # Write corrupted/non-dict content directly to state file
                self.state_file.write_bytes(payload)
                self.assertTrue(self.state_file.exists())

                # get_status() without mutation should report CORRUPTED
                status = self.sm.get_status()
                self.assertEqual(
                    status.get("state"),
                    "CORRUPTED",
                    f"get_status failed to report CORRUPTED on payload {payload!r}",
                )

                # handle_toggle() should heal corrupted state and transition to START_RECORDING
                res = self.sm.handle_toggle(exit_on_contention=False)
                self.assertEqual(
                    res.get("action"),
                    "START_RECORDING",
                    f"handle_toggle failed to recover to START_RECORDING for payload {payload!r}",
                )
                self.assertIsNotNone(res.get("lock_fd"))
                self.assertFalse(
                    self.state_file.exists(),
                    f"Corrupted state file was not purged during self-healing: {payload!r}",
                )
                self.sm.release_lock(res.get("lock_fd"))

    def test_null_and_invalid_transcribing_since(self):
        """Verify null, string, negative, zero, and collection transcribing_since values are self-healed."""
        invalid_timestamps = [
            None,
            "invalid_timestamp_str",
            -100.0,
            0.0,
            float("-inf"),
            [],
            {},
        ]

        for ts in invalid_timestamps:
            with self.subTest(timestamp=ts):
                state_data = {
                    "state": "TRANSCRIBING",
                    "pid": os.getpid(),  # Alive PID so liveness doesn't mask timestamp check
                    "transcribing_pid": os.getpid(),
                    "transcribing_since": ts,
                    "notification_id": 99,
                }
                with open(self.state_file, "w", encoding="utf-8") as f:
                    json.dump(state_data, f)

                # Status check should mark it STALE_TRANSCRIBING
                status = self.sm.get_status()
                self.assertEqual(
                    status.get("state"),
                    "STALE_TRANSCRIBING",
                    f"get_status did not detect stale transcription for timestamp {ts!r}",
                )

                # handle_toggle should recover and start recording
                res = self.sm.handle_toggle(exit_on_contention=False)
                try:
                    self.assertEqual(
                        res.get("action"),
                        "START_RECORDING",
                        f"handle_toggle failed to self-heal stale timestamp {ts!r}",
                    )
                finally:
                    if res.get("lock_fd") is not None:
                        self.sm.release_lock(res.get("lock_fd"))

    def test_nan_transcribing_since_behavior(self):
        """
        Empirically document IEEE 754 NaN timestamp behavior:
        - If PID is dead: recovers to START_RECORDING.
        - If PID is alive: get_status reports STALE_TRANSCRIBING, but handle_toggle
          returns REJECT_CONCURRENT because (nan <= 0.0) and (delta > 30.0) both evaluate to False.
        """
        # Case A: Dead PID + NaN timestamp -> recovers
        state_data_dead = {
            "state": "TRANSCRIBING",
            "pid": 9999999,
            "transcribing_pid": 9999999,
            "transcribing_since": float("nan"),
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state_data_dead, f)

        res_dead = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res_dead.get("action"), "START_RECORDING")
        finally:
            if res_dead.get("lock_fd") is not None:
                self.sm.release_lock(res_dead.get("lock_fd"))

        # Case B: Alive PID + NaN timestamp -> status is STALE, but toggle returns REJECT_CONCURRENT
        state_data_alive = {
            "state": "TRANSCRIBING",
            "pid": os.getpid(),
            "transcribing_pid": os.getpid(),
            "transcribing_since": float("nan"),
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state_data_alive, f)

        status = self.sm.get_status()
        self.assertEqual(status.get("state"), "STALE_TRANSCRIBING")

        res_alive = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res_alive.get("action"), "START_RECORDING")
        finally:
            if res_alive.get("lock_fd") is not None:
                self.sm.release_lock(res_alive.get("lock_fd"))

    def test_missing_transcribing_since_key(self):
        """Verify TRANSCRIBING state missing the transcribing_since key completely is self-healed."""
        state_data = {
            "state": "TRANSCRIBING",
            "pid": os.getpid(),
            "transcribing_pid": os.getpid(),
        }
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump(state_data, f)

        status = self.sm.get_status()
        self.assertEqual(status.get("state"), "STALE_TRANSCRIBING")

        res = self.sm.handle_toggle(exit_on_contention=False)
        self.assertEqual(res.get("action"), "START_RECORDING")
        self.sm.release_lock(res.get("lock_fd"))

    def test_pid_bounds_and_type_safety(self):
        """Verify negative, zero, non-integer, and boolean PIDs are strictly rejected."""
        invalid_pids = [
            0,
            -1,
            -9999,
            None,
            "1234",
            True,  # In Python isinstance(True, int) is True, must check bool
            False,
            3.1415,
            [],
            {},
        ]

        for pid in invalid_pids:
            with self.subTest(pid=pid):
                self.assertFalse(
                    self.sm.is_pid_alive(pid),
                    f"is_pid_alive accepted invalid PID: {pid!r}",
                )
                self.assertFalse(
                    self.sm.is_pid_valid_recorder(pid),
                    f"is_pid_valid_recorder accepted invalid PID: {pid!r}",
                )

    def test_unreadable_proc_comm_safety(self):
        """Verify /proc/<pid>/comm read failures (permissions, missing, empty, wrong comm) degrade safely."""
        pid = os.getpid()

        # 1. Non-pw-record process (our own python test runner process)
        self.assertTrue(self.sm.is_pid_alive(pid))
        self.assertFalse(
            self.sm.is_pid_valid_recorder(pid),
            "Python process should not match pw-record",
        )

        # 2. PermissionError reading /proc/<pid>/comm
        with patch.object(Path, "read_text", side_effect=PermissionError("Mock Permission Denied")):
            self.assertFalse(self.sm.is_pid_valid_recorder(pid))

        # 3. FileNotFoundError reading /proc/<pid>/comm (process died between kill(0) and read)
        with patch.object(Path, "read_text", side_effect=FileNotFoundError("Mock proc vanished")):
            self.assertFalse(self.sm.is_pid_valid_recorder(pid))

        # 4. ProcessLookupError
        with patch.object(Path, "read_text", side_effect=ProcessLookupError("Mock no such process")):
            self.assertFalse(self.sm.is_pid_valid_recorder(pid))

        # 5. Empty /proc/<pid>/comm
        with patch.object(Path, "read_text", return_value=""):
            self.assertFalse(self.sm.is_pid_valid_recorder(pid))

        # 6. /proc/<pid>/comm exists=False
        with patch.object(Path, "exists", return_value=False):
            self.assertFalse(self.sm.is_pid_valid_recorder(pid))

    def test_sweep_orphaned_audio_dual_parameter_compatibility(self):
        """
        Verify sweep_orphaned_audio correctly honors active_path, active_file, both, or neither,
        preserving the active recording while unlinking all orphaned audio files.
        """
        # Create multiple fake .oga audio files in runtime dir
        file_active = Path(self.test_dir) / "rec_active.oga"
        file_orphan1 = Path(self.test_dir) / "rec_orphan1.oga"
        file_orphan2 = Path(self.test_dir) / "rec_orphan2.oga"

        def _reset_files():
            file_active.write_bytes(b"active audio")
            file_orphan1.write_bytes(b"orphan audio 1")
            file_orphan2.write_bytes(b"orphan audio 2")

        # Case 1: active_path as Path
        _reset_files()
        self.sm.sweep_orphaned_audio(active_path=file_active)
        self.assertTrue(file_active.exists())
        self.assertFalse(file_orphan1.exists())
        self.assertFalse(file_orphan2.exists())

        # Case 2: active_file as str
        _reset_files()
        self.sm.sweep_orphaned_audio(active_file=str(file_active))
        self.assertTrue(file_active.exists())
        self.assertFalse(file_orphan1.exists())
        self.assertFalse(file_orphan2.exists())

        # Case 3: Both parameters passed (active_path takes precedence if different)
        _reset_files()
        self.sm.sweep_orphaned_audio(active_file=file_orphan1, active_path=file_active)
        self.assertTrue(file_active.exists())
        self.assertFalse(file_orphan1.exists())
        self.assertFalse(file_orphan2.exists())

        # Case 4: Neither passed -> all orphaned .oga files are swept
        _reset_files()
        self.sm.sweep_orphaned_audio()
        self.assertFalse(file_active.exists())
        self.assertFalse(file_orphan1.exists())
        self.assertFalse(file_orphan2.exists())


class TestPromptTier2PortugueseStress(unittest.TestCase):
    """
    Empirical stress tests for prompt token bounds under Tier 2 static fallback heuristic.
    Verifies that dense Portuguese technical text strictly stays <= 200 GPT-2 tokens
    with >= 24 tokens headroom under Whisper's 224 ceiling.
    """

    def setUp(self):
        self.enc = tiktoken.get_encoding("gpt2")

    def test_tier2_dense_portuguese_diacritics_base_prompt(self):
        """
        Verify dense Portuguese technical vocabulary in base prompt stays <= 200 tokens
        and has >= 24 tokens headroom under Tier 2 fallback.
        """
        dense_portuguese_base = (
            "Código, configuração, autenticação, requisição, autorização, exceção, "
            "implementação, extensão, abstração, otimização, persistência, concorrência, "
            "sincronização, transação, comunicação, distribuição, validação, integridade, "
            "manutenção, substituição, restrição, avaliação, execução, compilação, "
            "direção, relação, função, produção, recursão, expressão, solução, condição."
        )

        builder = PrimingPromptBuilder(
            base_prompt=dense_portuguese_base,
            max_tokens=200,
        )

        with patch("tiktoken.get_encoding", side_effect=ImportError("tiktoken mocked unavailable")):
            prompt = builder.build_prompt()

        tokens = len(self.enc.encode(prompt))
        self.assertLessEqual(tokens, 200, f"Tokens {tokens} exceeded safety target 200")
        self.assertGreaterEqual(224 - tokens, 24, f"Safety headroom {224 - tokens} is less than 24")
        self.assertGreater(len(prompt), 30)

    def test_tier2_dense_portuguese_custom_keywords(self):
        """
        Verify base prompt + dense Portuguese custom keywords with diacritics
        strictly bounds GPT-2 tokens <= 200 under Tier 2 fallback.
        """
        keywords = [
            "função", "produção", "recursão", "expressão", "direção", "solução",
            "condição", "iteração", "projeção", "precisão", "geração", "relação",
            "ação", "seção", "criação", "edição", "reação", "opção", "fração",
        ] * 5

        builder = PrimingPromptBuilder(
            custom_keywords=keywords,
            max_tokens=200,
        )

        with patch("tiktoken.get_encoding", side_effect=ImportError("tiktoken mocked unavailable")):
            prompt = builder.build_prompt()

        tokens = len(self.enc.encode(prompt))
        self.assertLessEqual(tokens, 200, f"Tokens {tokens} exceeded safety target 200")
        self.assertGreaterEqual(224 - tokens, 24, f"Safety headroom {224 - tokens} is less than 24")

    def test_tier2_extreme_diacritic_repetition(self):
        """
        Verify extreme adversarial diacritic repetition (1000+ characters of dense accents)
        does not breach token ceiling under Tier 2 fallback.
        """
        adversarial_text = "ação seção direção função opção fração relação criação adição conexão edição " * 20

        builder = PrimingPromptBuilder(
            base_prompt=adversarial_text,
            max_tokens=200,
        )

        with patch("tiktoken.get_encoding", side_effect=ImportError("tiktoken mocked unavailable")):
            prompt = builder.build_prompt()

        tokens = len(self.enc.encode(prompt))
        self.assertLessEqual(tokens, 200, f"Tokens {tokens} exceeded safety target 200")
        self.assertGreaterEqual(224 - tokens, 24, f"Safety headroom {224 - tokens} is less than 24")

    def test_tier2_tight_token_limit_scaling(self):
        """Verify dynamic scaling when token_limit is tightened below 200 (e.g. 150)."""
        adversarial_text = "ação seção direção função opção fração relação criação adição conexão edição " * 10

        builder = PrimingPromptBuilder(
            base_prompt=adversarial_text,
            max_tokens=150,
        )

        with patch("tiktoken.get_encoding", side_effect=ImportError("tiktoken mocked unavailable")):
            prompt = builder.build_prompt(max_tokens=150)

        tokens = len(self.enc.encode(prompt))
        self.assertLessEqual(tokens, 150, f"Tokens {tokens} exceeded tightened limit 150")
        self.assertGreaterEqual(224 - tokens, 74)


class TestWaylandInjectorTerminalMatchingStress(unittest.TestCase):
    """Empirical tests for WaylandInjector terminal vs GUI regex classification."""

    def setUp(self):
        self.cfg = InjectionConfig()
        self.injector = WaylandInjector(self.cfg)
        self.pattern = self.injector.terminal_pattern

    def test_terminal_matching_standard_and_expanded(self):
        """Verify terminator, konsole, org.kde.konsole, and standard terminals all match regex."""
        terminals = [
            "terminator",
            "konsole",
            "org.kde.konsole",
            "Terminator",
            "KONSOLE",
            "org.kde.Konsole",
            "ptyxis",
            "org.gnome.Ptyxis",
            "alacritty",
            "Alacritty",
            "kitty",
            "foot",
            "wezterm",
            "ghostty",
            "xterm",
            "rxvt",
            "contour",
            "tilix",
            "gnome-terminal",
            "xfce4-terminal",
            "mate-terminal",
            "lxterminal",
        ]

        for term in terminals:
            with self.subTest(terminal=term):
                self.assertIsNotNone(
                    self.pattern.search(term),
                    f"Terminal regex failed to match terminal app_id: {term!r}",
                )

    def test_gui_apps_never_match_terminal_regex(self):
        """Verify GUI applications are strictly rejected by terminal regex."""
        gui_apps = [
            "antigravity-ide",
            "code",
            "code-oss",
            "google-chrome",
            "brave-browser",
            "firefox",
            "slack",
            "discord",
            "spotify",
            "obsidian",
            "libreoffice",
            "thunderbird",
            "org.kde.kate",
            "org.gnome.TextEditor",
            "org.gnome.Nautilus",
            "zed",
        ]

        for app in gui_apps:
            with self.subTest(app=app):
                self.assertIsNone(
                    self.pattern.search(app),
                    f"Terminal regex falsely matched GUI app_id: {app!r}",
                )


class TestCLIInjectionExceptionSafetyStress(unittest.TestCase):
    """Empirical tests verifying CLI handle_toggle exception safety during injection."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="niri_stress_cli_")
        self.state_file = Path(self.test_dir) / "niri-dictate.state"
        self.lock_file = Path(self.test_dir) / "niri-dictate.lock"
        self.audio_file = Path(self.test_dir) / "test_audio.oga"
        self.audio_file.write_bytes(b"dummy_opus_audio_bytes" * 10)

        self.config = AppConfig(
            general=GeneralConfig(runtime_dir=self.test_dir),
            compositor=CompositorConfig(),
            audio=AudioConfig(),
            groq=GroqConfig(api_key="gsk_test_key_emp"),
            injection=InjectionConfig(),
            feedback=FeedbackConfig(),
        )
        self.args = argparse.Namespace(key="gsk_test_key_emp")

    def tearDown(self):
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_injection_exception_clears_state_and_shows_error(self):
        """
        Verify that when WaylandInjector.inject raises an exception:
        1. handle_toggle catches it and returns 1.
        2. State is purged from TRANSCRIBING to IDLE (state_file unlinked).
        3. feedback_mgr.show_error is called with the notification id and error details.
        4. Next handle_toggle cleanly starts recording without being blocked in TRANSCRIBING.
        """
        sm = HardenedStateManager(run_dir=self.test_dir)

        # 1. Setup RECORDING state
        sm.transition_to_recording(
            lock_fd=None,
            pid=os.getpid(),
            audio_path=self.audio_file,
            notification_id=42,
        )
        self.assertTrue(self.state_file.exists())

        # 2. Mock STT client and AudioCaptureManager to simulate successful transcription
        mock_audio_mgr = MagicMock()
        mock_audio_mgr.stop_recording_and_read.return_value = b"valid audio bytes header" * 10

        mock_stt_client = MagicMock()
        mock_stt_client.transcribe.return_value = "transcribed voice text"

        mock_feedback_mgr = MagicMock()

        # 3. Mock WaylandInjector to raise an exception during inject()
        mock_injector = MagicMock()
        mock_injector.inject.side_effect = RuntimeError("Wayland compositor pipe broken")

        with (
            patch("niri_dictate.cli.AudioCaptureManager", return_value=mock_audio_mgr),
            patch("niri_dictate.cli.GroqSTTClient", return_value=mock_stt_client),
            patch("niri_dictate.cli.VisualFeedbackManager", return_value=mock_feedback_mgr),
            patch("niri_dictate.cli.WaylandInjector", return_value=mock_injector),
            patch.object(HardenedStateManager, "is_pid_valid_recorder", return_value=True),
        ):
            # Run handle_toggle which will trigger STOP_AND_TRANSCRIBE -> INJECT (fails)
            ret = handle_toggle(self.args, self.config)

            # Assert returncode is 1
            self.assertEqual(ret, 1, "handle_toggle should return 1 on injection failure")

            # Assert state was cleared (unlinked)
            self.assertFalse(
                self.state_file.exists(),
                "State file was not unlinked after injection exception! Engine left in dirty state.",
            )

            # Assert feedback error was shown
            mock_feedback_mgr.show_error.assert_called_once()
            call_kwargs = mock_feedback_mgr.show_error.call_args[1]
            self.assertEqual(call_kwargs.get("notif_id"), 42)
            self.assertIn("Wayland compositor pipe broken", call_kwargs.get("message", ""))

        # 4. Empirical proof of self-healing: next toggle must start a new recording
        with (
            patch("niri_dictate.cli.AudioCaptureManager") as next_audio_cls,
            patch("niri_dictate.cli.VisualFeedbackManager") as next_fb_cls,
        ):
            next_audio = MagicMock()
            next_audio.start_recording.return_value = (12345, Path(self.test_dir) / "rec_new.oga")
            next_audio_cls.return_value = next_audio
            next_fb = MagicMock()
            next_fb.show_recording.return_value = 43
            next_fb_cls.return_value = next_fb

            next_ret = handle_toggle(self.args, self.config)
            self.assertEqual(next_ret, 0)
            self.assertTrue(self.state_file.exists())
            status = sm.get_status()
            # The newly created state should be RECORDING
            with open(self.state_file, "r") as f:
                saved = json.load(f)
            self.assertEqual(saved.get("state"), "RECORDING")
            self.assertEqual(saved.get("pid"), 12345)


if __name__ == "__main__":
    unittest.main(verbosity=2)
