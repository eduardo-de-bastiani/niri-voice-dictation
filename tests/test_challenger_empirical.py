#!/usr/bin/env python3
"""
Empirical Challenger Stress & Adversarial Suite for niri_dictate
Directly tests live niri_dictate.state.HardenedStateManager and niri_dictate.audio.AudioCaptureManager.
"""

from __future__ import annotations

import errno
import fcntl
import json
import multiprocessing
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

from niri_dictate.audio import AudioCaptureError, AudioCaptureManager, EmptyAudioError
from niri_dictate.config import AudioConfig
from niri_dictate.state import HardenedStateManager


def _worker_parallel_toggle_barrier(run_dir: str, barrier: multiprocessing.Barrier, queue: multiprocessing.Queue, exit_on_contention: bool):
    """Synchronizes all workers with a Barrier to hit handle_toggle at the exact same millisecond."""
    sm = HardenedStateManager(run_dir=run_dir)
    barrier.wait()  # Synchronized release
    try:
        res = sm.handle_toggle(exit_on_contention=exit_on_contention)
        lock_fd = res.get("lock_fd")
        if lock_fd is not None:
            time.sleep(0.08)  # Hold lock for 80ms while peers contend
            sm.release_lock(lock_fd)
        queue.put(("RESULT", res.get("action")))
    except SystemExit as se:
        if se.code == 0:
            queue.put(("DEBOUNCED", 0))
        else:
            queue.put(("EXIT_ERROR", se.code))
    except Exception as e:
        queue.put(("EXCEPTION", type(e).__name__, str(e)))


class TestChallengerEmpirical(unittest.TestCase):
    """Adversarial stress and verification harness for niri_dictate."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="niri_challenger_")
        self.state_file = Path(self.test_dir) / "niri-dictate.state"
        self.lock_file = Path(self.test_dir) / "niri-dictate.lock"
        self.sm = HardenedStateManager(run_dir=self.test_dir, timeout_threshold=30.0)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # =========================================================================
    # 1. High-Concurrency Rapid Toggle Contention
    # =========================================================================

    def test_multiprocess_toggle_contention_exit_mode(self):
        """
        Stress 1.1: 20 parallel processes synchronized via Barrier executing handle_toggle(exit_on_contention=True).
        Verifies:
        - Exactly one process acquires the lock.
        - Exactly 19 processes debounce cleanly with sys.exit(0).
        - Zero unhandled exceptions or non-zero exit codes.
        """
        num_procs = 20
        barrier = multiprocessing.Barrier(num_procs)
        queue = multiprocessing.Queue()
        procs = []

        for _ in range(num_procs):
            p = multiprocessing.Process(
                target=_worker_parallel_toggle_barrier,
                args=(self.test_dir, barrier, queue, True),
            )
            procs.append(p)

        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=5.0)

        for i, p in enumerate(procs):
            self.assertEqual(p.exitcode, 0, f"Worker {i} exited with unexpected code: {p.exitcode}")

        results = []
        while not queue.empty():
            results.append(queue.get())

        acquired = [r for r in results if r[0] == "RESULT" and r[1] == "START_RECORDING"]
        debounced = [r for r in results if r[0] == "DEBOUNCED"]
        exceptions = [r for r in results if r[0] == "EXCEPTION"]

        self.assertEqual(len(exceptions), 0, f"Concurrency raised exceptions: {exceptions}")
        self.assertEqual(len(acquired), 1, "Exactly one process must acquire the lock on clean IDLE start")
        self.assertEqual(len(debounced), num_procs - 1, "All contending processes must cleanly debounce")

    def test_multiprocess_toggle_contention_programmatic_mode(self):
        """
        Stress 1.2: 20 parallel processes synchronized via Barrier with exit_on_contention=False.
        Verifies:
        - Exactly one returns START_RECORDING.
        - Exactly 19 return action 'DEBOUNCED' with lock_fd=None.
        """
        num_procs = 20
        barrier = multiprocessing.Barrier(num_procs)
        queue = multiprocessing.Queue()
        procs = []

        for _ in range(num_procs):
            p = multiprocessing.Process(
                target=_worker_parallel_toggle_barrier,
                args=(self.test_dir, barrier, queue, False),
            )
            procs.append(p)

        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=5.0)

        results = []
        while not queue.empty():
            results.append(queue.get())

        acquired = [r for r in results if r[0] == "RESULT" and r[1] == "START_RECORDING"]
        debounced = [r for r in results if r[0] == "RESULT" and r[1] == "DEBOUNCED"]
        exceptions = [r for r in results if r[0] == "EXCEPTION"]

        self.assertEqual(len(exceptions), 0, f"Exceptions occurred: {exceptions}")
        self.assertEqual(len(acquired), 1)
        self.assertEqual(len(debounced), num_procs - 1)

    # =========================================================================
    # 2. Stale PID Handling & Process Verification
    # =========================================================================

    def test_stale_dead_recorder_pid_self_healing(self):
        """
        Stress 2.1: State is RECORDING with a dead PID (e.g. killed process).
        Verifies handle_toggle detects dead PID, unlinks state file, sweeps orphaned
        audio, and transitions cleanly to START_RECORDING.
        """
        dead_pid = 9999999
        orphan_audio = Path(self.test_dir) / "orphan_dead_rec.oga"
        orphan_audio.write_bytes(b"STALE_AUDIO_DATA")

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({
                "state": "RECORDING",
                "pid": dead_pid,
                "audio_file": str(orphan_audio),
                "timestamp": time.time(),
            }, f)

        res = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res["action"], "START_RECORDING")
            self.assertFalse(orphan_audio.exists(), "Orphaned audio must be swept")
            self.assertFalse(self.sm.is_pid_alive(dead_pid))
            self.assertFalse(self.sm.is_pid_valid_recorder(dead_pid))
        finally:
            self.sm.release_lock(res.get("lock_fd"))

    def test_stale_alive_non_pw_record_pid_rejected(self):
        """
        Stress 2.2: State is RECORDING with an alive PID that is NOT pw-record
        (e.g., a spawned sleep process).
        Verifies /proc/<pid>/comm check fails, self-healing clears state, and
        returns START_RECORDING.
        """
        proc = subprocess.Popen(["sleep", "10"])
        try:
            live_non_recorder_pid = proc.pid
            self.assertTrue(self.sm.is_pid_alive(live_non_recorder_pid))
            self.assertFalse(
                self.sm.is_pid_valid_recorder(live_non_recorder_pid),
                "Alive process with comm 'sleep' must NOT be accepted as valid recorder",
            )

            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump({
                    "state": "RECORDING",
                    "pid": live_non_recorder_pid,
                    "audio_file": str(Path(self.test_dir) / "rec_fake.oga"),
                    "timestamp": time.time(),
                }, f)

            res = self.sm.handle_toggle(exit_on_contention=False)
            try:
                self.assertEqual(
                    res["action"],
                    "START_RECORDING",
                    "Alive non-pw-record process must self-heal to START_RECORDING",
                )
            finally:
                self.sm.release_lock(res.get("lock_fd"))
        finally:
            proc.kill()
            proc.wait()

    def test_pid_validation_race_condition_file_not_found(self):
        """
        Stress 2.3: Race condition where process exits immediately after os.kill(pid, 0)
        and before /proc/<pid>/comm is read.
        Verifies is_pid_valid_recorder does not raise unhandled FileNotFoundError.
        """
        with patch("os.kill", return_value=None):
            with patch("pathlib.Path.exists", return_value=True):
                with patch("pathlib.Path.read_text", side_effect=FileNotFoundError("Process disappeared")):
                    try:
                        valid = self.sm.is_pid_valid_recorder(12345)
                        self.assertFalse(valid, "Process that disappeared during read_text must return False")
                    except FileNotFoundError:
                        self.fail("is_pid_valid_recorder crashed on FileNotFoundError during process race!")

    def test_pid_validation_missing_comm_does_not_return_true(self):
        """
        Stress 2.3b: If /proc/<pid>/comm does NOT exist, is_pid_valid_recorder must NOT return True!
        Line 101 currently returns True as a blind fallback, which is a false-positive hazard.
        """
        with patch("os.kill", return_value=None):
            with patch("pathlib.Path.exists", return_value=False):
                valid = self.sm.is_pid_valid_recorder(12345)
                self.assertFalse(valid, "Missing /proc/<pid>/comm must NOT assume valid pw-record!")

    def test_pid_validation_type_safety(self):
        """
        Stress 2.4: Ensure is_pid_alive and is_pid_valid_recorder reject all invalid types.
        """
        for bad_pid in [None, "1234", [1234], {"pid": 1234}, -5, 0, 3.14]:
            self.assertFalse(self.sm.is_pid_alive(bad_pid))
            self.assertFalse(self.sm.is_pid_valid_recorder(bad_pid))

    # =========================================================================
    # 3. State File Recovery from Corrupted / Non-Dict JSON
    # =========================================================================

    def test_recovery_from_zero_byte_state_file(self):
        """Stress 3.1: 0-byte state file recovers cleanly to START_RECORDING."""
        self.state_file.touch()
        self.assertEqual(self.state_file.stat().st_size, 0)

        res = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res["action"], "START_RECORDING")
        finally:
            self.sm.release_lock(res.get("lock_fd"))

    def test_recovery_from_truncated_malformed_json(self):
        """Stress 3.2: Partial JSON or binary noise recovers cleanly."""
        for malformed_content in [
            '{"state": "RECOR',
            '{"state": ',
            '{',
            '\x00\x01\xff\xfe\x00',
        ]:
            self.state_file.write_bytes(malformed_content.encode("utf-8", errors="replace"))
            res = self.sm.handle_toggle(exit_on_contention=False)
            try:
                self.assertEqual(res["action"], "START_RECORDING")
            finally:
                self.sm.release_lock(res.get("lock_fd"))

    def test_recovery_from_valid_json_non_dict(self):
        """
        Stress 3.3: Valid JSON that is NOT a dictionary:
        - List: `[]`
        - String: `"RECORDING"`
        - Number: `12345`
        - Null: `null`
        - Boolean: `true`
        Verifies handle_toggle handles non-dict JSON gracefully without AttributeError.
        """
        for non_dict_val in [[], "RECORDING", 12345, None, True]:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(non_dict_val, f)

            try:
                res = self.sm.handle_toggle(exit_on_contention=False)
                lock_fd = res.get("lock_fd")
                if lock_fd is not None:
                    self.sm.release_lock(lock_fd)
                self.assertIn(
                    res.get("action"),
                    ("START_RECORDING", "RECOVER_IDLE"),
                    f"Non-dict JSON {non_dict_val!r} failed to recover properly: {res}",
                )
            except AttributeError as e:
                self.fail(f"handle_toggle crashed with AttributeError on non-dict JSON {non_dict_val!r}: {e}")

    def test_recovery_from_corrupted_dict_fields(self):
        """
        Stress 3.4: JSON dictionary with missing, null, or wrong-type fields.
        """
        corrupted_payloads = [
            # Missing pid in RECORDING
            {"state": "RECORDING"},
            # None pid in RECORDING
            {"state": "RECORDING", "pid": None, "audio_file": None},
            # TRANSCRIBING with transcribing_since as None
            {"state": "TRANSCRIBING", "transcribing_pid": os.getpid(), "transcribing_since": None},
            # TRANSCRIBING with transcribing_since as invalid string
            {"state": "TRANSCRIBING", "transcribing_pid": os.getpid(), "transcribing_since": "yesterday"},
            # Unknown state string
            {"state": "BOGUS_STATE_123"},
            # Empty dict
            {},
        ]

        for payload in corrupted_payloads:
            with open(self.state_file, "w", encoding="utf-8") as f:
                json.dump(payload, f)

            try:
                res = self.sm.handle_toggle(exit_on_contention=False)
                lock_fd = res.get("lock_fd")
                if lock_fd is not None:
                    self.sm.release_lock(lock_fd)
                self.assertIn(
                    res.get("action"),
                    ("START_RECORDING", "RECOVER_IDLE", "REJECT_CONCURRENT"),
                    f"Payload {payload} caused unexpected action: {res}",
                )
            except (TypeError, AttributeError, KeyError) as e:
                self.fail(f"handle_toggle crashed on corrupted dict payload {payload}: {e}")

    # =========================================================================
    # 4. Transcription Timeout Self-Healing (>30s)
    # =========================================================================

    def test_transcription_timeout_healing(self):
        """
        Stress 4.1: State is TRANSCRIBING with alive PID, but >30.0s elapsed.
        Verifies self-healing cleans state file, sweeps orphans, returns START_RECORDING.
        """
        orphan = Path(self.test_dir) / "orphan_stt_hang.oga"
        orphan.write_bytes(b"STT_HANG_AUDIO")

        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({
                "state": "TRANSCRIBING",
                "transcribing_pid": os.getpid(),  # Alive
                "transcribing_since": time.time() - 31.0,  # 31s ago (>30s threshold)
            }, f)

        res = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res["action"], "START_RECORDING")
            self.assertFalse(orphan.exists(), "Orphaned audio must be swept upon STT timeout recovery")
        finally:
            self.sm.release_lock(res.get("lock_fd"))

    def test_transcription_in_flight_protection(self):
        """
        Stress 4.2: State is TRANSCRIBING with alive PID, and <30.0s elapsed (e.g. 3.0s).
        Verifies handle_toggle protects active STT and returns REJECT_CONCURRENT.
        """
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({
                "state": "TRANSCRIBING",
                "transcribing_pid": os.getpid(),  # Alive
                "transcribing_since": time.time() - 3.0,  # 3s ago
            }, f)

        res = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res["action"], "REJECT_CONCURRENT")
        finally:
            self.sm.release_lock(res.get("lock_fd"))

    def test_transcription_dead_pid_immediate_healing(self):
        """
        Stress 4.3: State is TRANSCRIBING with dead PID, even if 1.0s elapsed.
        Verifies immediate self-healing to START_RECORDING without waiting for 30s.
        """
        dead_pid = 9999998
        with open(self.state_file, "w", encoding="utf-8") as f:
            json.dump({
                "state": "TRANSCRIBING",
                "transcribing_pid": dead_pid,
                "transcribing_since": time.time() - 1.0,  # only 1s ago!
            }, f)

        res = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res["action"], "START_RECORDING")
        finally:
            self.sm.release_lock(res.get("lock_fd"))

    # =========================================================================
    # 5. Ephemeral Audio Management & Orphan Sweeping
    # =========================================================================

    def test_audio_immediate_inode_unlinking(self):
        """
        Stress 5.1: AudioCaptureManager.stop_recording_and_read unlinks audio immediately
        upon reading, guaranteeing zero disk persistence.
        """
        audio_cfg = AudioConfig()
        audio_mgr = AudioCaptureManager(config=audio_cfg, runtime_dir=self.test_dir)

        test_audio = Path(self.test_dir) / "test_capture.oga"
        test_audio.write_bytes(b"OPUS_HEADER_MOCK_PAYLOAD_1234567890")

        with patch("os.kill", return_value=None):
            bytes_read = audio_mgr.stop_recording_and_read(pid=os.getpid(), audio_path=test_audio)
            self.assertEqual(bytes_read, b"OPUS_HEADER_MOCK_PAYLOAD_1234567890")
            self.assertFalse(test_audio.exists(), "Audio file must be unlinked immediately upon reading!")

    def test_audio_unlink_guarantee_on_empty_audio_error(self):
        """
        Stress 5.2: Even when EmptyAudioError is raised (0 bytes recorded),
        the 0-byte file must still be unlinked from disk in the finally block.
        """
        audio_cfg = AudioConfig()
        audio_mgr = AudioCaptureManager(config=audio_cfg, runtime_dir=self.test_dir)

        empty_audio = Path(self.test_dir) / "empty_capture.oga"
        empty_audio.touch()

        with patch("os.kill", return_value=None):
            with self.assertRaises(EmptyAudioError):
                audio_mgr.stop_recording_and_read(pid=os.getpid(), audio_path=empty_audio)
            self.assertFalse(empty_audio.exists(), "Empty audio file must be unlinked even when exception is raised!")

    def test_audio_orphan_sweeping_preserves_active(self):
        """
        Stress 5.3: sweep_orphaned_audio removes all dangling .oga files except active_path.
        """
        audio_cfg = AudioConfig()
        audio_mgr = AudioCaptureManager(config=audio_cfg, runtime_dir=self.test_dir)

        active = Path(self.test_dir) / "active_rec.oga"
        active.write_bytes(b"ACTIVE")
        orphan1 = Path(self.test_dir) / "dangling_1.oga"
        orphan1.write_bytes(b"ORPHAN1")
        orphan2 = Path(self.test_dir) / "dangling_2.oga"
        orphan2.write_bytes(b"ORPHAN2")

        audio_mgr.sweep_orphaned_audio(active_path=active)
        self.assertTrue(active.exists())
        self.assertFalse(orphan1.exists())
        self.assertFalse(orphan2.exists())

    def test_state_manager_sweep_keyword_argument_compatibility(self):
        """
        Stress 5.4: PROJECT.md interface contract specifies:
        HardenedStateManager.sweep_orphaned_audio(active_path: Path | None = None)
        Verify that sweep_orphaned_audio accepts active_path without TypeError.
        """
        active = Path(self.test_dir) / "active_state.oga"
        active.write_bytes(b"ACTIVE_STATE")
        orphan = Path(self.test_dir) / "orphan_state.oga"
        orphan.write_bytes(b"ORPHAN_STATE")

        try:
            self.sm.sweep_orphaned_audio(active_path=active)
            self.assertTrue(active.exists())
            self.assertFalse(orphan.exists())
        except TypeError as e:
            self.fail(f"sweep_orphaned_audio failed keyword argument active_path check: {e}")

    # =========================================================================
    # 6. File Descriptor Leak Stress Test
    # =========================================================================

    def test_no_file_descriptor_leaks_under_rapid_toggles(self):
        """
        Stress 6.1: Execute 50 sequential toggles and aborts.
        Measure open file descriptors before and after to guarantee zero FD leaks.
        """
        pid = os.getpid()
        fd_dir = Path(f"/proc/{pid}/fd")
        if not fd_dir.exists():
            self.skipTest("/proc/<pid>/fd not accessible")

        initial_fds = set(os.listdir(fd_dir))

        for _ in range(50):
            res = self.sm.handle_toggle(exit_on_contention=False)
            lock_fd = res.get("lock_fd")
            if lock_fd is not None:
                self.sm.release_lock(lock_fd)

            abort_res = self.sm.handle_abort()

        final_fds = set(os.listdir(fd_dir))
        leaked = final_fds - initial_fds
        self.assertEqual(len(leaked), 0, f"Detected leaked file descriptors: {leaked}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
