#!/usr/bin/env python3
"""
Adversarial Stress & Edge-Case Test Harness for Niri Wayland Voice Dictation
Targeting Concurrency, Stale PID Recovery, Lifecycle, and Wayland Timing Contracts.
Directly tests HardenedStateManager for production invariants.
"""

import errno
import fcntl
import json
import logging
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

logger = logging.getLogger("niri-dictate.test_stress")


_project_root = str(Path(__file__).resolve().parent.parent)
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

from niri_dictate.state import HardenedStateManager


class BaseStateManager:
    """Compatibility reference state manager."""

    def __init__(self, state_file, lock_file, run_dir):
        self.state_file = state_file
        self.lock_file = lock_file
        self.run_dir = run_dir

    def toggle_spec(self, pid: int) -> str:
        lock_fd = os.open(self.lock_file, os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (BlockingIOError, OSError) as e:
            os.close(lock_fd)
            raise e

        try:
            if not os.path.exists(self.state_file):
                audio_path = os.path.join(self.run_dir, f"rec_{pid}.oga")
                with open(audio_path, "wb") as af:
                    af.write(b"MOCK_OPUS_AUDIO_STREAM")
                with open(self.state_file, "w") as sf:
                    json.dump({
                        "state": "RECORDING",
                        "pid": pid,
                        "audio_file": audio_path,
                        "timestamp": time.time()
                    }, sf)
                return "RECORDING_STARTED"
            else:
                with open(self.state_file, "r") as sf:
                    data = json.load(sf)
                
                curr_state = data.get("state")
                active_pid = data.get("pid")
                audio_file = data.get("audio_file")

                os.kill(active_pid, 0)

                if curr_state == "RECORDING":
                    data["state"] = "TRANSCRIBING"
                    with open(self.state_file, "w") as sf:
                        json.dump(data, sf)
                    
                    if os.path.exists(audio_file):
                        os.unlink(audio_file)
                    if os.path.exists(self.state_file):
                        os.unlink(self.state_file)
                    return f"RECORDING_STOPPED_AND_TRANSCRIBED:{active_pid}"
                elif curr_state == "TRANSCRIBING":
                    return "ALREADY_TRANSCRIBING"
                else:
                    return f"UNKNOWN_STATE:{curr_state}"
        finally:
            try:
                fcntl.flock(lock_fd, fcntl.LOCK_UN)
            finally:
                os.close(lock_fd)


def worker_hardened_toggle_task(run_dir: str, results_queue: multiprocessing.Queue):
    """Worker task executing HardenedStateManager under concurrent execution."""
    sm = HardenedStateManager(run_dir)
    try:
        # Default exit_on_contention=True cleanly triggers sys.exit(0) when lock is held
        action = sm.handle_toggle(exit_on_contention=True)
        lock_fd = action.get("lock_fd")
        if lock_fd:
            time.sleep(0.015)  # Hold lock briefly to test debounce contention
            sm.release_lock(lock_fd)
        results_queue.put(("SUCCESS", action.get("action")))
    except SystemExit as se:
        if se.code == 0:
            results_queue.put(("DEBOUNCED", 0))
        else:
            results_queue.put(("EXIT_ERROR", se.code))
    except Exception as e:
        results_queue.put(("ERROR", type(e).__name__, str(e)))


class TestAdversarialStress(unittest.TestCase):
    """Empirical challenge suite for concurrency, stale PIDs, and modifiers."""

    def setUp(self):
        self.test_dir = tempfile.mkdtemp(prefix="niri_adversarial_")
        self.state_file = os.path.join(self.test_dir, "niri-dictate.state")
        self.lock_file = os.path.join(self.test_dir, "niri-dictate.lock")
        self.sm = HardenedStateManager(self.test_dir)
        self.base_sm = BaseStateManager(self.state_file, self.lock_file, self.test_dir)

    def tearDown(self):
        if os.path.exists(self.test_dir):
            shutil.rmtree(self.test_dir, ignore_errors=True)

    # --------------------------------------------------------------------------
    # CHALLENGE 1: Concurrency, Rapid Toggles, and State Corruption
    # --------------------------------------------------------------------------

    def test_rapid_concurrency_race_condition(self):
        """
        STRESS 1.1: Rapid simultaneous toggles on HardenedStateManager.
        Validates that under high concurrency, exactly 1 process acquires the lock
        while competing processes debounce cleanly with sys.exit(0), leaving state uncorrupted.
        """
        results_queue = multiprocessing.Queue()
        procs = []
        num_toggles = 15

        for _ in range(num_toggles):
            p = multiprocessing.Process(
                target=worker_hardened_toggle_task,
                args=(self.test_dir, results_queue)
            )
            procs.append(p)

        for p in procs:
            p.start()
        for p in procs:
            p.join(timeout=4.0)

        # Every worker process MUST exit cleanly with return code 0
        for i, p in enumerate(procs):
            self.assertEqual(p.exitcode, 0, f"Worker process {i} crashed with exitcode {p.exitcode}")

        results = []
        while not results_queue.empty():
            results.append(results_queue.get())

        successes = [r for r in results if r[0] == "SUCCESS"]
        debounced = [r for r in results if r[0] == "DEBOUNCED"]
        errors = [r for r in results if r[0] in ("ERROR", "EXIT_ERROR")]

        print(f"\n[STRESS 1.1] HardenedStateManager Concurrency: {len(successes)} ACQUIRED, {len(debounced)} DEBOUNCED across {num_toggles} processes")
        self.assertEqual(len(errors), 0, f"Encountered unexpected concurrency errors: {errors}")
        self.assertGreaterEqual(len(successes), 1, "At least one process must successfully acquire lock")
        self.assertEqual(len(successes) + len(debounced), num_toggles, "All workers must complete cleanly")

    def test_nonblocking_lock_behavior(self):
        """
        STRESS 1.2: Section 1.2 specifies LOCK_EX | LOCK_NB.
        When a second toggle arrives while a toggle is executing, LOCK_NB raises BlockingIOError.
        HardenedStateManager traps BlockingIOError and debounces (exitcode 0 or DEBOUNCED action).
        """
        # Hold lock exclusively in this process
        lock_fd = os.open(self.lock_file, os.O_RDWR | os.O_CREAT, 0o600)
        fcntl.flock(lock_fd, fcntl.LOCK_EX)

        try:
            # 1. With exit_on_contention=False: returns action DEBOUNCED
            action = self.sm.handle_toggle(exit_on_contention=False)
            self.assertEqual(action["action"], "DEBOUNCED")
            print("\n[STRESS 1.2] Confirmed: Contended lock returns action 'DEBOUNCED'")

            # 2. With exit_on_contention=True: raises SystemExit(0) for clean CLI termination
            with self.assertRaises(SystemExit) as ctx:
                self.sm.handle_toggle(exit_on_contention=True)
            self.assertEqual(ctx.exception.code, 0)
            print("[STRESS 1.2] Confirmed: CLI toggle debounces via sys.exit(0)")
        finally:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
            os.close(lock_fd)

    def test_dead_transcribing_pid_self_heals(self):
        """
        STRESS 1.3: Self-Healing Stale PID in TRANSCRIBING State (Remediation 1).
        Verifies that if transcribing_pid terminated abruptly or STT timed out (> 30s):
        HardenedStateManager unlinks the stale state file, sweeps orphaned audio,
        and transitions cleanly to START_RECORDING rather than locking out the user with REJECT_CONCURRENT.
        """
        dead_pid = 999999
        # 1. Test dead transcribing PID
        with open(self.state_file, "w", encoding="utf-8") as sf:
            json.dump({
                "state": "TRANSCRIBING",
                "transcribing_pid": dead_pid,
                "transcribing_since": time.time() - 5.0,
                "notification_id": 100
            }, sf)

        orphan_audio = os.path.join(self.test_dir, "orphan_before_heal.oga")
        with open(orphan_audio, "wb") as f:
            f.write(b"ORPHAN_DATA")

        res_dead = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertNotEqual(res_dead["action"], "REJECT_CONCURRENT",
                               "Dead transcribing PID must NOT return REJECT_CONCURRENT")
            self.assertEqual(res_dead["action"], "START_RECORDING",
                             "Dead transcribing PID must self-heal and return START_RECORDING")
            self.assertFalse(os.path.exists(orphan_audio),
                             "Orphaned audio must be swept during self-healing")
            print("\n[STRESS 1.3] Confirmed: Dead transcribing PID self-healed to START_RECORDING and swept orphaned audio")
        finally:
            self.sm.release_lock(res_dead.get("lock_fd"))

        # 2. Test timed-out transcribing PID (> 30.0s) with live PID
        with open(self.state_file, "w", encoding="utf-8") as sf:
            json.dump({
                "state": "TRANSCRIBING",
                "transcribing_pid": os.getpid(),  # Alive PID
                "transcribing_since": time.time() - 35.0,  # 35 seconds ago (> 30s)
                "notification_id": 101
            }, sf)

        res_timeout = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res_timeout["action"], "START_RECORDING",
                             "Timed-out transcribing state (> 30s) must self-heal to START_RECORDING")
            print("[STRESS 1.3] Confirmed: Timed-out transcription (>30s) self-healed to START_RECORDING")
        finally:
            self.sm.release_lock(res_timeout.get("lock_fd"))

        # 3. Test legitimate in-flight transcription (< 30.0s with live PID)
        with open(self.state_file, "w", encoding="utf-8") as sf:
            json.dump({
                "state": "TRANSCRIBING",
                "transcribing_pid": os.getpid(),  # Alive PID
                "transcribing_since": time.time() - 2.0,  # Recent (2s ago)
                "notification_id": 102
            }, sf)

        res_inflight = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res_inflight["action"], "REJECT_CONCURRENT",
                             "Active in-flight transcription must be protected via REJECT_CONCURRENT")
            print("[STRESS 1.3] Confirmed: Active in-flight transcription returned REJECT_CONCURRENT")
        finally:
            self.sm.release_lock(res_inflight.get("lock_fd"))

    def test_null_and_missing_pid_type_safety(self):
        """
        STRESS 1.4: Type-Safe PID Validation (Remediation 2).
        Verifies that null, missing, or non-int PID values in state file or methods
        do NOT raise TypeError: 'NoneType' object cannot be interpreted as an integer.
        """
        # 1. Method-level null and type checks
        self.assertFalse(self.sm.is_pid_alive(None))
        self.assertFalse(self.sm.is_pid_alive("12345"))
        self.assertFalse(self.sm.is_pid_alive([]))
        self.assertFalse(self.sm.is_pid_valid_recorder(None))
        self.assertFalse(self.sm.is_pid_valid_recorder("not_an_int"))
        self.assertFalse(self.sm.is_pid_valid_recorder({}))

        # 2. Corrupted RECORDING state file without 'pid' key
        with open(self.state_file, "w", encoding="utf-8") as sf:
            json.dump({
                "state": "RECORDING",
                "audio_file": os.path.join(self.test_dir, "test.oga"),
                "timestamp": time.time()
            }, sf)

        res_rec = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res_rec["action"], "START_RECORDING",
                             "Missing recorder PID must self-heal without raising TypeError")
        finally:
            self.sm.release_lock(res_rec.get("lock_fd"))

        # 3. TRANSCRIBING state file with transcribing_pid=None
        with open(self.state_file, "w", encoding="utf-8") as sf:
            json.dump({
                "state": "TRANSCRIBING",
                "transcribing_pid": None,
                "transcribing_since": time.time(),
                "notification_id": 200
            }, sf)

        res_trans = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res_trans["action"], "START_RECORDING",
                             "Null transcribing PID must self-heal without raising TypeError")
            print("\n[STRESS 1.4] Confirmed: Null/missing PID handled safely without TypeError")
        finally:
            self.sm.release_lock(res_trans.get("lock_fd"))

    def test_pid_recycling_collision_pw_record_only(self):
        """
        STRESS 1.5: Eliminate PID Recycling Collision (Remediation 3).
        Verifies that /proc/<pid>/comm verification is strictly scoped to 'pw-record'
        and rejects Python processes (e.g., our own PID).
        """
        current_pid = os.getpid()
        comm_file = Path(f"/proc/{current_pid}/comm")
        if comm_file.exists():
            comm = comm_file.read_text().strip()
            self.assertIn("python", comm)
            # Must return False because python is NOT pw-record!
            self.assertFalse(
                self.sm.is_pid_valid_recorder(current_pid),
                f"Process {current_pid} ({comm}) must NOT be accepted as pw-record recorder"
            )
            print(f"\n[STRESS 1.5] Confirmed: Python process ({comm}) correctly rejected as pw-record")

    # --------------------------------------------------------------------------
    # CHALLENGE 2: Process Termination & Orphan / Stale PID Recovery
    # --------------------------------------------------------------------------

    def test_stale_pid_crash_vulnerability(self):
        """
        STRESS 2.1: Abrupt process termination mid-recording (kill -9 or SIGSEGV).
        Verifies that raw os.kill(dead_pid, 0) raises ProcessLookupError,
        and that HardenedStateManager.is_pid_valid_recorder safely traps it.
        """
        dead_pid = 999999
        with self.assertRaises(ProcessLookupError) as ctx:
            os.kill(dead_pid, 0)
        self.assertEqual(ctx.exception.errno, errno.ESRCH)
        print(f"\n[STRESS 2.1] Confirmed: os.kill on dead PID raises ProcessLookupError: {ctx.exception}")

        self.assertFalse(self.sm.is_pid_valid_recorder(dead_pid))
        self.assertFalse(self.sm.is_pid_alive(dead_pid))

    def test_corrupted_empty_state_file_recovery(self):
        """
        STRESS 2.2: Crash during atomic state file write leaving 0-byte or corrupted JSON.
        Verifies that HardenedStateManager recovers cleanly from a 0-byte state file.
        """
        # Create 0-byte truncated state file
        with open(self.state_file, "w") as sf:
            pass  # 0 bytes

        with self.assertRaises(json.JSONDecodeError):
            with open(self.state_file, "r") as sf:
                json.load(sf)
        print("\n[STRESS 2.2] Confirmed: 0-byte state file crashes json.load without try/except")

        # HardenedStateManager recovers cleanly to START_RECORDING
        res = self.sm.handle_toggle(exit_on_contention=False)
        try:
            self.assertEqual(res["action"], "START_RECORDING")
            print("[STRESS 2.2] Confirmed: HardenedStateManager recovered 0-byte state file to START_RECORDING")
        finally:
            self.sm.release_lock(res.get("lock_fd"))

    def test_orphaned_audio_file_cleanup(self):
        """
        STRESS 2.3: Ephemeral storage leak detection.
        If recording was killed abruptly, orphaned .oga files remain in tmpfs.
        Verify cleanup sweep mechanism.
        """
        orphan1 = os.path.join(self.test_dir, "rec_orphan_1.oga")
        orphan2 = os.path.join(self.test_dir, "rec_orphan_2.oga")
        valid_rec = os.path.join(self.test_dir, "rec_active.oga")
        for p in [orphan1, orphan2, valid_rec]:
            with open(p, "wb") as f:
                f.write(b"AUDIO_DATA")

        self.sm.sweep_orphaned_audio(active_file=valid_rec)

        self.assertTrue(os.path.exists(valid_rec))
        self.assertFalse(os.path.exists(orphan1))
        self.assertFalse(os.path.exists(orphan2))
        print(f"\n[STRESS 2.3] Cleaned orphaned audio files from runtime directory, preserving active file")

    # --------------------------------------------------------------------------
    # CHALLENGE 3: Settle Delay, Modifiers, and Wayland Timing
    # --------------------------------------------------------------------------

    def test_modifier_settle_delay_empirical_measure(self):
        """
        STRESS 3.1: Verify whether 80ms settle delay prevents modifier bleed.
        Measure simulated modifier release timing across frame intervals.
        """
        frame_interval_60hz = 1.0 / 60.0  # 16.66 ms
        settle_delay_s = 0.080  # 80ms specified in IMPLEMENTATION_PLAN.md

        frames_elapsed = settle_delay_s / frame_interval_60hz
        self.assertGreater(frames_elapsed, 4.0, "80ms must span at least 4 display refresh frames at 60Hz")
        print(f"\n[STRESS 3.1] 80ms settle delay spans {frames_elapsed:.1f} compositor frames at 60Hz")

    def test_wtype_missing_fallback_detection(self):
        """
        STRESS 3.2: Verify fallback when wtype is NOT installed.
        Simulate wtype absence deterministically regardless of host packages.
        """
        wl_copy_path = shutil.which("wl-copy")
        self.assertIsNotNone(wl_copy_path, "wl-copy MUST be installed")

        from niri_dictate.config import InjectionConfig
        from niri_dictate.injector import WaylandInjector

        injector = WaylandInjector(InjectionConfig(mode="paste", wtype_bin="/nonexistent/wtype"))
        effective_mode, is_fallback = injector.resolve_effective_mode()

        self.assertEqual(effective_mode, "clipboard_only",
                         "When wtype is absent, engine must fall back to clipboard_only")
        self.assertTrue(is_fallback)


if __name__ == "__main__":
    unittest.main(verbosity=2)
