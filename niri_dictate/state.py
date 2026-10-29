"""
State management module for Niri Voice Dictation (niri-dictate).
Enforces the 5 process-level hardening invariants:
1. Process Identity Verification (/proc/<pid>/comm) & Stale PID Self-Healing.
2. Non-blocking Contention Debounce (fcntl.flock LOCK_EX | LOCK_NB) with exit code 0.
3. State Retention & Crash Recovery Through Transcription.
4. Clean Action Mapping and Graceful Degradation.
5. Non-truncating Lock Open (O_RDWR | O_CREAT, 0o600) & Atomic JSON Serialization (os.fsync/os.replace).
"""

from __future__ import annotations

import errno
import fcntl
import json
import logging
import math
import os
import signal
import sys
import time
from pathlib import Path
from typing import Any

logger = logging.getLogger("niri-dictate.state")


class HardenedStateManager:
    """Production state machine enforcing the 5 process-level hardening invariants."""

    def __init__(
        self,
        run_dir: Path | str | None = None,
        runtime_dir: Path | str | None = None,
        timeout_threshold: float = 30.0,
    ) -> None:
        target_dir = run_dir if run_dir is not None else runtime_dir
        if target_dir is None:
            raise ValueError("Either run_dir or runtime_dir must be provided.")
        self.run_dir = Path(target_dir)
        self.run_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
        self.lock_file = self.run_dir / "niri-dictate.lock"
        self.state_file = self.run_dir / "niri-dictate.state"
        self.timeout_threshold = float(timeout_threshold)

    def acquire_lock_nonblocking(self, exit_on_contention: bool = True) -> int | None:
        """
        Opens lock file with O_RDWR | O_CREAT (Invariant 5) and traps BlockingIOError (Invariant 2).
        Returns open file descriptor or None / exits with 0 on lock contention.
        """
        lock_fd = os.open(str(self.lock_file), os.O_RDWR | os.O_CREAT, 0o600)
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
            return lock_fd
        except (BlockingIOError, OSError) as e:
            os.close(lock_fd)
            if isinstance(e, BlockingIOError) or e.errno in (errno.EAGAIN, errno.EWOULDBLOCK):
                logger.info("Another niri-dictate instance holds the lock. Debouncing toggle.")
                if exit_on_contention:
                    sys.exit(0)  # Clean debounce exit for CLI execution
                return None
            raise

    def release_lock(self, lock_fd: int | None) -> None:
        """Releases flock and closes file descriptor safely."""
        if lock_fd is None:
            return
        try:
            fcntl.flock(lock_fd, fcntl.LOCK_UN)
        finally:
            os.close(lock_fd)

    def _write_state_atomic(self, payload: dict[str, Any]) -> None:
        """Writes JSON state atomically via tempfile, fsync, and os.replace (Invariant 5)."""
        tmp_path = self.state_file.with_suffix(f".tmp.{os.getpid()}")
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f)
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp_path, self.state_file)

    def _read_state(self, self_heal: bool = True) -> dict[str, Any] | None:
        """
        Reads and validates the JSON state file.
        Returns the state dictionary if valid, or None if missing or corrupted.
        If self_heal is True, automatically unlinks corrupted or non-dict state files.
        """
        if not self.state_file.exists():
            return None
        try:
            with open(self.state_file, "r", encoding="utf-8") as f:
                data = json.load(f)
            if not isinstance(data, dict):
                logger.warning(
                    "State file contains non-dict JSON (%s); purging state.",
                    type(data).__name__,
                )
                if self_heal:
                    self.state_file.unlink(missing_ok=True)
                return None
            return data
        except (json.JSONDecodeError, OSError) as exc:
            logger.warning("State file corrupted or unreadable (%s); purging state.", exc)
            if self_heal:
                self.state_file.unlink(missing_ok=True)
            return None

    def is_pid_alive(self, pid: int | Any) -> bool:
        """Verifies generic process liveness with null, type, and positive bounds safety."""
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            return True
        except (ProcessLookupError, PermissionError, OSError):
            return False

    def is_pid_valid_recorder(self, pid: int | Any) -> bool:
        """
        Verifies process liveness and command identity matches pw-record (Invariant 1).
        Enforces positive pid > 0, checks /proc/<pid>/comm, and returns False if comm cannot be read.
        """
        if not isinstance(pid, int) or isinstance(pid, bool) or pid <= 0:
            return False
        try:
            os.kill(pid, 0)
            comm_path = Path(f"/proc/{pid}/comm")
            if comm_path.exists():
                comm = comm_path.read_text().strip()
                return "pw-record" in comm
            return False
        except (ProcessLookupError, PermissionError, FileNotFoundError, OSError):
            return False

    def sweep_orphaned_audio(
        self,
        active_file: str | Path | None = None,
        active_path: str | Path | None = None,
    ) -> None:
        """Deletes orphaned .oga files left by aborted processes (Invariant 1)."""
        target = active_path if active_path is not None else active_file
        active_str = str(Path(target).resolve()) if target is not None else None
        for item in self.run_dir.glob("*.oga"):
            try:
                if active_str and str(item.resolve()) == active_str:
                    continue
                item.unlink(missing_ok=True)
                logger.warning("Swept orphaned ephemeral audio file: %s", item)
            except OSError:
                pass

    def handle_toggle(self, exit_on_contention: bool = True) -> dict[str, Any]:
        """
        Executes atomic state toggle transition.
        Returns action dictionary directing the CLI execution loop.
        """
        lock_fd = self.acquire_lock_nonblocking(exit_on_contention=exit_on_contention)
        if lock_fd is None:
            return {"action": "DEBOUNCED", "lock_fd": None}
        try:
            data = self._read_state(self_heal=True)
            if data is None:
                # State: IDLE or Corrupted -> Start RECORDING
                self.sweep_orphaned_audio()
                audio_path = str(self.run_dir / f"rec_{os.getpid()}_{int(time.time())}.oga")
                return {"action": "START_RECORDING", "audio_path": audio_path, "lock_fd": lock_fd}

            curr_state = data.get("state")
            rec_pid = data.get("pid")
            audio_file = data.get("audio_file")

            if curr_state == "RECORDING":
                # Check recorder liveness and audio file validity (Invariant 1)
                if not self.is_pid_valid_recorder(rec_pid) or not audio_file:
                    logger.warning(
                        "Stale or invalid recorder state detected (PID: %s, Audio: %s). Clearing stale state.",
                        rec_pid,
                        audio_file,
                    )
                    self.state_file.unlink(missing_ok=True)
                    self.sweep_orphaned_audio()
                    audio_path = str(self.run_dir / f"rec_{os.getpid()}_{int(time.time())}.oga")
                    return {"action": "START_RECORDING", "audio_path": audio_path, "lock_fd": lock_fd}

                # Transition RECORDING -> TRANSCRIBING (Invariant 3)
                data["state"] = "TRANSCRIBING"
                data["transcribing_pid"] = os.getpid()
                data["transcribing_since"] = time.time()
                self._write_state_atomic(data)

                return {
                    "action": "STOP_AND_TRANSCRIBE",
                    "rec_pid": rec_pid,
                    "audio_file": audio_file,
                    "notification_id": data.get("notification_id"),
                    "lock_fd": lock_fd,
                }

            elif curr_state == "TRANSCRIBING":
                transcribing_pid = data.get("transcribing_pid") or data.get("pid")
                try:
                    raw_since = data.get("transcribing_since")
                    transcribing_since = float(raw_since) if raw_since is not None else 0.0
                except (ValueError, TypeError):
                    transcribing_since = 0.0

                # Check process liveness and timeout
                is_alive = self.is_pid_alive(transcribing_pid)
                is_timed_out = (
                    not math.isfinite(transcribing_since)
                    or (transcribing_since <= 0.0)
                    or ((time.time() - transcribing_since) > self.timeout_threshold)
                )

                if not is_alive or is_timed_out:
                    logger.warning("Recovered from dead/timed-out transcription.")
                    self.state_file.unlink(missing_ok=True)
                    self.sweep_orphaned_audio()
                    audio_path = str(self.run_dir / f"rec_{os.getpid()}_{int(time.time())}.oga")
                    return {"action": "START_RECORDING", "audio_path": audio_path, "lock_fd": lock_fd}

                # Concurrent toggle while STT is legitimately in-flight (Invariant 3)
                logger.info("Dictation toggle ignored: transcription currently in progress.")
                return {"action": "REJECT_CONCURRENT", "lock_fd": lock_fd}

            else:
                # Unknown state recovery
                self.state_file.unlink(missing_ok=True)
                self.sweep_orphaned_audio()
                return {"action": "RECOVER_IDLE", "lock_fd": lock_fd}

        except Exception:
            self.release_lock(lock_fd)
            raise

    def handle_abort(self) -> dict[str, Any]:
        """Terminates active recorder or transcription and cleans up all state and audio."""
        lock_fd = self.acquire_lock_nonblocking(exit_on_contention=False)
        if lock_fd is None:
            return {"action": "BUSY", "lock_fd": None}
        try:
            prev_state = "IDLE"
            if self.state_file.exists():
                data = self._read_state(self_heal=False)
                if data is not None:
                    prev_state = data.get("state", "UNKNOWN")
                    rec_pid = data.get("pid")
                    if self.is_pid_alive(rec_pid):
                        try:
                            os.kill(rec_pid, signal.SIGKILL)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                    trans_pid = data.get("transcribing_pid")
                    if self.is_pid_alive(trans_pid):
                        try:
                            os.kill(trans_pid, signal.SIGTERM)
                        except (ProcessLookupError, PermissionError, OSError):
                            pass
                self.state_file.unlink(missing_ok=True)

            self.sweep_orphaned_audio()
            return {"action": "ABORTED", "previous_state": prev_state, "lock_fd": lock_fd}
        finally:
            self.release_lock(lock_fd)

    def get_status(self) -> dict[str, Any]:
        """Inspects current state without mutating."""
        if not self.state_file.exists():
            return {"state": "IDLE"}
        data = self._read_state(self_heal=False)
        if data is None:
            return {"state": "CORRUPTED"}

        curr_state = data.get("state")
        if curr_state == "RECORDING":
            pid = data.get("pid")
            if self.is_pid_valid_recorder(pid):
                return {
                    "state": "RECORDING",
                    "pid": pid,
                    "audio_file": data.get("audio_file"),
                    "notification_id": data.get("notification_id"),
                    "started_at": data.get("started_at") or data.get("timestamp"),
                }
            return {"state": "STALE_RECORDING", "pid": pid}
        elif curr_state == "TRANSCRIBING":
            t_pid = data.get("transcribing_pid") or data.get("pid")
            try:
                raw_since = data.get("transcribing_since")
                since = float(raw_since) if raw_since is not None else 0.0
            except (ValueError, TypeError):
                since = 0.0
            if self.is_pid_alive(t_pid) and since > 0.0 and (time.time() - since <= self.timeout_threshold):
                return {
                    "state": "TRANSCRIBING",
                    "transcribing_pid": t_pid,
                    "transcribing_since": since,
                    "notification_id": data.get("notification_id"),
                }
            return {"state": "STALE_TRANSCRIBING", "transcribing_pid": t_pid}
        return {"state": curr_state if curr_state is not None else "UNKNOWN"}

    def transition_to_recording(
        self,
        lock_fd: int | None,
        pid: int,
        audio_path: str | Path,
        notification_id: int | None = None,
    ) -> None:
        """Atomically records RECORDING state metadata."""
        now = time.time()
        data = {
            "state": "RECORDING",
            "pid": pid,
            "audio_file": str(audio_path),
            "notification_id": notification_id,
            "started_at": now,
            "timestamp": now,
        }
        self._write_state_atomic(data)

    def transition_to_transcribing(
        self,
        lock_fd: int,
        rec_pid: int,
        notification_id: int | None = None,
    ) -> None:
        """Atomically updates state to TRANSCRIBING, retaining the state file (Invariant 3)."""
        data = {
            "state": "TRANSCRIBING",
            "pid": rec_pid,
            "transcribing_pid": os.getpid(),
            "transcribing_since": time.time(),
            "notification_id": notification_id,
        }
        self._write_state_atomic(data)

    def clear_state(self, lock_fd: int | None = None) -> None:
        """Unlinks the state file returning the engine to IDLE."""
        self.state_file.unlink(missing_ok=True)
