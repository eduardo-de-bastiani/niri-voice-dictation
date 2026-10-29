"""
Audio capture module for Niri Voice Dictation (niri-dictate).
Manages PipeWire native capture via pw-record, tmpfs RAM storage,
clean SIGINT termination with Ogg Opus trailer flush, and instant inode unlinking.
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import time
import uuid
from pathlib import Path
from typing import TYPE_CHECKING, Tuple

if TYPE_CHECKING:
    from .config import AudioConfig

logger = logging.getLogger("niri-dictate.audio")


class AudioCaptureError(Exception):
    """Raised when audio capture fails, hangs, or encounters missing files."""
    pass


class EmptyAudioError(AudioCaptureError):
    """Raised when recorded audio buffer is empty (0 bytes)."""
    pass


class AudioCaptureManager:
    """
    Manages native PipeWire ephemeral audio recording.
    Guarantees:
    - 16 kHz mono Opus encoding inside an Ogg container (.oga).
    - Strict tmpfs storage in $XDG_RUNTIME_DIR/niri-dictate/ (mode 0700/0600).
    - Child process detachment via start_new_session=True.
    - Clean SIGINT trailer flush.
    - Zero permanent disk persistence via immediate inode unlinking upon read.
    - Automatic orphan sweeping.
    """

    def __init__(self, config: AudioConfig, runtime_dir: Path | str) -> None:
        self.config = config
        self.runtime_dir = Path(runtime_dir)
        self.runtime_dir.mkdir(mode=0o700, parents=True, exist_ok=True)

    def start_recording(
        self,
        target_path: Path | str | None = None,
        target_audio_path: Path | str | None = None,
    ) -> Tuple[int, Path]:
        """
        Spawns detached pw-record process writing 16kHz mono Opus in Ogg container.
        Returns: (recorder_pid, audio_file_path)
        """
        chosen_path = target_path if target_path is not None else target_audio_path
        self.sweep_orphaned_audio(active_path=chosen_path)

        if chosen_path:
            audio_path = Path(chosen_path)
        else:
            filename = f"rec_{os.getpid()}_{int(time.time())}_{uuid.uuid4().hex[:6]}.oga"
            audio_path = self.runtime_dir / filename

        # Pre-create file with mode 0600 permissions
        fd = os.open(str(audio_path), os.O_CREAT | os.O_WRONLY, 0o600)
        os.close(fd)

        cmd = [
            self.config.recorder_bin,
            "--rate", str(self.config.sample_rate),
            "--channels", str(self.config.channels),
            "--format", self.config.format,
        ]
        if self.config.source and self.config.source != "@DEFAULT_AUDIO_SOURCE@":
            cmd.extend(["--target", self.config.source])
        cmd.append(str(audio_path))

        logger.info("Spawning pw-record capture: %s", " ".join(cmd))

        # Detach child session so it outlives parent CLI process; umask guarantees 0600
        old_umask = os.umask(0o077)
        try:
            proc = subprocess.Popen(
                cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        finally:
            os.umask(old_umask)

        return proc.pid, audio_path

    def stop_recording_and_read(self, pid: int, audio_path: Path | str) -> bytes:
        """
        Sends SIGINT to pw-record, waits for clean Ogg container trailer flush,
        reads bytes into memory, and IMMEDIATELY unlinks the file from tmpfs.
        """
        audio_path = Path(audio_path)
        if not pid or not isinstance(pid, int):
            raise AudioCaptureError(f"Invalid recorder PID: {pid}")

        # 1. Send SIGINT to allow pw-record/libsndfile to flush Ogg container
        try:
            os.kill(pid, signal.SIGINT)
        except ProcessLookupError:
            logger.warning("Recorder PID %d was already terminated.", pid)

        # 2. Wait up to 2.0s for pw-record to exit and write trailers
        deadline = time.time() + 2.0
        while time.time() < deadline:
            try:
                os.kill(pid, 0)
                status_file = Path(f"/proc/{pid}/status")
                if status_file.exists() and "State:\tZ" in status_file.read_text(encoding="utf-8", errors="ignore"):
                    break
                time.sleep(0.02)
            except (ProcessLookupError, PermissionError, OSError):
                break
        else:
            logger.warning("Recorder process %d did not exit after SIGINT; sending SIGKILL.", pid)
            try:
                os.kill(pid, signal.SIGKILL)
            except ProcessLookupError:
                pass

        # 3. Read compressed audio into RAM heap buffer
        if not audio_path.exists():
            raise AudioCaptureError(f"Audio file disappeared or failed to record: {audio_path}")

        try:
            audio_bytes = audio_path.read_bytes()
        finally:
            # 4. Immediate unlink guarantee: destroyed from tmpfs RAM before STT dispatch
            try:
                audio_path.unlink(missing_ok=True)
                logger.info("Ephemeral audio inode destroyed: %s", audio_path)
            except OSError as e:
                logger.warning("Could not unlink audio file %s: %s", audio_path, e)

        if not audio_bytes:
            raise EmptyAudioError("Captured audio buffer contains 0 bytes.")

        return audio_bytes

    def sweep_orphaned_audio(
        self,
        active_path: Path | str | None = None,
        active_file: Path | str | None = None,
    ) -> None:
        """Purges any lingering .oga files left by abrupt crashes or aborted runs."""
        target = active_path if active_path is not None else active_file
        active_str = str(target) if target else None
        for item in self.runtime_dir.glob("*.oga"):
            if active_str and str(item) == active_str:
                continue
            try:
                item.unlink(missing_ok=True)
                logger.warning("Swept orphaned ephemeral audio file: %s", item)
            except OSError:
                pass
