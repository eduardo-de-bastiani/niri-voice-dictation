"""
CLI Subcommand Engine for Niri Voice Dictation.
Orchestrates toggle, abort, status, and config-check subcommands.
Wires configuration, hardened state machine, ephemeral audio, Groq STT,
priming prompt builder, Wayland injection, and visual feedback.
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
import signal
import sys
import time
from pathlib import Path
from typing import Sequence

from .audio import AudioCaptureManager
from .config import AppConfig, load_config
from .feedback import VisualFeedbackManager
from .injector import WaylandInjector
from .postprocess import GroqPostProcessor
from .prompt import PrimingPromptBuilder
from .state import HardenedStateManager
from .stt import GroqSTTClient

logger = logging.getLogger("niri-dictate.cli")


def setup_logging(level_name: str) -> None:
    """Configures structured console logging."""
    level = getattr(logging, level_name.upper(), logging.INFO)
    logging.basicConfig(
        level=level,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )


def handle_toggle(args: argparse.Namespace, config: AppConfig) -> int:
    """
    Executes the toggle lifecycle (Mod+Alt+D):
    - IDLE -> RECORDING
    - RECORDING -> TRANSCRIBING -> INJECT -> IDLE
    - Concurrency debounced via Invariant 2; in-flight STT rejected with toast via Invariant 3.
    """
    timeout_threshold = max(
        45.0,
        config.groq.read_timeout_seconds + config.postprocess.read_timeout_seconds + 10.0,
    )
    state_mgr = HardenedStateManager(
        run_dir=config.general.runtime_dir,
        timeout_threshold=timeout_threshold,
    )
    feedback_mgr = VisualFeedbackManager(
        config.feedback,
        runtime_dir=config.general.runtime_dir,
        audio_source=config.audio.source,
    )

    # 1. Atomic state toggle attempt
    action_dict = state_mgr.handle_toggle(exit_on_contention=True)
    action = action_dict.get("action")
    lock_fd = action_dict.get("lock_fd")

    if action == "DEBOUNCED":
        return 0

    if action == "REJECT_CONCURRENT":
        if lock_fd is not None:
            state_mgr.release_lock(lock_fd)
        feedback_mgr.show_toast(config.feedback.toast_transcribing, level="warn")
        return 0

    if action == "START_RECORDING":
        audio_path = Path(action_dict["audio_path"])
        audio_mgr = AudioCaptureManager(config.audio, config.general.runtime_dir)
        try:
            rec_pid, actual_audio_path = audio_mgr.start_recording(target_path=audio_path)
            notif_id = feedback_mgr.show_recording()
            now = time.time()
            state_mgr._write_state_atomic({
                "state": "RECORDING",
                "pid": rec_pid,
                "audio_file": str(actual_audio_path),
                "notification_id": notif_id,
                "timestamp": now,
                "started_at": now,
            })
            return 0
        finally:
            if lock_fd is not None:
                state_mgr.release_lock(lock_fd)

    if action == "STOP_AND_TRANSCRIBE":
        rec_pid = action_dict.get("rec_pid")
        raw_audio_file = action_dict.get("audio_file")
        notif_id = action_dict.get("notification_id")

        # Invariant 3: Release lock immediately so subsequent toggles can inspect state
        if lock_fd is not None:
            state_mgr.release_lock(lock_fd)

        if not raw_audio_file or not rec_pid:
            logger.error("Missing audio file or recorder PID in STOP_AND_TRANSCRIBE.")
            state_mgr.clear_state()
            feedback_mgr.show_error(notif_id=notif_id, message="Corrupted recording state: missing audio file.")
            return 1

        audio_file = Path(raw_audio_file)

        # Transition visual indicator to State B
        feedback_mgr.show_transcribing(notif_id=notif_id)

        # 2. Stop capture and read audio bytes into heap; audio inode unlinked immediately
        audio_mgr = AudioCaptureManager(config.audio, config.general.runtime_dir)
        try:
            audio_bytes = audio_mgr.stop_recording_and_read(rec_pid, audio_file)
        except Exception as exc:
            logger.error("Audio capture termination failed: %s", exc)
            state_mgr.clear_state()
            feedback_mgr.show_error(notif_id=notif_id, message=f"Audio capture error: {exc}")
            return 1

        if not audio_bytes or len(audio_bytes) < 100:
            logger.warning("Empty or truncated audio captured.")
            state_mgr.clear_state()
            feedback_mgr.show_toast("No audio recorded", level="warn")
            feedback_mgr.show_completed(notif_id=notif_id, typed=True)
            return 0

        # 3. Resolve API credentials
        api_key = args.key or os.environ.get(config.groq.api_key_env_var) or config.groq.api_key
        if not api_key:
            err_msg = f"Missing Groq API Key (${config.groq.api_key_env_var})"
            logger.error(err_msg)
            state_mgr.clear_state()
            feedback_mgr.show_error(notif_id=notif_id, message=err_msg)
            return 1

        # 4. Construct priming prompt within token budget
        prompt_builder = PrimingPromptBuilder.from_config(config.groq)
        priming_prompt = prompt_builder.build_prompt(max_tokens=config.groq.max_prompt_tokens)

        # 5. Groq Cloud STT transcription
        stt_client = GroqSTTClient(config.groq, api_key=api_key)
        try:
            transcription = stt_client.transcribe(audio_bytes, prompt=priming_prompt)
        except Exception as exc:
            logger.error("Transcription failed: %s", exc)
            state_mgr.clear_state()
            feedback_mgr.show_error(notif_id=notif_id, message=str(exc))
            return 1

        cleaned_text = transcription.strip()
        lower_text = cleaned_text.lower()
        is_silence_hallucination = (
            not cleaned_text
            or lower_text.startswith("termos adicionais")
            or lower_text.startswith("código, commits e comandos")
            or lower_text in ("obrigado.", "obrigado por assistir.", "legendas pela comunidade amara.org")
        )
        if is_silence_hallucination:
            logger.info("Transcription yielded empty text or silence prompt echo.")
            state_mgr.clear_state()
            feedback_mgr.show_toast("No speech detected", level="info")
            feedback_mgr.show_completed(notif_id=notif_id, typed=True)
            return 0

        # 5b. Optional LLM Transcript Post-Processing (with resilient fallback to raw STT)
        if config.postprocess.enabled:
            feedback_mgr.show_refining(notif_id=notif_id)
            try:
                postprocessor = GroqPostProcessor(config.postprocess, api_key=api_key)
                refined_text = postprocessor.refine(cleaned_text)
                if refined_text:
                    cleaned_text = refined_text
            except Exception as exc:
                logger.warning("LLM post-processing failed (falling back to raw STT): %s", exc)

        # 6. Wayland text delivery with exception safety and clean state recovery
        try:
            injector = WaylandInjector(config.injection)
            inject_res = injector.inject(cleaned_text)
            typed = inject_res.get("typed", False)
            advisory = inject_res.get("advisory")

            # 7. Complete feedback and reset state
            feedback_mgr.show_completed(notif_id=notif_id, typed=typed, advisory=advisory)
            state_mgr.clear_state()
            return 0
        except Exception as exc:
            logger.error("Text delivery/injection failed: %s", exc)
            state_mgr.clear_state()
            feedback_mgr.show_error(notif_id=notif_id, message=f"Injection error: {exc}")
            return 1

    if action == "RECOVER_IDLE":
        if lock_fd is not None:
            state_mgr.release_lock(lock_fd)
        return 0

    return 0


def handle_abort(args: argparse.Namespace, config: AppConfig) -> int:
    """
    Executes the abort command:
    Terminates any active recorder process or in-flight transcription,
    purges ephemeral audio, resets state file to IDLE, and notifies user.
    """
    state_mgr = HardenedStateManager(run_dir=config.general.runtime_dir)
    state_mgr.handle_abort()
    feedback_mgr = VisualFeedbackManager(
        config.feedback,
        runtime_dir=config.general.runtime_dir,
    )
    feedback_mgr.dismiss_pill()
    if config.feedback.provider != "pill":
        feedback_mgr.show_toast("Dictation aborted", level="info")
    print("Dictation session aborted and cleaned up.")
    return 0


def handle_status(args: argparse.Namespace, config: AppConfig) -> int:
    """Reports current operational state, PID liveness, and duration."""
    state_mgr = HardenedStateManager(run_dir=config.general.runtime_dir)

    status_data = {
        "state": "IDLE",
        "pid": None,
        "alive": False,
        "duration_seconds": 0.0,
        "runtime_dir": str(config.general.runtime_dir),
    }

    if state_mgr.state_file.exists():
        try:
            data = json.loads(state_mgr.state_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                state = data.get("state", "IDLE")
                pid = data.get("pid")
                started_at = data.get("started_at") or data.get("timestamp") or time.time()
                alive = state_mgr.is_pid_alive(pid) if isinstance(pid, int) else False

                status_data["state"] = state
                status_data["pid"] = pid
                status_data["alive"] = alive
                try:
                    status_data["duration_seconds"] = max(0.0, round(time.time() - float(started_at), 2))
                except (ValueError, TypeError):
                    status_data["duration_seconds"] = 0.0
            else:
                status_data["state"] = "CORRUPTED"
        except Exception:
            status_data["state"] = "CORRUPTED"

    if getattr(args, "json", False):
        print(json.dumps(status_data, indent=2))
    else:
        state_str = status_data["state"]
        pid_str = (
            f" (PID: {status_data['pid']}, Alive: {status_data['alive']})"
            if status_data["pid"]
            else ""
        )
        dur_str = (
            f", Duration: {status_data['duration_seconds']}s"
            if status_data["duration_seconds"] > 0
            else ""
        )
        print(f"Status: {state_str}{pid_str}{dur_str}")

    return 0


def handle_config_check(args: argparse.Namespace, config: AppConfig) -> int:
    """Validates configuration syntax and probes required system binaries."""
    print("=== Niri Voice Dictation: Configuration & Health Check ===")
    print(f"Config File: {args.config or 'Default hierarchy'}")
    print(f"Runtime Dir: {config.general.runtime_dir} (Exists: {config.general.runtime_dir.exists()})")
    print(f"Feedback UI: {config.feedback.provider}")
    pp_status = (
        f"Enabled ({config.postprocess.model})"
        if config.postprocess.enabled
        else "Disabled"
    )
    print(f"LLM Cleanup: {pp_status}")

    has_error = False

    # Check binaries
    checks = [
        ("PipeWire Recorder (pw-record)", config.audio.recorder_bin, True),
        ("PipeWire Audio Cat (pw-cat)", "/usr/bin/pw-cat", False),
        ("Quickshell Overlay (qs)", config.feedback.quickshell_bin, False),
        ("Wayland Clipboard (wl-copy)", config.injection.wl_copy_bin, True),
        ("Wayland Paste (wl-paste)", config.injection.wl_paste_bin, False),
        ("Keystroke Simulator (wtype)", config.injection.wtype_bin, False),
        ("Niri Compositor (niri)", "niri", False),
        ("Notification (notify-send)", "notify-send", False),
        ("Dank Material Shell (dms)", "dms", False),
    ]

    print("\nDependency Audit:")
    for label, binary, required in checks:
        resolved = shutil.which(binary)
        if resolved:
            print(f"  [OK]       {label:<30} -> {resolved}")
        elif required:
            print(f"  [CRITICAL] {label:<30} -> NOT FOUND")
            has_error = True
        else:
            print(f"  [OPTIONAL] {label:<30} -> NOT FOUND (Graceful degradation enabled)")

    # Check API key
    api_key = args.key or os.environ.get(config.groq.api_key_env_var) or config.groq.api_key
    print("\nAuthentication Audit:")
    if api_key:
        masked = api_key[:6] + "..." + api_key[-4:] if len(api_key) > 10 else "***"
        print(f"  [OK]       Groq API Key configured ({masked})")
    else:
        print(f"  [WARN]     Groq API Key not found (${config.groq.api_key_env_var} unset)")

    print("==========================================================")
    return 1 if has_error else 0


def build_parser() -> argparse.ArgumentParser:
    """Builds CLI argument parser."""
    parser = argparse.ArgumentParser(
        prog="niri-dictate",
        description="Wayland-native voice dictation engine for Niri and Dank Material Shell.",
    )
    parser.add_argument(
        "-c", "--config",
        type=Path,
        default=None,
        help="Path to custom configuration TOML file.",
    )
    parser.add_argument(
        "-v", "--verbose",
        action="store_true",
        help="Enable verbose DEBUG logging.",
    )
    parser.add_argument(
        "--key",
        type=str,
        default=None,
        help="Groq API key override.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Groq STT model override.",
    )

    subparsers = parser.add_subparsers(dest="subcommand", required=True)

    # Subcommand: toggle
    subparsers.add_parser(
        "toggle",
        help="Toggle recording on/off (primary Niri shortcut target).",
    )

    # Subcommand: abort
    subparsers.add_parser(
        "abort",
        help="Immediately abort recording/transcription and wipe ephemeral files.",
    )

    # Subcommand: status
    status_parser = subparsers.add_parser(
        "status",
        help="Display current dictation operational status.",
    )
    status_parser.add_argument(
        "--json",
        action="store_true",
        help="Output status information as JSON.",
    )

    # Subcommand: config-check
    subparsers.add_parser(
        "config-check",
        help="Validate configuration and verify system dependencies.",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI execution entry point."""
    parser = build_parser()
    args = parser.parse_args(argv)

    try:
        config = load_config(args.config)
    except Exception as exc:
        print(f"Configuration load error: {exc}", file=sys.stderr)
        return 1

    if args.verbose:
        setup_logging("DEBUG")
    else:
        setup_logging(config.general.log_level)

    if args.model:
        # Override model dynamically via contract
        object.__setattr__(config.groq, "model", args.model)

    if args.subcommand == "toggle":
        return handle_toggle(args, config)
    elif args.subcommand == "abort":
        return handle_abort(args, config)
    elif args.subcommand == "status":
        return handle_status(args, config)
    elif args.subcommand == "config-check":
        return handle_config_check(args, config)
    else:
        parser.print_help()
        return 1


if __name__ == "__main__":
    sys.exit(main())
