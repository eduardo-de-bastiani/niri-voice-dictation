"""
Real-time PipeWire Microphone Amplitude Meter for Niri Voice Dictation Pill.
Streams 11 normalized bar heights (0..100) as semicolon-separated ASCII lines at ~30 FPS
by reading raw PCM frames directly from PipeWire's default audio source via pw-cat.
"""

from __future__ import annotations

import math
import signal
import struct
import subprocess
import sys


def main() -> int:
    source = sys.argv[1] if len(sys.argv) > 1 else ""
    bars_count = 11
    sample_rate = 8000
    chunk_samples = 256  # ~32ms per frame (~31 FPS)
    chunk_bytes = chunk_samples * 2  # 16-bit signed mono PCM

    cmd = [
        "/usr/bin/pw-cat",
        "--record",
        "--rate", str(sample_rate),
        "--channels", "1",
        "--format", "s16",
    ]
    if source and source != "@DEFAULT_AUDIO_SOURCE@":
        cmd.extend(["--target", source])
    cmd.append("-")

    proc = subprocess.Popen(
        cmd,
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        stderr=subprocess.DEVNULL,
    )

    def _shutdown(*_: object) -> None:
        try:
            proc.terminate()
        except OSError:
            pass
        sys.exit(0)

    signal.signal(signal.SIGTERM, _shutdown)
    signal.signal(signal.SIGINT, _shutdown)

    # Symmetric center-weighted envelope weights (matches Wispr Flow capsule waveform)
    envelope = [0.34, 0.50, 0.68, 0.84, 0.95, 1.00, 0.95, 0.84, 0.68, 0.50, 0.34]
    smoothed = [8.0] * bars_count

    try:
        assert proc.stdout is not None
        sub_size = chunk_samples // bars_count
        while True:
            raw = proc.stdout.read(chunk_bytes)
            if not raw or len(raw) < chunk_bytes:
                break

            samples = struct.unpack(f"<{chunk_samples}h", raw)

            # Overall RMS boost + localized sub-window energy for lively per-bar movement
            total_sq = sum(s * s for s in samples)
            global_rms = math.sqrt(total_sq / chunk_samples) / 32768.0
            # Perceptual non-linear speech curve (quiet voice still visibly modulates bars)
            global_gain = min(1.0, pow(max(0.0, global_rms * 7.5), 0.58))

            out_vals: list[str] = []
            for i in range(bars_count):
                start = i * sub_size
                sub = samples[start : start + sub_size]
                sub_rms = math.sqrt(sum(s * s for s in sub) / len(sub)) / 32768.0
                local_energy = min(1.0, pow(max(0.0, sub_rms * 8.5), 0.60))

                # Blend global envelope with local speech transients
                target = (0.55 * global_gain + 0.45 * local_energy) * envelope[i] * 100.0
                target = max(8.0, min(100.0, target))

                # Fast attack, smooth decay
                alpha = 0.62 if target > smoothed[i] else 0.32
                smoothed[i] = smoothed[i] * (1.0 - alpha) + target * alpha
                out_vals.append(str(int(round(smoothed[i]))))

            sys.stdout.write(";".join(out_vals) + "\n")
            sys.stdout.flush()
    except (BrokenPipeError, KeyboardInterrupt):
        pass
    finally:
        try:
            proc.terminate()
            proc.wait(timeout=0.5)
        except Exception:
            pass

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

