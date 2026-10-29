# Niri Voice Dictation

A fast, ephemeral, Wayland-native voice dictation engine built specifically for the **[Niri](https://github.com/YaLTeR/niri)** scrollable-tiling Wayland compositor and modern Wayland desktops (such as Dank Material Shell, Waybar, or desktop notification daemons).

Powered by Groq's ultra-low latency LPU infrastructure running **Whisper Large v3 Turbo**, this tool enables seamless speech-to-text with developer-centric vocabulary priming and zero focus stealing.

---

## Features

- ⚡ **Sub-Second Latency (~350ms - 650ms):** Direct in-memory Opus audio capture with instantaneous Groq LPU transcription.
- 🎯 **100% Keyboard Focus Preservation:** Captures, transcribes, and injects text without the active window (editor, browser, terminal) ever losing Wayland keyboard focus.
- 🔒 **Ephemeral RAM-Only Audio:** Audio files are recorded directly into RAM-backed temporary storage (`$XDG_RUNTIME_DIR` / `tmpfs`, mode `0700`) and unlinked immediately after memory buffering. No audio is ever written to persistent disk storage.
- 💻 **Smart Target Injection:** Automatically detects whether the focused window is a terminal or GUI app via Niri IPC and sends the appropriate paste sequence (`Ctrl+Shift+V` vs `Ctrl+V`), with fallback to direct virtual keyboard typing via `wtype`.
- 🛠️ **Developer Vocabulary Priming:** Autoregressively primed with bilingual programming terminology (Portuguese & English) within strict token budgets (≤ 200 tokens) to prevent phonetic drift on words like `commit`, `rebase`, `pull request`, `await`, `docker`.
- 🔔 **Visual OSD Feedback:** Non-stealing toast/notification indicators via D-Bus notifications (`notify-send`) and Dank Material Shell IPC.

---

## Architecture Overview

```
                      +-------------------+
                      |  Niri Compositor  |
                      |    (Mod+Alt+D)    |
                      +---------+---------+
                                | spawn
                                v
                      +-------------------+
                      |   niri-dictate    |
                      |   (State Lock)    |
                      +---+-----------+---+
                          |           |
            1. Record     |           | 2. Notify (OSD)
            (pw-record)   |           | (D-Bus / DMS)
                          v           v
           +----------------+       +-------------------+
           | In-Memory Opus |       | Non-Focus Toasts  |
           +-------+--------+       +-------------------+
                   |
                   | 3. HTTPS STT
                   v
           +------------------------+
           | Groq Cloud API (LPU)   |
           | whisper-large-v3-turbo |
           +-------+----------------+
                   |
                   | 4. Text Output
                   v
           +------------------------+
           | Wayland Injector       |
           | (wl-copy + wtype)      |
           +------------------------+
```

---

## Prerequisites

Ensure the following tools are installed on your Linux system:

- **Wayland Compositor:** [Niri](https://github.com/YaLTeR/niri)
- **Audio Stack:** PipeWire with WirePlumber (`pw-record`)
- **Clipboard & Keystroke Injection:**
  - `wl-clipboard` (`wl-copy`, `wl-paste`)
  - `wtype` (virtual keyboard typing and paste triggering)
- **Notifications (Optional):** `libnotify` (`notify-send`) or Dank Material Shell
- **Python:** 3.11+
- **Groq API Key:** Get a free key at [console.groq.com](https://console.groq.com)

On Ubuntu / Debian:
```bash
sudo apt install pipewire-bin wireplumber wl-clipboard wtype libnotify-bin
```

---

## Installation

1. **Clone the repository:**
   ```bash
   git clone https://github.com/eduardo-de-bastiani/niri-voice-dictation.git
   cd niri-voice-dictation
   ```

2. **Install package & dependencies:**
   ```bash
   pip install .
   # or with uv:
   # uv pip install -e .
   ```

3. **Install launcher command:**
   You can symlink the launcher script to `/usr/local/bin`:
   ```bash
   sudo ln -s "$(pwd)/scripts/niri-dictate" /usr/local/bin/niri-dictate
   ```

---

## Configuration

Copy the example configuration to your user config directory:

```bash
mkdir -p ~/.config/niri-dictate
cp config/config.toml.example ~/.config/niri-dictate/config.toml
chmod 600 ~/.config/niri-dictate/config.toml
```

Edit `~/.config/niri-dictate/config.toml` and configure your Groq API key:

```toml
[groq]
api_key = "gsk_your_api_key_here"
```

> **Tip:** You can also pass the key via environment variable:
> ```bash
> export GROQ_API_KEY="gsk_..."
> ```

---

## Niri Keybinding

Add the following binding to your `~/.config/niri/config.kdl` inside the `binds { ... }` block:

```kdl
binds {
    Mod+Alt+D repeat=false cooldown-ms=500 hotkey-overlay-title="Toggle Voice Dictation" {
        spawn "niri-dictate" "toggle";
    }
}
```

Reload Niri config (`niri msg action reload-config`).

---

## Usage

- Press <kbd>Mod</kbd> + <kbd>Alt</kbd> + <kbd>D</kbd> once to **start recording**. An indicator will notify that audio is being recorded.
- Speak your sentence or code.
- Press <kbd>Mod</kbd> + <kbd>Alt</kbd> + <kbd>D</kbd> again to **stop and transcribe**.
- The transcribed text is automatically pasted into the active application.

You can also run commands manually:
```bash
niri-dictate toggle       # Toggle recording / transcribing
niri-dictate status       # Show current engine status
niri-dictate abort        # Cancel active recording or transcription
```

---

## License

MIT
