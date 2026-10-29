"""
Niri Voice Dictation (niri-dictate).
Wayland-native voice dictation engine for Niri and Dank Material Shell.
"""

from __future__ import annotations

from .audio import AudioCaptureManager
from .config import AppConfig, load_config
from .feedback import VisualFeedbackManager
from .injector import WaylandInjector
from .prompt import PrimingPromptBuilder
from .state import HardenedStateManager
from .stt import GroqSTTClient

__version__ = "0.1.0"
__all__ = [
    "AudioCaptureManager",
    "AppConfig",
    "GroqSTTClient",
    "HardenedStateManager",
    "PrimingPromptBuilder",
    "VisualFeedbackManager",
    "WaylandInjector",
    "load_config",
]
