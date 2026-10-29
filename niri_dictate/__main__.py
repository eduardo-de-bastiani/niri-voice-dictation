"""
Package execution entry point: python3 -m niri_dictate "$@"
"""

from __future__ import annotations

import sys
from niri_dictate.cli import main

if __name__ == "__main__":
    sys.exit(main())
