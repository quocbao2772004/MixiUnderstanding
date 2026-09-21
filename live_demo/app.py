#!/usr/bin/env python3
"""Run the canonical QCES AudioQA chat application."""

from pathlib import Path
import runpy


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
CANONICAL_APP = (
    REPOSITORY_ROOT
    / "code"
    / "mixi_understanding"
    / "apps"
    / "qces_audio_chat_demo.py"
)

if not CANONICAL_APP.is_file():
    raise FileNotFoundError(f"Live demo source is missing: {CANONICAL_APP}")

runpy.run_path(str(CANONICAL_APP), run_name="__main__")
