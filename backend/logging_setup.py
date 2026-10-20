"""
Local diagnostic logging.

Logs go to ``logs/app.log`` (git-ignored, rotated at 1 MB, 3 files kept).
They contain timings, states and error tracebacks, and never transcripts,
audio or video.
"""

from __future__ import annotations

import logging
from logging.handlers import RotatingFileHandler
from pathlib import Path

_configured = False


def configure_logging(log_dir: Path, level: int = logging.INFO) -> Path:
    global _configured
    log_dir.mkdir(parents=True, exist_ok=True)
    path = log_dir / "app.log"
    if _configured:
        return path
    handler = RotatingFileHandler(path, maxBytes=1_000_000, backupCount=3, encoding="utf-8")
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
    root = logging.getLogger("backend")
    root.setLevel(level)
    root.addHandler(handler)
    _configured = True
    return path
