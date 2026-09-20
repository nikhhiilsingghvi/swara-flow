from __future__ import annotations

import logging
import os
import sys

def configure_logging(level: str | None = None) -> None:
    log_level = (level or os.getenv("SF_LOG_LEVEL", "INFO")).upper()

    root = logging.getLogger("swaraflow")
    root.setLevel(log_level)

    if root.handlers:
        return

    handler = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter(
        fmt="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )
    handler.setFormatter(formatter)
    root.addHandler(handler)
