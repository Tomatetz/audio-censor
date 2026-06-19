from __future__ import annotations

import os
import shutil
import sys
from pathlib import Path


APP_NAME = "Stream Censor"


def resource_root() -> Path:
    frozen_root = getattr(sys, "_MEIPASS", None)
    if frozen_root:
        return Path(frozen_root)
    return Path(__file__).resolve().parent.parent


def data_root() -> Path:
    override = os.environ.get("STREAM_CENSOR_DATA_DIR")
    if override:
        return Path(override).expanduser()
    if getattr(sys, "frozen", False):
        return Path.home() / "Library" / "Application Support" / APP_NAME
    return resource_root()


def ensure_user_data() -> Path:
    destination = data_root()
    destination.mkdir(parents=True, exist_ok=True)
    for name in ("config.jsonc", "words.txt", "test_script.txt"):
        target = destination / name
        source = resource_root() / name
        if not target.exists() and source.exists():
            shutil.copy2(source, target)
    (destination / "recordings").mkdir(exist_ok=True)
    return destination
