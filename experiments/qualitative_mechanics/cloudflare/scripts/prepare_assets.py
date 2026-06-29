#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Prepare static Cloudflare assets from existing local motion JSON files."""

from __future__ import annotations

import shutil
from pathlib import Path


CLOUDFLARE_DIR = Path(__file__).resolve().parents[1]
EXPERIMENT_DIR = CLOUDFLARE_DIR.parent
SOURCE_MOTION_DIR = EXPERIMENT_DIR / "web_motion"
TARGET_MOTION_DIR = CLOUDFLARE_DIR / "public" / "web_motion"


def main() -> None:
    if not SOURCE_MOTION_DIR.exists():
        raise FileNotFoundError(f"Missing source motion directory: {SOURCE_MOTION_DIR}")
    files = sorted(SOURCE_MOTION_DIR.glob("*.json"))
    if not files:
        raise RuntimeError(f"No motion JSON files found in {SOURCE_MOTION_DIR}")
    if TARGET_MOTION_DIR.exists():
        shutil.rmtree(TARGET_MOTION_DIR)
    TARGET_MOTION_DIR.mkdir(parents=True)
    total_bytes = 0
    for source in files:
        target = TARGET_MOTION_DIR / source.name
        shutil.copy2(source, target)
        total_bytes += target.stat().st_size
    print(f"motion_files={len(files)}")
    print(f"motion_bytes={total_bytes}")
    print(f"wrote={TARGET_MOTION_DIR}")


if __name__ == "__main__":
    main()
