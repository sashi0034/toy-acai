"""Load the locally built ``toy_acai_core`` nanobind module."""

import sys
from pathlib import Path


def module_path() -> Path:
    path = Path(__file__).resolve().parents[1] / "linux-python" / "build"
    if not path.exists():
        raise FileNotFoundError(
            f"toy_acai_core was not built: {path}. "
            "Build it with: cmake --build linux-python/build"
        )
    return path


_core_path = module_path()
if str(_core_path) not in sys.path:
    sys.path.insert(0, str(_core_path))

import toy_acai_core as core
