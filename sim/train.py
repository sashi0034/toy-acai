#!/usr/bin/env python3
import sys
from pathlib import Path

import numpy as np


RENDER_INTERVAL = 0.1


def repository_root():
    return Path(__file__).resolve().parents[1]


def add_default_module_paths():
    path = repository_root() / "linux-python" / "build"
    if path.exists():
        sys.path.insert(0, str(path))


def main():
    add_default_module_paths()

    import toy_acai_core

    print("Running toy-acai simulation from Python.")


if __name__ == "__main__":
    main()
