import os
from pathlib import Path
import sys


def repository_root():
    return Path(__file__).resolve().parents[1]


def module_path():
    path = repository_root() / "linux-python" / "build"
    if not path.exists():
        raise FileNotFoundError(
            f"toy_acai_core was not built: {path}. "
            "Build it with: cmake --build linux-python/build"
        )
    return path


def output_path():
    return repository_root() / "outputs"


class SimulationContext:
    def __init__(self):
        sys.path.insert(0, str(module_path()))
        os.chdir(module_path())

        import toy_acai_core

        self.m = toy_acai_core

        self.battlefiled = self.m.BattlefieldContext()

        self.renderer = self.m.BattlefieldRenderer()
