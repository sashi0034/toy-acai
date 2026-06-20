import os
from pathlib import Path

from .core import core, module_path


def repository_root():
    return Path(__file__).resolve().parents[1]


def output_path():
    return repository_root() / "outputs"


class SimulationContext:
    def __init__(self):
        os.chdir(module_path())

        self.battlefield: core.BattlefieldContext = core.BattlefieldContext()
        self.renderer: core.BattlefieldRenderer = core.BattlefieldRenderer()
