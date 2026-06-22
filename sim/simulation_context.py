from datetime import datetime
import os
from pathlib import Path

from .core import core, module_path


def repository_root():
    return Path(__file__).resolve().parents[1]


def _output_directory():
    return repository_root() / "outputs"


class SimulationContext:
    def __init__(self):
        os.chdir(module_path())

        self.started_at = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_directory().mkdir(parents=True, exist_ok=True)

        self.battlefield: core.BattlefieldContext = core.BattlefieldContext()
        self.renderer: core.BattlefieldRenderer = core.BattlefieldRenderer()

    def output_directory(self) -> Path:
        output_directory = _output_directory() / self.started_at
        return output_directory
