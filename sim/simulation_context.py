from datetime import datetime
import os
from pathlib import Path
import random

from ._slack import load_dotenv
from .core import core, module_path


def repository_root():
    return Path(__file__).resolve().parents[1]


def _output_directory():
    return repository_root() / "outputs"


class SimulationMetadata:
    def __init__(self):
        self.started_at = datetime.now().strftime("%Y%m%d_%H%M%S")

    def output_directory(self) -> Path:
        output_directory = _output_directory() / self.started_at
        return output_directory


class WorkerContext:
    """State owned by one rollout and never shared between processes."""

    def __init__(self, worker_id: int, seed: int):
        self.worker_id = worker_id
        self.battlefield = core.BattlefieldContext()
        self.rng = random.Random(seed)


def _parse_positive_int(value: str | None, name: str) -> int:
    if value is None:
        raise ValueError(f"{name} must be set in .env")
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a positive integer") from error
    if parsed_value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return parsed_value


class SimulationContext:
    def __init__(self):
        os.chdir(module_path())

        load_dotenv(repository_root() / ".env")

        self.rollout_worker_count = _parse_positive_int(
            os.environ.get("ROLLOUT_WORKER_COUNT"), "ROLLOUT_WORKER_COUNT"
        )

        self.data = SimulationMetadata()

        self.output_directory().mkdir(parents=True, exist_ok=True)

        self.renderer: core.BattlefieldRenderer = core.BattlefieldRenderer()

    def output_directory(self) -> Path:
        return self.data.output_directory()
