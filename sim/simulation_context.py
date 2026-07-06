from dataclasses import dataclass
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


@dataclass
class WorkerContextState:
    worker_id: int
    battlefield: core.BattlefieldContext
    rng_state: object


class WorkerContext:
    """State owned by one rollout and never shared between processes."""

    def __init__(self, worker_id: int, seed: int):
        self.worker_id = worker_id
        self.battlefield = core.BattlefieldContext()
        self.rng = random.Random(seed)

    def save_state(self) -> WorkerContextState:
        return WorkerContextState(
            worker_id=self.worker_id,
            battlefield=core.BattlefieldContext(self.battlefield),
            rng_state=self.rng.getstate(),
        )

    def restore_state(self, state: WorkerContextState):
        self.battlefield = core.BattlefieldContext(state.battlefield)
        self.rng.setstate(state.rng_state)

    @classmethod
    def from_state(cls, state: WorkerContextState) -> "WorkerContext":
        ctx = cls(state.worker_id, seed=0)
        ctx.restore_state(state)
        return ctx


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


def parse_optional_non_negative_int(value: str | None, name: str) -> int:
    if value is None or value.strip() == "":
        return 0
    try:
        parsed_value = int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be a non-negative integer") from error
    if parsed_value < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return parsed_value


def resolve_optional_repository_path(value: str | None) -> Path | None:
    if value is None or value.strip() == "":
        return None

    path = Path(value).expanduser()
    if path.is_absolute():
        return path
    return repository_root() / path


class SimulationContext:
    def __init__(self):
        os.chdir(module_path())

        load_dotenv(repository_root() / ".env")

        self.rollout_worker_count = _parse_positive_int(
            os.environ.get("ROLLOUT_WORKER_COUNT"), "ROLLOUT_WORKER_COUNT"
        )
        self.checkpoint_save_interval_updates = parse_optional_non_negative_int(
            os.environ.get("CHECKPOINT_SAVE_INTERVAL_UPDATES"),
            "CHECKPOINT_SAVE_INTERVAL_UPDATES",
        )
        self.checkpoint_resume_path = resolve_optional_repository_path(
            os.environ.get("CHECKPOINT_RESUME_PATH")
        )

        self.data = SimulationMetadata()

        self.output_directory().mkdir(parents=True, exist_ok=True)

        self.renderer: core.BattlefieldRenderer = core.BattlefieldRenderer()

    def output_directory(self) -> Path:
        return self.data.output_directory()
