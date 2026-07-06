from ..core import core
from ..simulation_context import SimulationContext


def random_inputs(ctx: SimulationContext, rng):
    return [
        core.FighterInput(
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.random() < 0.15,
        )
        for _ in range(core.FIGHTER_COUNT)
    ]


def copy_inputs(inputs: list[core.FighterInput]) -> list[core.FighterInput]:
    return [
        core.FighterInput(
            fighter_input.acceleration, fighter_input.turn, fighter_input.fire
        )
        for fighter_input in inputs
    ]
