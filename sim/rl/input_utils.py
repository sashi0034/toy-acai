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
