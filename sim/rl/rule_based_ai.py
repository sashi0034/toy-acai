"""A simple stateful controller for the non-learning team.

The controller intentionally works with the objects exposed by the current
``toy_acai_core`` nanobind module.  In particular, target bearings are
calculated with ``compute_relative_pose`` rather than relying on the array
layout used by the old simulator.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

from ..core import core
from ..math_utils import clamp, normalize_angle

if TYPE_CHECKING:
    from ..simulation_context import SimulationContext


# TURN_SIMILAR_DURATION_STEPS の間で同じ方向へ回転入力し続けた場合、TURN_PAUSE_STEPS の間回転入力を無効化
TURN_SIMILAR_DURATION_STEPS = 120
TURN_PAUSE_STEPS = 30
TURN_INPUT_THRESHOLD = 0.15

CRUISE_ACCELERATION = 0.55
TURN_FULL_SCALE_ANGLE = math.pi / 4
FIRE_ANGLE = 0.35
EDGE_MARGIN = 80.0


@dataclass
class MemberState:
    turn_direction_sign: int = 0
    similar_turn_steps: int = 0
    turn_pause_remaining: int = 0


class RuleBasedAI:
    """Control one team by pursuing the nearest living opponent.

    A short pause after sustained turns keeps fighters from orbiting forever
    when a target is behind them.  The instance is stateful, so call ``reset``
    at the start of each episode.
    """

    def __init__(self, team_id: int = 1):
        self.team_id = team_id
        self._member_states: dict[int, MemberState] = {}

    def reset(self) -> None:
        self._member_states.clear()

    def inputs(self, ctx: SimulationContext) -> dict[int, core.FighterInput]:
        """Return ``FighterInput`` values keyed by controlled fighter index."""
        fighters = ctx.battlefield.fighters
        target_team_id = 1 - self.team_id
        targets = [
            fighter
            for fighter in fighters
            if fighter.team_id == target_team_id and fighter.health > 0.0
        ]

        inputs: dict[int, core.FighterInput] = {}
        for fighter_index, fighter in enumerate(fighters):
            if fighter.team_id != self.team_id or fighter.health <= 0.0:
                continue

            if targets:
                target = min(
                    targets,
                    key=lambda target: fighter.position.distance_from_sq(
                        target.position
                    ),
                )
                relative_pose = core.compute_relative_pose(
                    core.AbsolutePose(fighter),
                    core.AbsolutePose(target),
                )
                # compute_relative_pose uses x=right and y=forward in local
                # coordinates.  Thus atan2(right, forward) is the yaw error.
                yaw_delta = math.atan2(
                    relative_pose.relative_position.x,
                    relative_pose.relative_position.y,
                )
                turn = self._turn_for_yaw_delta(yaw_delta)
                fire = abs(yaw_delta) < FIRE_ANGLE
            else:
                turn = self._turn_toward_battlefield_center(ctx, fighter)
                fire = False

            inputs[fighter_index] = core.FighterInput(
                CRUISE_ACCELERATION,
                self._apply_turn_pause_rule(fighter_index, turn),
                fire,
            )

        return inputs

    def _turn_toward_battlefield_center(
        self, ctx: SimulationContext, fighter: core.FighterState
    ) -> float:
        area = ctx.battlefield.battlefield_area
        if (
            EDGE_MARGIN <= fighter.position.x <= area.w - EDGE_MARGIN
            and EDGE_MARGIN <= fighter.position.y <= area.h - EDGE_MARGIN
        ):
            return 0.0

        center_yaw = math.atan2(
            area.h * 0.5 - fighter.position.y,
            area.w * 0.5 - fighter.position.x,
        )
        return self._turn_for_yaw_delta(normalize_angle(center_yaw - fighter.yaw))

    @staticmethod
    def _turn_for_yaw_delta(yaw_delta: float) -> float:
        return clamp(yaw_delta / TURN_FULL_SCALE_ANGLE, -1.0, 1.0)

    def _apply_turn_pause_rule(self, fighter_index: int, turn: float) -> float:
        state = self._member_states.setdefault(fighter_index, MemberState())
        if state.turn_pause_remaining > 0:
            state.turn_pause_remaining -= 1
            state.turn_direction_sign = 0
            state.similar_turn_steps = 0
            return 0.0

        if turn >= TURN_INPUT_THRESHOLD:
            direction = 1
        elif turn <= -TURN_INPUT_THRESHOLD:
            direction = -1
        else:
            direction = 0

        if direction != 0 and direction == state.turn_direction_sign:
            state.similar_turn_steps += 1
        else:
            state.turn_direction_sign = direction
            state.similar_turn_steps = 1 if direction != 0 else 0

        if state.similar_turn_steps >= TURN_SIMILAR_DURATION_STEPS:
            state.turn_direction_sign = 0
            state.similar_turn_steps = 0
            state.turn_pause_remaining = TURN_PAUSE_STEPS - 1
            return 0.0

        return turn
