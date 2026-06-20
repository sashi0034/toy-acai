"""A simple stateful controller for the non-learning team.

The controller intentionally works with the objects exposed by the current
``toy_acai_core`` nanobind module.  In particular, target bearings are
calculated with ``compute_relative_pose`` rather than relying on the array
layout used by the old simulator.
"""

from __future__ import annotations

import math
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from ..simulation_context import SimulationContext


# TURN_SIMILAR_DURATION_STEPS の間で同じ方向へ回転入力し続けた場合、TURN_PAUSE_STEPS の間回転入力を無効化
TURN_SIMILAR_DURATION_STEPS = 120
TURN_PAUSE_STEPS = 30
TURN_INPUT_THRESHOLD = 0.15

CRUISE_ACCELERATION = 0.55
TURN_FULL_SCALE_ANGLE = 0.7
FIRE_ANGLE = 0.35
EDGE_MARGIN = 80.0


class RuleBasedAI:
    """Control one team by pursuing the nearest living opponent.

    A short pause after sustained turns keeps fighters from orbiting forever
    when a target is behind them.  The instance is stateful, so call ``reset``
    at the start of each episode.
    """

    def __init__(self, team_id: int = 1):
        self.team_id = team_id
        self._similar_turn_steps: dict[int, int] = {}
        self._turn_pause_remaining: dict[int, int] = {}
        self._turn_direction_sign: dict[int, int] = {}

    def reset(self) -> None:
        self._similar_turn_steps.clear()
        self._turn_pause_remaining.clear()
        self._turn_direction_sign.clear()

    def inputs(self, ctx: SimulationContext) -> dict[int, Any]:
        """Return ``FighterInput`` values keyed by controlled fighter index."""
        fighters = ctx.battlefiled.fighters
        target_team_id = 1 - self.team_id
        target_indices = [
            index
            for index, fighter in enumerate(fighters)
            if fighter.team_id == target_team_id and fighter.health > 0.0
        ]

        inputs: dict[int, Any] = {}
        for fighter_index, fighter in enumerate(fighters):
            if fighter.team_id != self.team_id or fighter.health <= 0.0:
                continue

            if target_indices:
                target_index = min(
                    target_indices,
                    key=lambda index: self._distance_squared(fighter, fighters[index]),
                )
                relative_pose = ctx.m.compute_relative_pose(
                    ctx.m.AbsolutePose(fighter),
                    ctx.m.AbsolutePose(fighters[target_index]),
                )
                # compute_relative_pose uses x=right and y=forward in local
                # coordinates.  Thus atan2(right, forward) is the yaw error.
                yaw_delta = math.atan2(
                    relative_pose.relative_position.x,
                    relative_pose.relative_position.y,
                )
                turn = self._clamp(yaw_delta / TURN_FULL_SCALE_ANGLE, -1.0, 1.0)
                fire = abs(yaw_delta) < FIRE_ANGLE
            else:
                turn = self._turn_toward_battlefield_center(ctx, fighter)
                fire = False

            inputs[fighter_index] = ctx.m.FighterInput(
                CRUISE_ACCELERATION,
                self._apply_turn_pause_rule(fighter_index, turn),
                fire,
            )

        return inputs

    @staticmethod
    def _distance_squared(first: Any, second: Any) -> float:
        dx = first.position.x - second.position.x
        dy = first.position.y - second.position.y
        return dx * dx + dy * dy

    def _turn_toward_battlefield_center(
        self, ctx: SimulationContext, fighter: Any
    ) -> float:
        area = ctx.battlefiled.battlefield_area
        if (
            EDGE_MARGIN <= fighter.position.x <= area.w - EDGE_MARGIN
            and EDGE_MARGIN <= fighter.position.y <= area.h - EDGE_MARGIN
        ):
            return 0.0

        center_yaw = math.atan2(
            area.h * 0.5 - fighter.position.y,
            area.w * 0.5 - fighter.position.x,
        )
        yaw_delta = self._angle_delta(center_yaw, fighter.yaw)
        return self._clamp(yaw_delta / TURN_FULL_SCALE_ANGLE, -1.0, 1.0)

    def _apply_turn_pause_rule(self, fighter_index: int, turn: float) -> float:
        pause_remaining = self._turn_pause_remaining.get(fighter_index, 0)
        if pause_remaining > 0:
            self._turn_pause_remaining[fighter_index] = pause_remaining - 1
            self._similar_turn_steps[fighter_index] = 0
            self._turn_direction_sign[fighter_index] = 0
            return 0.0

        turn_sign = 0
        if abs(turn) >= TURN_INPUT_THRESHOLD:
            turn_sign = 1 if turn > 0.0 else -1

        previous_sign = self._turn_direction_sign.get(fighter_index, 0)
        if turn_sign != 0 and turn_sign == previous_sign:
            similar_steps = self._similar_turn_steps.get(fighter_index, 0) + 1
        elif turn_sign != 0:
            similar_steps = 1
        else:
            similar_steps = 0

        self._similar_turn_steps[fighter_index] = similar_steps
        self._turn_direction_sign[fighter_index] = turn_sign

        if similar_steps >= TURN_SIMILAR_DURATION_STEPS:
            self._similar_turn_steps[fighter_index] = 0
            self._turn_direction_sign[fighter_index] = 0
            self._turn_pause_remaining[fighter_index] = TURN_PAUSE_STEPS - 1
            return 0.0

        return turn

    @staticmethod
    def _angle_delta(target_yaw: float, current_yaw: float) -> float:
        return (target_yaw - current_yaw + math.pi) % (2.0 * math.pi) - math.pi

    @staticmethod
    def _clamp(value: float, minimum: float, maximum: float) -> float:
        return max(minimum, min(value, maximum))
