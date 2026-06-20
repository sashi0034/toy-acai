import math
import random
from abc import ABC, abstractmethod
from dataclasses import dataclass

from sim.math_utils import clamp

from .. import constants
from ..core import core
from ..simulation_context import SimulationContext
from .rule_based_ai import RuleBasedAI


AGENT_FIGHTER_INDICES = 0
OPPONENT_FIGHTER_INDICES = (4, 5, 6, 7)

MOVE_CURRICULUM_UPDATES = 10
PROMOTION_SUCCESS_RATE = 0.8


def initial_curriculum(ctx: SimulationContext) -> "Curriculum":
    return MissileSurvivalCurriculum(ctx)


def _is_inside_battlefield(ctx: SimulationContext, position: core.Vec2) -> bool:
    area = ctx.battlefield.battlefield_area
    return 0.0 <= position.x <= area.w and 0.0 <= position.y <= area.h


def _setup_fighters(ctx: SimulationContext, rng: random.Random, opponent_count: int):
    core.init_battlefield(ctx.battlefield)

    for fighter in ctx.battlefield.fighters:
        fighter.health = 0.0

    active_fighter_indices = [AGENT_FIGHTER_INDICES]
    active_fighter_indices.extend(OPPONENT_FIGHTER_INDICES[:opponent_count])

    area = ctx.battlefield.battlefield_area.size
    margin = 80.0
    minimum_opponent_distance = 400.0

    occupied_positions: list[core.Vec2] = []
    for fighter_index in active_fighter_indices:
        fighter = ctx.battlefield.fighters[fighter_index]
        fighter.health = 1.0

        position = core.Vec2(0.0, 0.0)
        for _ in range(1000):
            position = core.Vec2(
                margin + (area.x - 2 * margin) * rng.random(),
                margin + (area.y - 2 * margin) * rng.random(),
            )
            minimum_distance = (
                0.0
                if fighter_index == AGENT_FIGHTER_INDICES
                else minimum_opponent_distance
            )
            if all(
                position.distance_from_sq(other) >= minimum_distance**2
                for other in occupied_positions
            ):
                break

        fighter.position = position
        fighter.yaw = 2 * math.pi * rng.random()
        occupied_positions.append(position)


def _is_combat_terminal(ctx: SimulationContext, opponent_count: int) -> bool:
    agent = ctx.battlefield.fighters[AGENT_FIGHTER_INDICES]
    red_alive = any(
        fighter.health > 0.0 and fighter.team_id == 1
        for fighter in ctx.battlefield.fighters
    )
    return agent.health <= 0.0 or (opponent_count > 0 and not red_alive)


def _combat_step_reward(
    ctx: SimulationContext, previous_battlefield: core.BattlefieldContext
) -> float:
    reward = sum(
        1.0
        for hit_event in ctx.battlefield.hit_events
        if hit_event.shooter_fighter_index == AGENT_FIGHTER_INDICES
    )
    if (
        previous_battlefield.fighters[AGENT_FIGHTER_INDICES].health > 0.0
        and ctx.battlefield.fighters[AGENT_FIGHTER_INDICES].health <= 0.0
    ):
        reward -= 1.0
    return reward


def _is_agent_winner(ctx: SimulationContext, opponent_count: int) -> bool:
    agent = ctx.battlefield.fighters[AGENT_FIGHTER_INDICES]
    if agent.health <= 0.0:
        return False
    return opponent_count == 0 or not any(
        fighter.health > 0.0 and fighter.team_id == 1
        for fighter in ctx.battlefield.fighters
    )


@dataclass
class CurriculumProgress:
    episode_count: int = 0
    success_count: int = 0
    update_count: int = 0
    success_rate: float = 0.0

    def record_episode(self, is_success: bool):
        self.episode_count += 1
        if is_success:
            self.success_count += 1

    def complete_update(self):
        if self.episode_count == 0:
            raise RuntimeError("Cannot finish a curriculum update without episodes")

        self.success_rate = self.success_count / self.episode_count
        self.update_count += 1
        self.episode_count = 0
        self.success_count = 0


# カリキュラムの共通インターフェース。振る舞いは各実装が持つ。
class Curriculum(ABC):
    name: str

    @abstractmethod
    def setup_battlefield(self, ctx: SimulationContext, rng: random.Random): ...

    @abstractmethod
    def before_step(self, ctx: SimulationContext, rng: random.Random): ...

    @abstractmethod
    def opponent_inputs(
        self, ctx: SimulationContext, rng: random.Random
    ) -> dict[int, core.FighterInput]: ...

    @abstractmethod
    def is_terminal(self, ctx: SimulationContext) -> bool: ...

    @abstractmethod
    def step_reward(
        self,
        ctx: SimulationContext,
        previous_battlefield: core.BattlefieldContext,
        inputs: list[core.FighterInput],
    ) -> float: ...

    @abstractmethod
    def record_episode(self): ...

    @abstractmethod
    def after_update(self, ctx: SimulationContext) -> "Curriculum | None": ...


class MoveCurriculum(Curriculum):
    name = "move"

    def __init__(self, ctx: SimulationContext):
        self.ctx = ctx
        self.progress = CurriculumProgress()

    def setup_battlefield(self, ctx: SimulationContext, rng: random.Random):
        self.ctx = ctx
        _setup_fighters(ctx, rng, opponent_count=0)

    def before_step(self, ctx: SimulationContext, rng: random.Random):
        pass

    def opponent_inputs(
        self, ctx: SimulationContext, rng: random.Random
    ) -> dict[int, core.FighterInput]:
        return {}

    def is_terminal(self, ctx: SimulationContext) -> bool:
        return _is_combat_terminal(ctx, opponent_count=0)

    def step_reward(
        self,
        ctx: SimulationContext,
        previous_battlefield: core.BattlefieldContext,
        inputs: list[core.FighterInput],
    ) -> float:
        agent = ctx.battlefield.fighters[AGENT_FIGHTER_INDICES]
        if agent.health <= 0.0:
            return (
                -1.0
                if previous_battlefield.fighters[AGENT_FIGHTER_INDICES].health > 0.0
                else 0.0
            )
        if _is_inside_battlefield(ctx, agent.position):
            return (
                inputs[AGENT_FIGHTER_INDICES].acceleration
                * constants.SIMULATION_DELTA_TIME
            )
        else:
            return (
                -clamp(agent.out_of_bounds_time * 2.0, 1.0, 2.0)
                * constants.SIMULATION_DELTA_TIME
            )

    def record_episode(self):
        self.progress.record_episode(_is_agent_winner(self.ctx, opponent_count=0))

    def after_update(self, ctx: SimulationContext) -> "Curriculum | None":
        self.progress.complete_update()

        # if self.progress.update_count >= MOVE_CURRICULUM_UPDATES:
        #     return MissileSurvivalCurriculum(ctx)

        if (
            self.progress.update_count >= MOVE_CURRICULUM_UPDATES
            and self.progress.success_rate >= PROMOTION_SUCCESS_RATE
        ):
            return MissileSurvivalCurriculum(ctx)
        return None


class MissileSurvivalCurriculum(Curriculum):
    name = "missile_survival"

    def __init__(self, ctx: SimulationContext):
        self.ctx = ctx
        self.progress = CurriculumProgress()
        self.step_count = 0

    def setup_battlefield(self, ctx: SimulationContext, rng: random.Random):
        self.ctx = ctx
        _setup_fighters(ctx, rng, opponent_count=0)
        self.step_count = 0

    def before_step(self, ctx: SimulationContext, rng: random.Random):
        self.step_count += 1

        missile_spawn_interval_steps = round(1.5 / constants.SIMULATION_DELTA_TIME)
        if self.step_count % missile_spawn_interval_steps == 0:
            # ミサイル発射
            self._fire_missile_around_player(ctx, rng)
            return

    def _fire_missile_around_player(self, ctx, rng):
        agent = ctx.battlefield.fighters[AGENT_FIGHTER_INDICES]

        angle = 2.0 * math.pi * rng.random()
        distance = 400.0
        position = core.Vec2(
            agent.position.x + math.cos(angle) * distance,
            agent.position.y + math.sin(angle) * distance,
        )

        missile = core.MissileState()

        missile.id = ctx.battlefield.next_missile_id
        ctx.battlefield.next_missile_id += 1

        missile.team_id = 1
        missile.shooter_fighter_index = -1
        missile.target_fighter_index = AGENT_FIGHTER_INDICES

        missile.position = position
        missile.yaw = math.atan2(
            agent.position.y - position.y, agent.position.x - position.x
        )

        missile.speed = 50.0

        ctx.battlefield.missiles.append(missile)

    def opponent_inputs(
        self, ctx: SimulationContext, rng: random.Random
    ) -> dict[int, core.FighterInput]:
        return {}

    def is_terminal(self, ctx: SimulationContext) -> bool:
        return _is_combat_terminal(ctx, opponent_count=0)

    def step_reward(
        self,
        ctx: SimulationContext,
        previous_battlefield: core.BattlefieldContext,
        inputs: list[core.FighterInput],
    ) -> float:
        agent = ctx.battlefield.fighters[AGENT_FIGHTER_INDICES]
        if agent.health <= 0.0:
            return (
                -10.0
                if previous_battlefield.fighters[AGENT_FIGHTER_INDICES].health > 0.0
                else 0.0
            )

        if _is_inside_battlefield(ctx, agent.position):
            return (
                inputs[AGENT_FIGHTER_INDICES].acceleration
                * constants.SIMULATION_DELTA_TIME
            )
        else:
            return (
                -clamp(agent.out_of_bounds_time * 2.0, 1.0, 2.0)
                * constants.SIMULATION_DELTA_TIME
            )

    def record_episode(self):
        self.progress.record_episode(_is_agent_winner(self.ctx, opponent_count=0))

    def after_update(self, ctx: SimulationContext) -> "Curriculum | None":
        self.progress.complete_update()
        if self.progress.success_rate >= PROMOTION_SUCCESS_RATE:
            return RandomOpponentCurriculum(ctx)
        return None


class RandomOpponentCurriculum(Curriculum):
    name = "random_opponents_4"

    def __init__(self, ctx: SimulationContext):
        self.ctx = ctx
        self.opponent_count = 4
        self.progress = CurriculumProgress()

    def setup_battlefield(self, ctx: SimulationContext, rng: random.Random):
        self.ctx = ctx
        _setup_fighters(ctx, rng, opponent_count=self.opponent_count)

    def before_step(self, ctx: SimulationContext, rng: random.Random):
        pass

    def opponent_inputs(
        self, ctx: SimulationContext, rng: random.Random
    ) -> dict[int, core.FighterInput]:
        return {
            fighter_index: core.FighterInput(
                -1.0,
                -1.0 if fighter_index % 2 == 0 else 1.0,
                rng.random() < 0.15,
            )
            for fighter_index in OPPONENT_FIGHTER_INDICES
            if ctx.battlefield.fighters[fighter_index].health > 0.0
        }

    def is_terminal(self, ctx: SimulationContext) -> bool:
        return _is_combat_terminal(ctx, opponent_count=self.opponent_count)

    def step_reward(
        self,
        ctx: SimulationContext,
        previous_battlefield: core.BattlefieldContext,
        inputs: list[core.FighterInput],
    ) -> float:
        return _combat_step_reward(ctx, previous_battlefield)

    def record_episode(self):
        self.progress.record_episode(
            _is_agent_winner(self.ctx, opponent_count=self.opponent_count)
        )

    def after_update(self, ctx: SimulationContext) -> "Curriculum | None":
        self.progress.complete_update()
        if self.progress.success_rate >= PROMOTION_SUCCESS_RATE:
            return RuleBasedOpponentCurriculum(ctx, opponent_count=1)
        return None


class RuleBasedOpponentCurriculum(Curriculum):
    def __init__(self, ctx: SimulationContext, opponent_count: int):
        self.ctx = ctx
        self.opponent_count = opponent_count
        self.progress = CurriculumProgress()
        self.name = f"rule_based_opponents_{opponent_count}"
        self.opponent_ai = RuleBasedAI(team_id=1)

    def setup_battlefield(self, ctx: SimulationContext, rng: random.Random):
        self.ctx = ctx
        _setup_fighters(ctx, rng, opponent_count=self.opponent_count)
        self.opponent_ai.reset()

    def before_step(self, ctx: SimulationContext, rng: random.Random):
        pass

    def opponent_inputs(
        self, ctx: SimulationContext, rng: random.Random
    ) -> dict[int, core.FighterInput]:
        return self.opponent_ai.inputs(ctx)

    def is_terminal(self, ctx: SimulationContext) -> bool:
        return _is_combat_terminal(ctx, opponent_count=self.opponent_count)

    def step_reward(
        self,
        ctx: SimulationContext,
        previous_battlefield: core.BattlefieldContext,
        inputs: list[core.FighterInput],
    ) -> float:
        return _combat_step_reward(ctx, previous_battlefield)

    def record_episode(self):
        self.progress.record_episode(
            _is_agent_winner(self.ctx, opponent_count=self.opponent_count)
        )

    def after_update(self, ctx: SimulationContext) -> "Curriculum | None":
        self.progress.complete_update()
        if self.progress.success_rate < PROMOTION_SUCCESS_RATE:
            return None
        if self.opponent_count >= 3:
            return None
        return RuleBasedOpponentCurriculum(ctx, opponent_count=self.opponent_count + 1)
