import math
from abc import ABC, abstractmethod
from dataclasses import dataclass
from enum import Enum

from .. import constants
from ..core import core
from ..simulation_context import WorkerContext
from .rule_based_ai import RuleBasedAI


AGENT_FIGHTER_INDICES = 0
OPPONENT_FIGHTER_INDICES = (4, 5, 6, 7)

MOVE_CURRICULUM_UPDATES = 10
PROMOTION_SUCCESS_RATE = 0.8


def _is_inside_battlefield(ctx: WorkerContext, position: core.Vec2) -> bool:
    area = ctx.battlefield.battlefield_area
    return 0.0 <= position.x <= area.w and 0.0 <= position.y <= area.h


def _setup_fighters(ctx: WorkerContext, opponent_count: int):
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
                margin + (area.x - 2 * margin) * ctx.rng.random(),
                margin + (area.y - 2 * margin) * ctx.rng.random(),
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
        fighter.yaw = 2 * math.pi * ctx.rng.random()
        occupied_positions.append(position)


def _is_combat_terminal(ctx: WorkerContext, opponent_count: int) -> bool:
    agent = ctx.battlefield.fighters[AGENT_FIGHTER_INDICES]
    red_alive = any(
        fighter.health > 0.0 and fighter.team_id == 1
        for fighter in ctx.battlefield.fighters
    )
    return agent.health <= 0.0 or (opponent_count > 0 and not red_alive)


def _kill_death_reward(
    ctx: WorkerContext, previous_battlefield: core.BattlefieldContext
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


def _is_agent_winner(ctx: WorkerContext, opponent_count: int) -> bool:
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
    def setup_battlefield(self, ctx: WorkerContext): ...

    @abstractmethod
    def before_step(self, ctx: WorkerContext): ...

    @abstractmethod
    def opponent_inputs(self, ctx: WorkerContext) -> dict[int, core.FighterInput]: ...

    @abstractmethod
    def is_terminal(self, ctx: WorkerContext) -> bool: ...

    @abstractmethod
    def step_reward(
        self,
        ctx: WorkerContext,
        previous_battlefield: core.BattlefieldContext,
        inputs: list[core.FighterInput],
    ) -> float: ...

    @abstractmethod
    def is_success(self, ctx: WorkerContext) -> bool: ...


class MoveCurriculum(Curriculum):
    name = "move"

    def setup_battlefield(self, ctx: WorkerContext):
        _setup_fighters(ctx, opponent_count=0)

    def before_step(self, ctx: WorkerContext):
        pass

    def opponent_inputs(self, ctx: WorkerContext) -> dict[int, core.FighterInput]:
        return {}

    def is_terminal(self, ctx: WorkerContext) -> bool:
        return _is_combat_terminal(ctx, opponent_count=0)

    def is_success(self, ctx: WorkerContext) -> bool:
        return _is_agent_winner(ctx, opponent_count=0)

    def step_reward(
        self,
        ctx: WorkerContext,
        previous_battlefield: core.BattlefieldContext,
        inputs: list[core.FighterInput],
    ) -> float:
        agent = ctx.battlefield.fighters[AGENT_FIGHTER_INDICES]
        # if agent.health <= 0.0:
        #     return (
        #         -5.0
        #         if previous_battlefield.fighters[AGENT_FIGHTER_INDICES].health > 0.0
        #         else 0.0
        #     )

        boundary_distance = core.compute_distance_from_boundary(
            ctx.battlefield, AGENT_FIGHTER_INDICES
        )
        if boundary_distance.distance > 0.0:
            return (
                inputs[AGENT_FIGHTER_INDICES].acceleration
                * constants.SIMULATION_DELTA_TIME
            )
        else:
            # 境界法線方向なら最悪ペナルティ、逆向きになるにつれて緩和するイメージ
            return (
                -1.0
                * (1.0 + math.cos(boundary_distance.relative_angle))
                * constants.SIMULATION_DELTA_TIME
            )


class MissileSurvivalCurriculum(Curriculum):
    name = "missile_survival"

    def __init__(self):
        self.step_count = 0

    def setup_battlefield(self, ctx: WorkerContext):
        _setup_fighters(ctx, opponent_count=0)
        self.step_count = 0

    def before_step(self, ctx: WorkerContext):
        self.step_count += 1  # TODO: battlefield の frameCount を使うようにする

        missile_spawn_interval_steps = round(1.5 / constants.SIMULATION_DELTA_TIME)
        if self.step_count % missile_spawn_interval_steps == 0:
            # ミサイル発射
            self._fire_missile_around_player(ctx)

    def _fire_missile_around_player(self, ctx: WorkerContext):
        agent = ctx.battlefield.fighters[AGENT_FIGHTER_INDICES]

        angle = 2.0 * math.pi * ctx.rng.random()
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

    def opponent_inputs(self, ctx: WorkerContext) -> dict[int, core.FighterInput]:
        return {}

    def is_terminal(self, ctx: WorkerContext) -> bool:
        return _is_combat_terminal(ctx, opponent_count=0)

    def is_success(self, ctx: WorkerContext) -> bool:
        return _is_agent_winner(ctx, opponent_count=0)

    def step_reward(
        self,
        ctx: WorkerContext,
        previous_battlefield: core.BattlefieldContext,
        inputs: list[core.FighterInput],
    ) -> float:
        agent = ctx.battlefield.fighters[AGENT_FIGHTER_INDICES]

        kill_death_reward = _kill_death_reward(ctx, previous_battlefield)
        if kill_death_reward != 0.0:
            return kill_death_reward * 1.0

        boundary_distance = core.compute_distance_from_boundary(
            ctx.battlefield, AGENT_FIGHTER_INDICES
        )
        if boundary_distance.distance > 0.0:
            return (
                inputs[AGENT_FIGHTER_INDICES].acceleration
                * constants.SIMULATION_DELTA_TIME
            )
        else:
            # 境界法線方向なら最悪ペナルティ、逆向きになるにつれて緩和するイメージ
            return (
                -1.0
                * (1.0 + math.cos(boundary_distance.relative_angle))
                * constants.SIMULATION_DELTA_TIME
            )


class RandomOpponentCurriculum(Curriculum):
    name = "random_opponents_4"
    opponent_count = 4

    def setup_battlefield(self, ctx: WorkerContext):
        _setup_fighters(ctx, opponent_count=self.opponent_count)

    def before_step(self, ctx: WorkerContext):
        pass

    def opponent_inputs(self, ctx: WorkerContext) -> dict[int, core.FighterInput]:
        return {
            fighter_index: core.FighterInput(
                -1.0,
                -1.0 if fighter_index % 2 == 0 else 1.0,
                ctx.rng.random() < 0.15,
            )
            for fighter_index in OPPONENT_FIGHTER_INDICES
            if ctx.battlefield.fighters[fighter_index].health > 0.0
        }

    def is_terminal(self, ctx: WorkerContext) -> bool:
        return _is_combat_terminal(ctx, opponent_count=self.opponent_count)

    def is_success(self, ctx: WorkerContext) -> bool:
        return _is_agent_winner(ctx, opponent_count=self.opponent_count)

    def step_reward(
        self,
        ctx: WorkerContext,
        previous_battlefield: core.BattlefieldContext,
        inputs: list[core.FighterInput],
    ) -> float:
        agent = ctx.battlefield.fighters[AGENT_FIGHTER_INDICES]

        # TODO: 撃墜報酬を発射時点に遡って与える

        # 場外ペナルティ
        reward = 0
        if _is_inside_battlefield(ctx, agent.position):
            reward += -1.0 * constants.SIMULATION_DELTA_TIME

        # 低速ペナルティ
        if agent.speed < 50.0 and inputs[AGENT_FIGHTER_INDICES].acceleration <= 0.0:
            reward += -0.1 * constants.SIMULATION_DELTA_TIME

        # 撃破と被弾報酬
        reward += _kill_death_reward(ctx, previous_battlefield) * 5

        nearest_opponent = min(
            (
                fighter
                for fighter in ctx.battlefield.fighters
                if fighter.team_id == 1 and fighter.health > 0.0
            ),
            key=lambda fighter: agent.position.distance_from_sq(fighter.position),
            default=None,
        )
        if nearest_opponent is not None and agent.health > 0.0:
            relative_pose = core.compute_relative_pose(
                core.AbsolutePose(agent), core.AbsolutePose(nearest_opponent)
            )

            # 近距離で敵機に正面を向けている場合は報酬を与える
            if (
                relative_pose.relative_position.length_sq() < 200.0**2
                and math.sin(relative_pose.relative_bearing) > math.sqrt(2) / 2
            ):
                reward += (
                    0.1
                    * math.cos(relative_pose.relative_bearing)
                    * constants.SIMULATION_DELTA_TIME
                )

        return reward


class RuleBasedOpponentCurriculum(Curriculum):
    def __init__(self, opponent_count: int):
        self.opponent_count = opponent_count
        self.name = f"rule_based_opponents_{opponent_count}"
        self.opponent_ai = RuleBasedAI(team_id=1)

    def setup_battlefield(self, ctx: WorkerContext):
        _setup_fighters(ctx, opponent_count=self.opponent_count)
        self.opponent_ai.reset()

    def before_step(self, ctx: WorkerContext):
        pass

    def opponent_inputs(self, ctx: WorkerContext) -> dict[int, core.FighterInput]:
        return self.opponent_ai.inputs(ctx)

    def is_terminal(self, ctx: WorkerContext) -> bool:
        return _is_combat_terminal(ctx, opponent_count=self.opponent_count)

    def is_success(self, ctx: WorkerContext) -> bool:
        return _is_agent_winner(ctx, opponent_count=self.opponent_count)

    def step_reward(
        self,
        ctx: WorkerContext,
        previous_battlefield: core.BattlefieldContext,
        inputs: list[core.FighterInput],
    ) -> float:
        return _kill_death_reward(ctx, previous_battlefield)


class CurriculumKind(str, Enum):
    MOVE = "move"
    MISSILE_SURVIVAL = "missile_survival"
    RANDOM_OPPONENTS = "random_opponents"
    RULE_BASED_OPPONENTS = "rule_based_opponents"


class CurriculumController:
    """親プロセスでカリキュラムの段階と進捗だけを管理する。"""

    def __init__(self):
        self.kind = CurriculumKind.MOVE
        self.opponent_count = 0
        self.progress = CurriculumProgress()

    # TODO: 分岐を簡単化したい

    @property
    def name(self) -> str:
        if self.kind == CurriculumKind.RANDOM_OPPONENTS:
            return RandomOpponentCurriculum.name
        if self.kind == CurriculumKind.RULE_BASED_OPPONENTS:
            return f"rule_based_opponents_{self.opponent_count}"
        return self.kind.value

    def create_episode(self) -> Curriculum:
        if self.kind == CurriculumKind.MOVE:
            return MoveCurriculum()
        if self.kind == CurriculumKind.MISSILE_SURVIVAL:
            return MissileSurvivalCurriculum()
        if self.kind == CurriculumKind.RANDOM_OPPONENTS:
            return RandomOpponentCurriculum()
        if self.kind == CurriculumKind.RULE_BASED_OPPONENTS:
            return RuleBasedOpponentCurriculum(self.opponent_count)
        raise ValueError(f"Unknown curriculum kind: {self.kind}")

    def record_episode(self, is_success: bool):
        self.progress.record_episode(is_success)

    def after_update(self) -> tuple[str, str] | None:
        self.progress.complete_update()
        next_kind = None
        next_opponent_count = 0

        if self.kind == CurriculumKind.MOVE:
            if (
                self.progress.update_count >= MOVE_CURRICULUM_UPDATES
                and self.progress.success_rate >= PROMOTION_SUCCESS_RATE
            ):
                next_kind = CurriculumKind.MISSILE_SURVIVAL
        elif self.kind == CurriculumKind.MISSILE_SURVIVAL:
            # ミサイル全回避まで続ける
            if self.progress.success_rate >= 1.0:
                next_kind = CurriculumKind.RANDOM_OPPONENTS
        elif self.kind == CurriculumKind.RANDOM_OPPONENTS:
            if self.progress.success_rate >= PROMOTION_SUCCESS_RATE:
                next_kind = CurriculumKind.RULE_BASED_OPPONENTS
                next_opponent_count = 1
        elif self.kind == CurriculumKind.RULE_BASED_OPPONENTS:
            if (
                self.progress.success_rate >= PROMOTION_SUCCESS_RATE
                and self.opponent_count < 3
            ):
                next_kind = CurriculumKind.RULE_BASED_OPPONENTS
                next_opponent_count = self.opponent_count + 1
        else:
            raise ValueError(f"Unknown curriculum kind: {self.kind}")

        if next_kind is None:
            return None

        previous_name = self.name
        self.kind = next_kind
        self.opponent_count = next_opponent_count
        self.progress = CurriculumProgress()
        return previous_name, self.name
