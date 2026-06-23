import math
from abc import ABC, abstractmethod
from dataclasses import dataclass

from .. import constants
from ..core import core
from ..simulation_context import WorkerContext
from .rule_based_ai import RuleBasedAI


AGENT_FIGHTER_INDICES = 0
OPPONENT_FIGHTER_INDICES = (4, 5, 6, 7)

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


def _agent_death_penalty(
    ctx: WorkerContext,
    previous_battlefield: core.BattlefieldContext,
    fighter_index: int,
) -> float:
    if (
        previous_battlefield.fighters[fighter_index].health > 0.0
        and ctx.battlefield.fighters[fighter_index].health <= 0.0
    ):
        return 1.0
    return 0.0


def _agent_missile_hit_frames(ctx: WorkerContext, fighter_index: int) -> list[int]:
    return [
        death_event.killer_missile.fired_frame
        for death_event in ctx.battlefield.death_events
        if death_event.reason == core.DeathEvent.Reason.HitByMissile
        and death_event.killer_missile.shooter_fighter_index == fighter_index
    ]


def _agent_missile_delayed_rewards(
    ctx: WorkerContext,
    previous_battlefield: core.BattlefieldContext,
    fighter_index: int,
) -> list[tuple[int, float]]:
    """Return delayed rewards once each of the agent's missiles has an outcome."""
    hit_missile_ids = {
        death_event.killer_missile.id
        for death_event in ctx.battlefield.death_events
        if death_event.reason == core.DeathEvent.Reason.HitByMissile
    }
    active_missile_ids = {missile.id for missile in ctx.battlefield.missiles}
    
    # 今回のフレームで自然消滅したミサイル
    expired_missiles = [
        missile
        for missile in previous_battlefield.missiles
        if missile.id not in active_missile_ids and missile.id not in hit_missile_ids
    ]

    # 命中報酬
    rewards = [
        (fired_frame, 2.0)
        for fired_frame in _agent_missile_hit_frames(ctx, fighter_index)
    ]
    
    # 空振りペナルティ
    rewards.extend(
        (missile.fired_frame, -0.5)
        for missile in expired_missiles
        if missile.shooter_fighter_index == fighter_index
    )
    
    return rewards


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

    def delayed_reward(
        self, ctx: WorkerContext, previous_battlefield: core.BattlefieldContext
    ) -> list[tuple[int, float]]:
        return []

    @abstractmethod
    def is_success(self, ctx: WorkerContext) -> bool: ...


class CurriculumConfig(ABC):
    @property
    @abstractmethod
    def name(self) -> str: ...

    @abstractmethod
    def create_episode(self) -> Curriculum: ...

    @abstractmethod
    def next_config(
        self, progress: CurriculumProgress
    ) -> "CurriculumConfig | None": ...


class MoveCurriculum(Curriculum):
    name = "move"

    class Config(CurriculumConfig):
        @property
        def name(self) -> str:
            return MoveCurriculum.name

        def create_episode(self) -> Curriculum:
            return MoveCurriculum()

        def next_config(self, progress: CurriculumProgress) -> CurriculumConfig | None:
            if (
                progress.update_count >= 50
                and progress.success_rate >= PROMOTION_SUCCESS_RATE
            ):
                return MissileSurvivalCurriculum.Config()
            return None

    def setup_battlefield(self, ctx: WorkerContext):
        _setup_fighters(ctx, opponent_count=0)

        # 境界外を初期配置とする
        # agent = ctx.battlefield.fighters[AGENT_FIGHTER_INDICES]
        # area = ctx.battlefield.battlefield_area.size

        # margin = 120.0
        # edge = int(ctx.rng.random() * 4.0)
        # if edge == 0:
        #     agent.position = core.Vec2(-margin, area.y * ctx.rng.random())
        # elif edge == 1:
        #     agent.position = core.Vec2(area.x + margin, area.y * ctx.rng.random())
        # elif edge == 2:
        #     agent.position = core.Vec2(area.x * ctx.rng.random(), -margin)
        # else:
        #     agent.position = core.Vec2(area.x * ctx.rng.random(), area.y + margin)

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
            return -1.0 * constants.SIMULATION_DELTA_TIME

            # 境界法線方向なら最悪ペナルティ、逆向きになるにつれて緩和するイメージ
            # return (
            #     -1.0
            #     * (1.5 + math.cos(boundary_distance.relative_angle))
            #     * constants.SIMULATION_DELTA_TIME
            # )


class MissileSurvivalCurriculum(Curriculum):
    name = "missile_survival"

    class Config(CurriculumConfig):
        @property
        def name(self) -> str:
            return MissileSurvivalCurriculum.name

        def create_episode(self) -> Curriculum:
            return MissileSurvivalCurriculum()

        def next_config(self, progress: CurriculumProgress) -> CurriculumConfig | None:
            # ミサイル全回避まで続ける
            if progress.success_rate >= 1.0:
                return RandomOpponentCurriculum.Config()
            return None

    def __init__(self):
        pass

    def setup_battlefield(self, ctx: WorkerContext):
        _setup_fighters(ctx, opponent_count=0)

    def before_step(self, ctx: WorkerContext):
        missile_spawn_interval_steps = round(1.5 / constants.SIMULATION_DELTA_TIME)
        if (ctx.battlefield.frame_count + 1) % missile_spawn_interval_steps == 0:
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

        death_penalty = _agent_death_penalty(
            ctx, previous_battlefield, AGENT_FIGHTER_INDICES
        )
        if death_penalty != 0.0:
            return -death_penalty

        boundary_distance = core.compute_distance_from_boundary(
            ctx.battlefield, AGENT_FIGHTER_INDICES
        )
        if boundary_distance.distance > 0.0:
            return (
                inputs[AGENT_FIGHTER_INDICES].acceleration
                * constants.SIMULATION_DELTA_TIME
            )
        else:
            return -1.0 * constants.SIMULATION_DELTA_TIME


class RandomOpponentCurriculum(Curriculum):
    name = "random_opponents_4"
    opponent_count = 4

    class Config(CurriculumConfig):
        @property
        def name(self) -> str:
            return RandomOpponentCurriculum.name

        def create_episode(self) -> Curriculum:
            return RandomOpponentCurriculum()

        def next_config(self, progress: CurriculumProgress) -> CurriculumConfig | None:
            if progress.success_rate >= PROMOTION_SUCCESS_RATE:
                return RuleBasedOpponentCurriculum.Config(opponent_count=1)
            return None

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

        # 場外ペナルティ
        reward = 0
        if _is_inside_battlefield(ctx, agent.position):
            reward += -1.0 * constants.SIMULATION_DELTA_TIME

        # 低速ペナルティ
        # if agent.speed < 50.0 and inputs[AGENT_FIGHTER_INDICES].acceleration <= 0.0:
        #     reward += -0.1 * constants.SIMULATION_DELTA_TIME

        # 被撃墜ペナルティ
        reward -= (
            _agent_death_penalty(ctx, previous_battlefield, AGENT_FIGHTER_INDICES) * 1.0
        )

        # nearest_opponent = min(
        #     (
        #         fighter
        #         for fighter in ctx.battlefield.fighters
        #         if fighter.team_id == 1 and fighter.health > 0.0
        #     ),
        #     key=lambda fighter: agent.position.distance_from_sq(fighter.position),
        #     default=None,
        # )
        # if nearest_opponent is not None and agent.health > 0.0:
        #     relative_pose = core.compute_relative_pose(
        #         core.AbsolutePose(agent), core.AbsolutePose(nearest_opponent)
        #     )

        #     # 近距離で敵機に正面を向けている場合は報酬を与える
        #     if (
        #         relative_pose.relative_position.length_sq() < 200.0**2
        #         and math.sin(relative_pose.relative_bearing) > math.sqrt(2) / 2
        #     ):
        #         reward += (
        #             0.1
        #             * math.cos(relative_pose.relative_bearing)
        #             * constants.SIMULATION_DELTA_TIME
        #         )

        return reward

    def delayed_reward(
        self, ctx: WorkerContext, previous_battlefield: core.BattlefieldContext
    ) -> list[tuple[int, float]]:
        return _agent_missile_delayed_rewards(
            ctx, previous_battlefield, AGENT_FIGHTER_INDICES
        )


class RuleBasedOpponentCurriculum(Curriculum):
    class Config(CurriculumConfig):
        def __init__(self, opponent_count: int):
            self.opponent_count = opponent_count

        @property
        def name(self) -> str:
            return f"rule_based_opponents_{self.opponent_count}"

        def create_episode(self) -> Curriculum:
            return RuleBasedOpponentCurriculum(self.opponent_count)

        def next_config(self, progress: CurriculumProgress) -> CurriculumConfig | None:
            if (
                progress.success_rate >= PROMOTION_SUCCESS_RATE
                and self.opponent_count < 3
            ):
                return RuleBasedOpponentCurriculum.Config(self.opponent_count + 1)
            return None

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
        return -_agent_death_penalty(ctx, previous_battlefield, AGENT_FIGHTER_INDICES)

    def delayed_reward(
        self, ctx: WorkerContext, previous_battlefield: core.BattlefieldContext
    ) -> list[tuple[int, float]]:
        return _agent_missile_delayed_rewards(
            ctx, previous_battlefield, AGENT_FIGHTER_INDICES
        )


class CurriculumController:
    """親プロセスでカリキュラムの段階と進捗だけを管理する"""

    def __init__(self):
        self.config: CurriculumConfig = initial_curriculum
        self.progress = CurriculumProgress()

    @property
    def name(self) -> str:
        return self.config.name

    def create_episode(self) -> Curriculum:
        return self.config.create_episode()

    def record_episode(self, is_success: bool):
        self.progress.record_episode(is_success)

    def after_update(self) -> tuple[str, str] | None:
        self.progress.complete_update()
        next_config = self.config.next_config(self.progress)
        if next_config is None:
            return None

        previous_name = self.name
        self.config = next_config
        self.progress = CurriculumProgress()
        return previous_name, self.name


initial_curriculum = MoveCurriculum.Config()
