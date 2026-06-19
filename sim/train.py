#!/usr/bin/env python3
import math
import random

from PIL import Image

from .simulation_context import SimulationContext, output_path
from . import constants


def setup_battlefield(ctx: SimulationContext):
    ctx.m.init_battlefield(ctx.battlefiled)

    for fighter in ctx.battlefiled.fighters:
        fighter.health = 0.0
        fighter.speed = 0.0

    blue = ctx.battlefiled.fighters[0]
    blue.position = ctx.m.Vec2(240.0, 450.0)
    blue.yaw = 0.0
    blue.speed = 130.0
    blue.health = 1.0
    blue.missile_cooldown = 0.0
    blue.out_of_bounds_time = 0.0

    for fighter_index, position in ((4, (1360.0, 300.0)), (5, (1360.0, 600.0))):
        fighter = ctx.battlefiled.fighters[fighter_index]
        fighter.position = ctx.m.Vec2(*position)
        fighter.yaw = math.pi
        fighter.speed = 130.0
        fighter.health = 1.0
        fighter.missile_cooldown = 0.0
        fighter.out_of_bounds_time = 0.0


def random_inputs(ctx: SimulationContext, rng):
    return [
        ctx.m.FighterInput(
            rng.uniform(-1.0, 1.0),
            rng.uniform(-1.0, 1.0),
            rng.random() < 0.15,
        )
        for _ in range(ctx.m.FIGHTER_COUNT)
    ]


def run():
    ctx = SimulationContext()
    battlefiled = ctx.battlefiled
    renderer = ctx.renderer

    setup_battlefield(ctx)

    renderer.enable_render_to_image_buffer(ctx.m.Size(*constants.RENDER_SIZE))

    rng = random.Random()
    frames = []
    max_step_count = round(
        constants.MAX_SIMULATION_SECONDS / constants.SIMULATION_DELTA_TIME
    )
    render_every_steps = round(
        constants.RENDER_INTERVAL / constants.SIMULATION_DELTA_TIME
    )

    # シミュレーションループ
    for step in range(1, max_step_count + 1):
        ctx.m.update_battlefield(
            battlefiled, random_inputs(ctx, rng), constants.SIMULATION_DELTA_TIME
        )
        renderer.update(battlefiled, constants.SIMULATION_DELTA_TIME)

        if step % render_every_steps == 0:
            renderer.render(battlefiled)
            frames.append(Image.fromarray(renderer.image_buffer(), mode="RGBA").copy())

    if not frames:
        print("No frames were rendered.")
        return

    # GIF 作成
    output_directory = output_path()
    output_directory.mkdir(parents=True, exist_ok=True)
    gif_path = output_directory / "result.gif"
    frames[0].save(
        gif_path,
        format="GIF",
        append_images=frames[1:],
        save_all=True,
        duration=round(constants.RENDER_INTERVAL * 1000),
        loop=0,
        disposal=2,
    )
    print(f"Saved {len(frames)} frames to {gif_path}")


if __name__ == "__main__":
    run()
