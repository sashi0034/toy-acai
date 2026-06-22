from ..core import core


def get_alive_fighters_sorted_by_distance(
    battlefield: core.BattlefieldContext, pose: core.AbsolutePose, team_id: int
) -> list[tuple[core.RelativePose, int]]:
    fighter_futures = []
    for i, fighter in enumerate(battlefield.fighters):
        if fighter.health <= 0.0 or fighter.team_id != team_id:
            continue
        relative_pose = core.compute_relative_pose(pose, core.AbsolutePose(fighter))
        fighter_futures.append((relative_pose, i))

    fighter_futures.sort(key=lambda future: future[0].relative_position.length_sq())
    return fighter_futures


def get_missiles_sorted_by_distance(
    battlefield: core.BattlefieldContext, pose: core.AbsolutePose, team_id: int
) -> list[tuple[core.RelativePose, int]]:
    missile_futures = []
    for i, missile in enumerate(battlefield.missiles):
        if missile.team_id != team_id:
            continue
        relative_pose = core.compute_relative_pose(pose, core.AbsolutePose(missile))
        missile_futures.append((relative_pose, i))

    missile_futures.sort(key=lambda future: future[0].relative_position.length_sq())
    return missile_futures
