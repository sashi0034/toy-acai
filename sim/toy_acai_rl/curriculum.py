from typing import Tuple


CURRICULUM_LEARNER_COUNT = 1
CURRICULUM_STAGES = (1, 2, 3, 4)
CURRICULUM_PROMOTION_EVALS = 20
CURRICULUM_PROMOTION_WINS = 14
CURRICULUM_EVAL_EVERY = 200
CURRICULUM_STAGE_MAX_EPISODES = 10000


def should_promote_curriculum_stage(
    *,
    stage_index: int,
    stage_episode: int,
    wins: int,
    evals: int,
) -> Tuple[bool, str]:
    if stage_index >= len(CURRICULUM_STAGES) - 1:
        return False, "final_stage"
    if evals > 0 and wins >= CURRICULUM_PROMOTION_WINS:
        return True, "win_rate"
    if stage_episode >= CURRICULUM_STAGE_MAX_EPISODES:
        return True, "stage_max"
    return False, "continue"
