import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

from toy_acai_rl import curriculum  # noqa: E402


class CurriculumPromotionTest(unittest.TestCase):
    def test_promotes_on_fourteen_wins_out_of_twenty(self):
        promote, reason = curriculum.should_promote_curriculum_stage(
            stage_index=0,
            stage_episode=200,
            wins=14,
            evals=20,
        )

        self.assertTrue(promote)
        self.assertEqual(reason, "win_rate")

    def test_continues_below_threshold_before_stage_max(self):
        promote, reason = curriculum.should_promote_curriculum_stage(
            stage_index=0,
            stage_episode=curriculum.CURRICULUM_STAGE_MAX_EPISODES - 1,
            wins=13,
            evals=20,
        )

        self.assertFalse(promote)
        self.assertEqual(reason, "continue")

    def test_force_promotes_at_stage_max(self):
        promote, reason = curriculum.should_promote_curriculum_stage(
            stage_index=0,
            stage_episode=curriculum.CURRICULUM_STAGE_MAX_EPISODES,
            wins=13,
            evals=20,
        )

        self.assertTrue(promote)
        self.assertEqual(reason, "stage_max")

    def test_final_stage_does_not_promote(self):
        promote, reason = curriculum.should_promote_curriculum_stage(
            stage_index=len(curriculum.CURRICULUM_STAGES) - 1,
            stage_episode=curriculum.CURRICULUM_STAGE_MAX_EPISODES,
            wins=20,
            evals=20,
        )

        self.assertFalse(promote)
        self.assertEqual(reason, "final_stage")


if __name__ == "__main__":
    unittest.main()
