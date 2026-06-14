import sys
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

from toy_acai_rl.env import terminal_score


class TerminalScoreTest(unittest.TestCase):
    def score(self, blue_alive, red_alive, episode_steps=50):
        return terminal_score(
            blue_alive=blue_alive,
            red_alive=red_alive,
            episode_steps=episode_steps,
            max_steps=100,
            team_size=4,
        )

    def test_red_eliminated_beats_any_non_win(self):
        worst_win = self.score(blue_alive=0, red_alive=0, episode_steps=100)
        best_non_win = self.score(blue_alive=4, red_alive=1, episode_steps=1)
        self.assertGreater(worst_win, best_non_win)

    def test_wins_prefer_more_blue_alive_before_speed(self):
        slow_more_survivors = self.score(blue_alive=2, red_alive=0, episode_steps=100)
        fast_fewer_survivors = self.score(blue_alive=1, red_alive=0, episode_steps=1)
        self.assertGreater(slow_more_survivors, fast_fewer_survivors)

    def test_wins_tie_break_by_earlier_finish(self):
        fast = self.score(blue_alive=2, red_alive=0, episode_steps=25)
        slow = self.score(blue_alive=2, red_alive=0, episode_steps=75)
        self.assertGreater(fast, slow)

    def test_non_wins_prefer_fewer_red_alive_before_blue_alive(self):
        fewer_red_no_survivors = self.score(blue_alive=0, red_alive=1)
        more_red_all_survivors = self.score(blue_alive=4, red_alive=2)
        self.assertGreater(fewer_red_no_survivors, more_red_all_survivors)

    def test_non_wins_tie_break_by_more_blue_alive(self):
        more_blue = self.score(blue_alive=3, red_alive=2)
        fewer_blue = self.score(blue_alive=1, red_alive=2)
        self.assertGreater(more_blue, fewer_blue)


if __name__ == "__main__":
    unittest.main()
