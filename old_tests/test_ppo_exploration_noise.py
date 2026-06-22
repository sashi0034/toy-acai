import sys
import unittest
from pathlib import Path

import numpy as np

try:
    import torch
except ModuleNotFoundError:
    torch = None


REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "sim"))

if torch is not None:
    from toy_acai_rl.ppo import PPOConfig, PPOTrainer, _SingleAgentPPOTrainer  # noqa: E402


@unittest.skipIf(torch is None, "PyTorch is required for PPO tests")
class PPOExplorationNoiseTest(unittest.TestCase):
    def test_continuous_noise_keeps_previous_direction(self):
        agent = _SingleAgentPPOTrainer(
            obs_dim=3,
            config=PPOConfig(),
            device=torch.device("cpu"),
        )
        previous_noise = torch.tensor([[1.0, -1.0]], dtype=torch.float32)
        agent._continuous_noise = previous_noise.clone()

        torch.manual_seed(0)
        noise = agent._next_exploration_noise(torch.Size((1, 2)))

        self.assertEqual(noise.shape, previous_noise.shape)
        self.assertGreater(float(torch.sum(noise * previous_noise)), 0.0)

    def test_reset_exploration_noise_clears_agent_state(self):
        trainer = PPOTrainer(
            obs_dim=3,
            config=PPOConfig(),
            device=torch.device("cpu"),
            agent_count=1,
        )
        trainer.agents[0]._continuous_noise = torch.ones((1, 2), dtype=torch.float32)

        trainer.reset_exploration_noise()

        self.assertIsNone(trainer.agents[0]._continuous_noise)

    def test_deterministic_actions_do_not_advance_exploration_noise(self):
        trainer = PPOTrainer(
            obs_dim=3,
            config=PPOConfig(),
            device=torch.device("cpu"),
            agent_count=1,
        )
        previous_noise = torch.tensor([[0.5, -0.5]], dtype=torch.float32)
        trainer.agents[0]._continuous_noise = previous_noise.clone()

        trainer.act(np.zeros((1, 3), dtype=np.float32), deterministic=True)

        torch.testing.assert_close(trainer.agents[0]._continuous_noise, previous_noise)


if __name__ == "__main__":
    unittest.main()
