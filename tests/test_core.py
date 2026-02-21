import unittest

import torch

from pte.cache import SlidingWindowCache
from pte.config import CacheConfig, ThoughtConfig
from pte.grpo import grpo_objective, normalize_group_rewards
from pte.rewards import answers_equivalent, extract_final_answer, math_equivalence_reward
from pte.thought_state import ProgressiveThoughtEncoder


class TestCache(unittest.TestCase):
    def test_prune_keeps_prompt_prefix(self) -> None:
        cache = SlidingWindowCache(CacheConfig(window_size=4, evict_ratio=0.25, keep_prompt_tokens=True))
        key = torch.randn(1, 2, 10, 4)
        value = torch.randn(1, 2, 10, 4)
        pruned, evicted = cache.prune(((key, value),), prompt_lengths=3)
        self.assertIsNotNone(evicted)
        self.assertGreater(evicted.num_evicted, 0)
        self.assertTrue(torch.allclose(pruned[0][0][:, :, :3, :], key[:, :, :3, :]))


class TestThoughtState(unittest.TestCase):
    def test_state_init_and_update_shape(self) -> None:
        enc = ProgressiveThoughtEncoder(ThoughtConfig(hidden_size=8, rank=4, num_global_tokens=2))
        state = enc.init_state(batch_size=2)
        self.assertEqual(tuple(state.shape), (2, 4))

        ev_k = torch.randn(2, 3, 8)
        ev_v = torch.randn(2, 3, 8)
        updated = enc.update_state(state, ev_k, ev_v)
        self.assertEqual(tuple(updated.shape), (2, 4))
        norms = torch.norm(updated, dim=-1)
        self.assertTrue(torch.allclose(norms, torch.ones_like(norms), atol=1e-4))


class TestGRPO(unittest.TestCase):
    def test_reward_normalization_and_objective(self) -> None:
        scores = torch.tensor([[0.0, 1.0, 2.0, 3.0]], dtype=torch.float32)
        rewards = normalize_group_rewards(scores)
        self.assertEqual(tuple(rewards.shape), (1, 4))

        logp = torch.tensor([0.0, -1.0, -0.5, -0.2])
        logp_ref = torch.tensor([-0.1, -1.2, -0.7, -0.3])
        loss, metrics = grpo_objective(logp, logp_ref, rewards.squeeze(0), beta_kl=0.02)
        self.assertTrue(torch.isfinite(loss).item())
        self.assertIsInstance(metrics.loss, float)


class TestRewards(unittest.TestCase):
    def test_extract_final_answer_nested_boxed(self) -> None:
        text = "Reasoning...\n\\boxed{\\frac{3}{4}}"
        self.assertEqual(extract_final_answer(text), "\\frac{3}{4}")

    def test_numeric_equivalence_fraction_decimal(self) -> None:
        self.assertTrue(answers_equivalent("\\boxed{\\frac{1}{2}}", "0.5"))

    def test_numeric_equivalence_percent(self) -> None:
        rewards = math_equivalence_reward(["Answer: 50%"], ["0.5"])
        self.assertEqual(float(rewards.item()), 1.0)

    def test_equation_side_extraction(self) -> None:
        self.assertTrue(answers_equivalent("x = 12", "12"))


if __name__ == "__main__":
    unittest.main()
