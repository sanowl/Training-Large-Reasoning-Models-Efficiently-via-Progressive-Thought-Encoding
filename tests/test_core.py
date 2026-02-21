import unittest
import copy
from dataclasses import asdict
from pathlib import Path
import tempfile

import torch

from pte.cache import SlidingWindowCache
from pte.config import CacheConfig, ThoughtConfig
from pte.dynamic_lora import DynamicLoRAConfig, DynamicLoRALinear, dynamic_lora_signature
from pte.grpo import grpo_objective, normalize_group_rewards
from pte.hf_integration import load_pte_checkpoint
from pte.rewards import answers_equivalent, extract_final_answer, math_equivalence_reward
from pte.rollout import CacheAwareRolloutEngine
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

    def test_prune_preserves_layerwise_evicted_signals(self) -> None:
        cache = SlidingWindowCache(CacheConfig(window_size=1, evict_ratio=1.0, keep_prompt_tokens=True))

        # Two layers, one head, two tokens; prompt length zero means token-0 is evicted.
        key_l0 = torch.tensor([[[[1.0], [9.0]]]])  # [B=1, H=1, T=2, D=1]
        val_l0 = torch.tensor([[[[10.0], [90.0]]]])
        key_l1 = torch.tensor([[[[2.0], [8.0]]]])
        val_l1 = torch.tensor([[[[20.0], [80.0]]]])

        _, evicted = cache.prune(
            ((key_l0, val_l0), (key_l1, val_l1)),
            prompt_lengths=0,
        )
        self.assertIsNotNone(evicted)
        assert evicted is not None
        # New behavior: keep explicit layer axis [B, L, E, H] (no layer averaging).
        self.assertEqual(tuple(evicted.keys.shape), (1, 2, 1, 1))
        self.assertEqual(tuple(evicted.values.shape), (1, 2, 1, 1))
        self.assertEqual(evicted.num_evicted, 1)
        self.assertTrue(torch.allclose(evicted.keys[0, :, 0, 0], torch.tensor([1.0, 2.0])))
        self.assertTrue(torch.allclose(evicted.values[0, :, 0, 0], torch.tensor([10.0, 20.0])))


class TestThoughtState(unittest.TestCase):
    def test_state_init_and_update_shape(self) -> None:
        enc = ProgressiveThoughtEncoder(
            ThoughtConfig(hidden_size=8, rank=4, num_global_tokens=2, max_state_norm=4.0)
        )
        state = enc.init_state(batch_size=2)
        self.assertEqual(tuple(state.shape), (2, 4))

        ev_k = torch.randn(2, 3, 8)
        ev_v = torch.randn(2, 3, 8)
        updated = enc.update_state(state, ev_k, ev_v)
        self.assertEqual(tuple(updated.shape), (2, 4))
        norms = torch.norm(updated, dim=-1)
        self.assertTrue(torch.all(norms <= 4.0 + 1e-5))
        self.assertTrue(torch.all(norms > 0))

    def test_update_gate_dampens_contradictory_signal(self) -> None:
        enc = ProgressiveThoughtEncoder(
            ThoughtConfig(
                hidden_size=8,
                rank=4,
                num_global_tokens=2,
                min_update_gate=0.05,
                max_update_gate=0.30,
            )
        )
        state = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        aligned = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        opposite = torch.tensor([[-1.0, 0.0, 0.0, 0.0]])
        gate_aligned = enc._compute_update_gate(state, aligned).item()
        gate_opposite = enc._compute_update_gate(state, opposite).item()
        self.assertGreater(gate_aligned, gate_opposite)

    def test_update_gate_scales_with_evidence(self) -> None:
        enc = ProgressiveThoughtEncoder(
            ThoughtConfig(
                hidden_size=8,
                rank=4,
                num_global_tokens=2,
                min_update_gate=0.05,
                max_update_gate=0.30,
                evidence_tokens_scale=8.0,
            )
        )
        state = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        candidate = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        gate_low = enc._compute_update_gate(
            state,
            candidate,
            evicted_tokens_per_layer=1,
        ).item()
        gate_high = enc._compute_update_gate(
            state,
            candidate,
            evicted_tokens_per_layer=16,
        ).item()
        self.assertGreater(gate_high, gate_low)

    def test_update_gate_scales_with_token_confidence(self) -> None:
        enc = ProgressiveThoughtEncoder(
            ThoughtConfig(
                hidden_size=8,
                rank=4,
                num_global_tokens=2,
                min_update_gate=0.05,
                max_update_gate=0.30,
                use_token_confidence_gate=True,
                min_token_confidence=0.05,
            )
        )
        state = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        candidate = torch.tensor([[1.0, 0.0, 0.0, 0.0]])
        gate_low = enc._compute_update_gate(
            state,
            candidate,
            evicted_token_confidence=torch.tensor([0.1]),
        ).item()
        gate_high = enc._compute_update_gate(
            state,
            candidate,
            evicted_token_confidence=torch.tensor([0.9]),
        ).item()
        self.assertGreater(gate_high, gate_low)

    def test_layer_attention_accepts_layered_evicted_kv(self) -> None:
        enc = ProgressiveThoughtEncoder(
            ThoughtConfig(
                hidden_size=8,
                rank=4,
                num_global_tokens=2,
                use_layer_attention=True,
            )
        )
        state = enc.init_state(batch_size=2)
        ev_k = torch.randn(2, 3, 5, 8)  # [B, L, E, H]
        ev_v = torch.randn(2, 3, 5, 8)
        updated = enc.update_state(state, ev_k, ev_v)
        self.assertEqual(tuple(updated.shape), (2, 4))

    def test_magnitude_preserved_and_bounded(self) -> None:
        enc = ProgressiveThoughtEncoder(
            ThoughtConfig(
                hidden_size=8,
                rank=4,
                num_global_tokens=2,
                max_state_norm=2.0,
                min_update_gate=0.1,
                max_update_gate=0.3,
            )
        )
        state = torch.zeros(1, 4)
        # Stronger evidence should yield a larger state update magnitude.
        ev_k = torch.ones(1, 16, 8)
        ev_v = torch.ones(1, 16, 8)
        updated_low = enc.update_state(
            state,
            ev_k,
            ev_v,
            evicted_tokens_per_layer=1,
        )
        updated_high = enc.update_state(
            state,
            ev_k,
            ev_v,
            evicted_tokens_per_layer=16,
        )
        low_norm = float(torch.norm(updated_low, dim=-1).item())
        high_norm = float(torch.norm(updated_high, dim=-1).item())
        self.assertGreater(high_norm, low_norm)
        self.assertLessEqual(high_norm, 2.0 + 1e-5)

    def test_ood_confidence_drops_for_far_state(self) -> None:
        enc = ProgressiveThoughtEncoder(
            ThoughtConfig(
                hidden_size=8,
                rank=4,
                num_global_tokens=2,
                ood_guard_enabled=True,
                ood_threshold=1.0,
                ood_temperature=0.25,
                ood_min_confidence=0.2,
            )
        )
        with torch.no_grad():
            enc.state_mean.copy_(torch.zeros(4))
            enc.state_var.copy_(torch.ones(4))
            enc.state_stats_initialized.fill_(True)
        enc.eval()

        near = torch.zeros(1, 4)
        far = torch.full((1, 4), 8.0)
        conf_near = float(enc.adaptation_confidence(near).item())
        conf_far = float(enc.adaptation_confidence(far).item())
        self.assertGreater(conf_near, conf_far)
        self.assertGreaterEqual(conf_far, 0.2 - 1e-6)


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


class TestDynamicLoRA(unittest.TestCase):
    def test_outer_interaction_is_more_expressive_than_diagonal(self) -> None:
        thought = ProgressiveThoughtEncoder(ThoughtConfig(hidden_size=8, rank=2, num_global_tokens=2))
        base = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            base.weight.zero_()

        diagonal = DynamicLoRALinear(
            copy.deepcopy(base),
            thought,
            config=DynamicLoRAConfig(
                rank=2,
                alpha=2.0,
                interaction_mode="diagonal",
                outer_scale=1.0,
                freeze_base=False,
            ),
        )
        outer = DynamicLoRALinear(
            copy.deepcopy(base),
            thought,
            config=DynamicLoRAConfig(
                rank=2,
                alpha=2.0,
                interaction_mode="outer",
                outer_scale=1.0,
                freeze_base=False,
            ),
        )
        with torch.no_grad():
            eye = torch.eye(2)
            diagonal.lora_b.copy_(eye)
            diagonal.lora_a.copy_(eye)
            outer.lora_b.copy_(eye)
            outer.lora_a.copy_(eye)

        x = torch.tensor([[1.0, 2.0]])
        state = torch.tensor([[1.0, 1.0]])
        with thought.use_runtime_state(state):
            out_diagonal = diagonal(x)
            out_outer = outer(x)

        self.assertFalse(torch.allclose(out_diagonal, out_outer))
        self.assertTrue(torch.allclose(out_diagonal, torch.tensor([[1.0, 2.0]]), atol=1e-5))
        self.assertTrue(torch.allclose(out_outer, torch.tensor([[4.0, 5.0]]), atol=1e-5))

    def test_ood_confidence_scales_delta(self) -> None:
        thought = ProgressiveThoughtEncoder(
            ThoughtConfig(
                hidden_size=8,
                rank=2,
                num_global_tokens=2,
                ood_guard_enabled=True,
                ood_threshold=1.0,
                ood_temperature=0.25,
                ood_min_confidence=0.1,
            )
        )
        with torch.no_grad():
            thought.state_mean.copy_(torch.ones(2))
            thought.state_var.copy_(torch.full((2,), 0.01))
            thought.state_stats_initialized.fill_(True)
        thought.eval()

        base = torch.nn.Linear(2, 2, bias=False)
        with torch.no_grad():
            base.weight.zero_()
        layer = DynamicLoRALinear(
            base,
            thought,
            config=DynamicLoRAConfig(
                rank=2,
                alpha=2.0,
                interaction_mode="diagonal",
                outer_scale=0.0,
                freeze_base=False,
            ),
        )
        with torch.no_grad():
            eye = torch.eye(2)
            layer.lora_b.copy_(eye)
            layer.lora_a.copy_(eye)

        x = torch.tensor([[1.0, 1.0]])
        near_state = torch.tensor([[1.0, 1.0]])
        far_state = torch.tensor([[2.0, 2.0]])
        with thought.use_runtime_state(near_state):
            out_near = layer(x)
        with thought.use_runtime_state(far_state):
            out_far = layer(x)
        self.assertGreater(float(out_near.norm().item()), float(out_far.norm().item()))


class TestRolloutConfidenceBuffer(unittest.TestCase):
    def test_append_token_confidences_vectorized(self) -> None:
        conf_buffer = torch.zeros(3, 4, dtype=torch.float32)
        conf_tail = torch.tensor([0, 0, 1], dtype=torch.long)
        token_prob = torch.tensor([0.2, 0.3, 0.9], dtype=torch.float32)
        active_mask = torch.tensor([True, False, True], dtype=torch.bool)

        CacheAwareRolloutEngine._append_token_confidences(
            conf_buffer=conf_buffer,
            conf_tail=conf_tail,
            token_prob=token_prob,
            active_mask=active_mask,
        )
        self.assertTrue(torch.allclose(conf_tail, torch.tensor([1, 0, 2], dtype=torch.long)))
        self.assertAlmostEqual(float(conf_buffer[0, 0].item()), 0.2, places=6)
        self.assertAlmostEqual(float(conf_buffer[2, 1].item()), 0.9, places=6)

    def test_pop_evicted_confidence_handles_partial_availability(self) -> None:
        conf_buffer = torch.tensor(
            [
                [0.8, 0.6, 0.0],
                [0.3, 0.0, 0.0],
            ],
            dtype=torch.float32,
        )
        conf_head = torch.tensor([0, 0], dtype=torch.long)
        conf_tail = torch.tensor([2, 1], dtype=torch.long)
        mean_conf = CacheAwareRolloutEngine._pop_evicted_confidence(
            conf_buffer=conf_buffer,
            conf_head=conf_head,
            conf_tail=conf_tail,
            num_evicted=2,
            default_confidence=0.05,
        )
        self.assertTrue(torch.allclose(mean_conf, torch.tensor([0.7, 0.3], dtype=torch.float32), atol=1e-6))
        self.assertTrue(torch.allclose(conf_head, torch.tensor([2, 1], dtype=torch.long)))


class TestCheckpointRuntimeContract(unittest.TestCase):
    @staticmethod
    def _build_model(interaction_mode: str) -> tuple[torch.nn.Module, ProgressiveThoughtEncoder]:
        thought = ProgressiveThoughtEncoder(ThoughtConfig(hidden_size=8, rank=2, num_global_tokens=2))
        base = torch.nn.Linear(2, 2, bias=False)
        layer = DynamicLoRALinear(
            base,
            thought,
            config=DynamicLoRAConfig(
                rank=2,
                alpha=2.0,
                interaction_mode=interaction_mode,
                outer_scale=1.0,
                freeze_base=False,
            ),
        )
        model = torch.nn.Module()
        model.proj = layer
        return model, thought

    def test_runtime_contract_mismatch_is_rejected(self) -> None:
        model_train, thought_train = self._build_model("outer")
        payload = {
            "model": model_train.state_dict(),
            "thought_encoder": thought_train.state_dict(),
            "step": 3,
            "metadata": {
                "format_version": 2,
                "runtime_contract": {
                    "dynamic_lora": dynamic_lora_signature(model_train),
                    "thought_config": asdict(thought_train.config),
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "pte.pt"
            torch.save(payload, ckpt)
            model_runtime, thought_runtime = self._build_model("diagonal")
            with self.assertRaisesRegex(RuntimeError, "runtime contract mismatch"):
                load_pte_checkpoint(model_runtime, thought_runtime, str(ckpt))

    def test_runtime_contract_can_be_bypassed_with_allow_partial(self) -> None:
        model_train, thought_train = self._build_model("outer")
        payload = {
            "model": model_train.state_dict(),
            "thought_encoder": thought_train.state_dict(),
            "step": 7,
            "metadata": {
                "format_version": 2,
                "runtime_contract": {
                    "dynamic_lora": dynamic_lora_signature(model_train),
                    "thought_config": asdict(thought_train.config),
                },
            },
        }
        with tempfile.TemporaryDirectory() as tmpdir:
            ckpt = Path(tmpdir) / "pte.pt"
            torch.save(payload, ckpt)
            model_runtime, thought_runtime = self._build_model("diagonal")
            step = load_pte_checkpoint(
                model_runtime,
                thought_runtime,
                str(ckpt),
                allow_partial=True,
            )
            self.assertEqual(step, 7)


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
