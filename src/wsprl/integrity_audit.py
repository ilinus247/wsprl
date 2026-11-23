"""Stress-test exported inference networks on real and constructed observations."""
import json
from itertools import product

import numpy as np
from stable_baselines3 import DQN
import torch

from wsprl.environment import make_environment
from wsprl.evaluation import intervention_matrix
from wsprl.serialization import action_probabilities, canonical_json, reconstruct, select_actions


def original_scores(model, observations: np.ndarray) -> torch.Tensor:
    batch, _ = model.policy.obs_to_tensor(observations)
    if isinstance(model, DQN):
        return model.policy.q_net(batch)
    features = model.policy.extract_features(batch)
    return model.policy.action_net(model.policy.mlp_extractor.forward_actor(features))


def compare_inference(model, serialized: dict, observations: np.ndarray, *, reconstructed=None) -> dict:
    if reconstructed is None:
        reconstructed = reconstruct(json.loads(canonical_json(serialized)))
    output_type = serialized["architecture"]["output_type"]
    with torch.no_grad():
        original = original_scores(model, observations)
        restored = reconstructed(torch.as_tensor(observations))
        reference_actions, _ = model.predict(observations, deterministic=True)
        predicted = select_actions(restored, output_type).numpy()
        torch.testing.assert_close(restored, original, atol=1e-7, rtol=1e-6)
        np.testing.assert_array_equal(predicted, reference_actions)
        score_gaps = original.topk(2, dim=1).values.diff(dim=1).abs()
        score_error = (restored.double() - original.double()).abs()
        result = {
            "observations": len(observations),
            "max_score_error": score_error.max().item(),
            "mean_score_error": score_error.mean().item(),
            "action_mismatches": int(np.count_nonzero(predicted != reference_actions)),
            "raw_argmax_action_mismatches": int(np.count_nonzero(
                restored.argmax(dim=1).numpy() != reference_actions
            )),
            "top_score_gaps_le_1e-6": int(torch.count_nonzero(score_gaps <= 1e-6)),
        }
        if output_type == "categorical_logits":
            original_dist = model.policy.get_distribution(model.policy.obs_to_tensor(observations)[0])
            probs = action_probabilities(restored)
            torch.testing.assert_close(probs, original_dist.distribution.probs, atol=1e-7, rtol=1e-6)
            error = (probs.double() - original_dist.distribution.probs.double()).abs()
            result["max_probability_error"] = error.max().item()
            result["mean_probability_error"] = error.mean().item()
    return result


def rollout_observations(model, env_id: str) -> np.ndarray:
    observations = []
    for case in intervention_matrix(env_id):
        with make_environment(env_id, case.settings()) as env:
            for seed in range(22000, 22004):
                observation, _ = env.reset(seed=seed)
                observations.append(observation)
                for _ in range(64):
                    action, _ = model.predict(observation, deterministic=True)
                    observation, _, terminated, truncated, _ = env.step(action)
                    observations.append(observation)
                    if terminated or truncated:
                        break
    return np.stack(observations)


def near_cartpole_boundaries(model, synthetic: np.ndarray) -> np.ndarray:
    """Bisect segments between contrasting actions, keeping both float32 endpoints."""
    with torch.no_grad():
        scores = original_scores(model, synthetic.astype(np.float32))
        mask = (scores[:, 1] > scores[:, 0]).numpy()
        count = min(256, np.count_nonzero(mask), np.count_nonzero(~mask))
        if not count:
            raise ValueError("Boundary audit needs a policy that uses both CartPole actions")
        low = synthetic[~mask][:count].astype(np.float32)
        high = synthetic[mask][:count].astype(np.float32)
        for _ in range(28):
            middle = (low + high) * np.float32(0.5)
            values = original_scores(model, middle)
            upper = (values[:, 1] > values[:, 0]).numpy()
            high[upper] = middle[upper]
            low[~upper] = middle[~upper]
    return np.concatenate((low, high))


def audit_cartpole(model, export: dict) -> dict:
    limits = np.array([4.8, 8.0, 0.42, 8.0])
    rng = np.random.default_rng(97354)
    synthetic = rng.uniform(-limits, limits, size=(4096, 4))
    corners = np.asarray(list(product([-1, 1], repeat=4))) * limits
    masked = np.concatenate([
        synthetic[:128] * mask for mask in product([0, 1], repeat=4)
    ])
    banks = {
        "rollout": rollout_observations(model, "CartPole-v1"),
        "uniform_and_corners": np.concatenate((synthetic, corners)),
        "masked": masked,
        "near_boundary": near_cartpole_boundaries(model, synthetic),
    }
    return {name: compare_inference(model, export, values) for name, values in banks.items()}
