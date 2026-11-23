import gymnasium as gym
import numpy as np
import pytest
from stable_baselines3 import PPO
import torch

from wsprl import behavioral_audit, integrity_audit
from wsprl.environment import make_environment
from wsprl.evaluation import Case
from wsprl.serialization import VectorInput, reconstruct, select_actions, serialize_policy


@pytest.fixture
def actor():
    with make_environment("CartPole-v1") as env:
        model = PPO(
            "MlpPolicy", env, seed=72, device="cpu",
            policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}},
        )
        yield model, serialize_policy(model, "CartPole-v1")


PROBES = np.array([
    [-3.0, 1.0, 0.17, -0.9],
    [2.5, -0.7, -0.21, 0.5],
    [0.1, 1.8, 0.02, 2.4],
], dtype=np.float64)


@pytest.mark.parametrize("defect", ["transpose", "reverse_order", "activation"])
def test_output_comparison_detects_defects_that_preserve_tensor_counts(actor, defect):
    model, exported = actor
    integrity_audit.compare_inference(model, exported, PROBES)
    if defect == "activation":
        activation, = [layer for layer in exported["architecture"]["layers"] if layer["type"] == "Tanh"]
        activation["type"] = "ReLU"
    else:
        tensor = exported["parameters"]["mlp_extractor.policy_net.0.weight"]
        values = np.asarray(tensor["values"]).reshape(tensor["shape"])
        corrupted = values.T.ravel() if defect == "transpose" else values.ravel()[::-1]
        tensor["values"] = corrupted.tolist()
    with pytest.raises(AssertionError, match="Tensor-likes are not close"):
        integrity_audit.compare_inference(model, exported, PROBES)


@pytest.mark.parametrize("name", [
    "mlp_extractor.policy_net.0.bias", "action_net.weight", "action_net.bias",
])
def test_missing_bias_or_action_head_cannot_reconstruct(actor, name):
    _, exported = actor
    del exported["parameters"][name]
    with pytest.raises(ValueError, match="Missing or mismatched parameter"):
        reconstruct(exported)


def test_omitted_preprocessing_breaks_float64_observations(actor, monkeypatch):
    model, exported = actor
    integrity_audit.compare_inference(model, exported, PROBES)
    monkeypatch.setattr(VectorInput, "forward", lambda _, inputs: inputs)
    with pytest.raises(RuntimeError, match="same dtype"):
        integrity_audit.compare_inference(model, exported, PROBES)


def test_rounded_weights_change_an_actual_policy_decision(actor):
    model, _ = actor
    with torch.no_grad():
        model.policy.action_net.weight.zero_()
        model.policy.action_net.bias.copy_(torch.tensor([0.0, 0.0004]))
    exported = serialize_policy(model, "CartPole-v1")
    expected, _ = model.predict(PROBES, deterministic=True)
    assert expected.tolist() == [1, 1, 1]
    for tensor in exported["parameters"].values():
        tensor["values"] = [round(value, 3) for value in tensor["values"]]
    with torch.no_grad():
        scores = reconstruct(exported)(torch.as_tensor(PROBES))
    assert select_actions(scores, "categorical_logits").tolist() == [0, 0, 0]
    with pytest.raises(AssertionError, match="Actor parameter differs"):
        behavioral_audit.compare_actor_parameters(model, exported, reconstruct(exported))


def test_probability_check_detects_wrong_distribution_with_same_logits_and_actions(actor, monkeypatch):
    model, exported = actor
    reference = integrity_audit.action_probabilities

    def wrong_distribution(scores):
        # Mix with a uniform distribution while preserving the probability ordering.
        return reference(scores) * 0.5 + 0.25

    with torch.no_grad():
        scores = reconstruct(exported)(torch.as_tensor(PROBES))
        assert torch.equal(reference(scores).argmax(1), wrong_distribution(scores).argmax(1))
    monkeypatch.setattr(integrity_audit, "action_probabilities", wrong_distribution)
    with pytest.raises(AssertionError, match="Tensor-likes are not close"):
        integrity_audit.compare_inference(model, exported, PROBES)


def test_complete_episode_matches_under_coupled_stochastic_observation_noise(actor):
    model, exported = actor
    case = Case("position_noise_std", 0.20)
    with torch.no_grad():
        trajectory, result = behavioral_audit.paired_episode(
            model, reconstruct(exported), "CartPole-v1", case, seed=42001,
        )
    assert len(trajectory) == result["steps"] + 1
    assert result["terminated"] or result["truncated"]
    assert result["original_return"] == result["reconstructed_return"] == result["steps"]
    assert result["max_observation_error"] == result["max_reward_error"] == 0
    with make_environment("CartPole-v1") as nominal:
        assert nominal.reset(seed=42001)[0][0] != trajectory[0, 0]


def test_complete_episode_detects_environment_drift_with_exact_network_outputs(actor, monkeypatch):
    model, exported = actor
    restored = reconstruct(exported)
    assert integrity_audit.compare_inference(model, exported, PROBES)["max_score_error"] == 0

    class DriftingObservation(gym.Wrapper):
        def step(self, action):
            observation, *outcome = self.env.step(action)
            observation = observation.copy()
            observation[0] += 0.001
            return observation, *outcome

    original, replica = [make_environment("CartPole-v1") for _ in range(2)]
    environments = iter((original, DriftingObservation(replica)))
    monkeypatch.setattr(behavioral_audit, "make_environment", lambda *args: next(environments))
    with torch.no_grad(), pytest.raises(AssertionError, match="Observations differ at step 1"):
        behavioral_audit.paired_episode(model, restored, "CartPole-v1", Case("nominal", None), 42001)
