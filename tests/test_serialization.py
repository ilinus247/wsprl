import json

import numpy as np
import pytest
from stable_baselines3 import A2C, DQN, PPO
import torch

from wsprl.environment import make_environment
from wsprl.integrity_audit import compare_inference
from wsprl.serialization import canonical_json, export_checkpoint, reconstruct, select_actions, serialize_policy
from wsprl.training import file_sha256


@pytest.mark.parametrize("algorithm,env_id,widths,activation", [
    (PPO, "CartPole-v1", [8], torch.nn.Tanh),
    (A2C, "MountainCar-v0", [8], torch.nn.ReLU),
    (DQN, "Acrobot-v1", [8, 8], torch.nn.Tanh),
])
def test_weights_roundtrip_preserves_scores_and_actions(algorithm, env_id, widths, activation):
    net_arch = widths if algorithm is DQN else {"pi": widths, "vf": widths}
    with make_environment(env_id) as env:
        model = algorithm(
            "MlpPolicy", env, seed=130, device="cpu",
            policy_kwargs={"net_arch": net_arch, "activation_fn": activation},
        )
        first, _ = env.reset(seed=20)
        second = env.step(1)[0]
        observations = np.stack([first, second, -first, np.zeros_like(first)]).astype(np.float64)
        export = serialize_policy(model, env_id)

        encoded = canonical_json(export)
        assert encoded == canonical_json(serialize_policy(model, env_id))
        restored = reconstruct(json.loads(encoded))
        actual_scores = restored(torch.from_numpy(observations))
        batch, _ = model.policy.obs_to_tensor(observations)
        with torch.no_grad():
            if algorithm is DQN:
                expected_scores = model.policy.q_net(batch)
                expected_names = {
                    f"q_net.{name}" for name, _ in model.policy.q_net.named_parameters()
                }
            else:
                features = model.policy.extract_features(batch)
                expected_scores = model.policy.action_net(model.policy.mlp_extractor.forward_actor(features))
                expected_names = {
                    f"mlp_extractor.policy_net.{name}"
                    for name, _ in model.policy.mlp_extractor.policy_net.named_parameters()
                } | {"action_net.weight", "action_net.bias"}
        assert set(export["parameters"]) == expected_names
        original = dict(model.policy.named_parameters())
        for name, parameter in export["parameters"].items():
            assert parameter["shape"] == list(original[name].shape)
            assert parameter["values"] == original[name].detach().flatten().tolist()
        torch.testing.assert_close(actual_scores, expected_scores, rtol=1e-6, atol=1e-7)
        expected_actions, _ = model.predict(observations, deterministic=True)
        actual_actions = select_actions(actual_scores, export["architecture"]["output_type"])
        np.testing.assert_array_equal(actual_actions.numpy(), expected_actions)


def test_categorical_rounding_can_change_action_from_raw_logit_argmax():
    with make_environment("CartPole-v1") as env:
        model = PPO("MlpPolicy", env, device="cpu")
        with torch.no_grad():
            model.policy.action_net.weight.zero_()
            model.policy.action_net.bias.copy_(torch.tensor([0.0, 1e-8]))
        observations = np.zeros((2, 4), dtype=np.float32)
        serialized = serialize_policy(model, "CartPole-v1")
        with torch.no_grad():
            scores = reconstruct(serialized)(torch.from_numpy(observations))
        assert scores.argmax(dim=1).tolist() == [1, 1]
        actual = select_actions(scores, serialized["architecture"]["output_type"])
        expected, _ = model.predict(observations, deterministic=True)
        np.testing.assert_array_equal(expected, [0, 0])
        np.testing.assert_array_equal(actual.numpy(), expected)


def test_payload_files_only_expose_allowlisted_inference_information(tmp_path):
    checkpoint = tmp_path / "PRIVATE_TRAINING_VARIANT"
    checkpoint.mkdir()
    with make_environment("CartPole-v1") as env:
        model = PPO("MlpPolicy", env, device="cpu")
        model.save(checkpoint / "policy.zip")
    record = {
        "environment": "CartPole-v1",
        "algorithm": "ppo",
        "status": "complete",
        "checkpoint_id": "PRIVATE_CHECKPOINT_ID",
        "checkpoint_sha256": file_sha256(checkpoint / "policy.zip"),
        "training": {"interventions": "PRIVATE_TRAINING_INTERVENTIONS", "seed": 987654321},
        "training_returns": "PRIVATE_TRAINING_RETURNS",
        "evaluation_profile": "PRIVATE_GROUND_TRUTH",
    }
    (checkpoint / "metadata.json").write_text(json.dumps(record))
    output = tmp_path / "exports"
    export_checkpoint(checkpoint, output)

    payloads = []
    for filename in ("prompt-with-weights.json", "prompt-without-weights.json"):
        raw = (output / filename).read_text()
        assert "PRIVATE_" not in raw
        assert "987654321" not in raw
        message, = json.loads(raw)["messages"]
        assert message["role"] == "user"
        payloads.append(json.loads(message["content"]))
    with_weights, without_weights = payloads
    assert set(without_weights) == {"task", "environment_task", "architecture", "intervention_cases"}
    assert len(without_weights["intervention_cases"]) == 12
    assert with_weights.pop("parameters") == json.loads((output / "weights.json").read_text())["parameters"]
    assert with_weights == without_weights


def test_reconstructor_rejects_truncated_tensor():
    with make_environment("CartPole-v1") as env:
        model = PPO("MlpPolicy", env, device="cpu")
        export = serialize_policy(model, "CartPole-v1")
    export["parameters"]["action_net.weight"]["values"].pop()
    with pytest.raises(ValueError, match="Invalid parameter values"):
        reconstruct(export)


def test_integrity_audit_detects_changed_value_even_with_valid_shape():
    with make_environment("CartPole-v1") as env:
        model = PPO("MlpPolicy", env, device="cpu")
        export = serialize_policy(model, "CartPole-v1")
        export["parameters"]["action_net.bias"]["values"][0] += 0.001
        with pytest.raises(AssertionError):
            compare_inference(model, export, np.zeros((2, 4), dtype=np.float32))
