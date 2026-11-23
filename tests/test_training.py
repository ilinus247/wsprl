import json
from pathlib import Path
import sys

import numpy as np
import pytest
from stable_baselines3 import A2C, DQN, PPO
import torch

from wsprl.environment import Interventions, make_environment
from wsprl.integrity_audit import compare_inference
from wsprl.serialization import serialize_policy
from wsprl.training import TrainingConfig, load_checkpoint, main, train


def test_train_save_load_and_detect_corruption(tmp_path):
    config = TrainingConfig(seed=13, timesteps=129, interventions=Interventions(mask_probability=0.1))
    run = train(config, tmp_path)
    metadata = json.loads((run / "metadata.json").read_text())
    episodes = [json.loads(line) for line in (run / "training_episodes.jsonl").read_text().splitlines()]
    assert metadata["status"] == "complete"
    assert metadata["checkpoint_id"] == run.name
    assert metadata["training"] == config.model_dump()
    assert metadata["actual_timesteps"] == 256
    assert metadata["completed_training_episodes"] == len(episodes) > 0
    assert metadata["mean_training_return"] == np.mean([episode["return"] for episode in episodes])
    assert all(episode["return"] == episode["length"] for episode in episodes)

    model = load_checkpoint(run)
    assert isinstance(model, PPO)
    assert model.device.type == "cpu"
    assert model.num_timesteps == 256
    assert model.policy.action_net.in_features == 8
    assert model.policy.action_net.out_features == 2
    assert sum(parameter.numel() for parameter in model.policy.parameters()) == 107
    with make_environment("CartPole-v1") as env:
        observation, _ = env.reset(seed=99)
        assert env.observation_space == model.observation_space
        action, _ = model.predict(observation, deterministic=True)
        assert env.action_space.contains(action)
        batch, _ = model.policy.obs_to_tensor(observation)
        with torch.no_grad():
            assert int(action) == model.policy.get_distribution(batch).mode().item()

    metadata.pop("algorithm")
    metadata["training"].pop("algorithm")
    (run / "metadata.json").write_text(json.dumps(metadata))
    assert isinstance(load_checkpoint(run), PPO)

    with (run / "policy.zip").open("ab") as stream:
        stream.write(b"corruption")
    with pytest.raises(ValueError, match="checksum"):
        load_checkpoint(run)


def test_training_failure_retains_record(tmp_path, monkeypatch):
    def fail(*args, **kwargs):
        raise RuntimeError("synthetic environment failure")

    monkeypatch.setattr("wsprl.training.make_environment", fail)
    with pytest.raises(RuntimeError, match="synthetic environment failure"):
        train(TrainingConfig(seed=4), tmp_path)
    run, = tmp_path.iterdir()
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["status"] == "failed"
    assert metadata["error_type"] == "RuntimeError"
    assert metadata["training"]["seed"] == 4
    with pytest.raises(ValueError, match="Incomplete"):
        load_checkpoint(run)


@pytest.mark.parametrize("env_id,algorithm,model_class,policy,obs_size,hidden_sizes,activation,limit", [
    ("MountainCar-v0", "a2c", A2C, "relu", 2, [8], torch.nn.ReLU, 200),
    ("Acrobot-v1", "dqn", DQN, "deep-tanh", 6, [8, 8], torch.nn.Tanh, 500),
])
def test_cli_train_and_reload_new_variants(
    tmp_path, monkeypatch, capsys, env_id, algorithm, model_class, policy,
    obs_size, hidden_sizes, activation, limit,
):
    settings = tmp_path / "interventions.json"
    settings.write_text('{"mask_probability": 0.2, "action_flip_probability": 0.1}')
    monkeypatch.setattr(sys, "argv", [
        "training", "--env", env_id, "--algorithm", algorithm, "--policy", policy,
        "--seed", "42", "--timesteps", "128", "--output-root", str(tmp_path),
        "--interventions", str(settings),
    ])
    main()
    run = Path(capsys.readouterr().out.strip())
    model = load_checkpoint(run)
    assert isinstance(model, model_class)
    assert model._n_updates > 0
    metadata = json.loads((run / "metadata.json").read_text())
    assert metadata["environment"] == metadata["training"]["env_id"] == env_id
    assert metadata["training"]["policy_type"] == policy
    assert metadata["algorithm"] == metadata["training"]["algorithm"] == algorithm
    assert metadata["training"]["interventions"]["mask_probability"] == 0.2
    assert metadata["training"]["interventions"]["action_flip_probability"] == 0.1
    assert metadata["actual_timesteps"] == model.num_timesteps == 128
    if algorithm == "dqn":
        layers = {"q_hidden_sizes": hidden_sizes}
        networks = [model.policy.q_net.q_net[:-1], model.policy.q_net_target.q_net[:-1]]
        assert model.policy.q_net.q_net[-1].out_features == 3
    else:
        layers = {"actor_hidden_sizes": hidden_sizes, "critic_hidden_sizes": hidden_sizes}
        networks = [model.policy.mlp_extractor.policy_net, model.policy.mlp_extractor.value_net]
    assert metadata["architecture"] == {
        "policy": "MlpPolicy",
        **layers,
        "activation": activation.__name__,
        "observation_size": obs_size,
        "actions": 3,
    }
    for network in networks:
        assert len(network) == 2 * len(hidden_sizes)
        for index, (input_size, width) in enumerate(zip([obs_size, *hidden_sizes], hidden_sizes)):
            linear, nonlinearity = network[2 * index:2 * index + 2]
            assert (linear.in_features, linear.out_features) == (input_size, width)
            assert isinstance(nonlinearity, activation)
    with make_environment(env_id) as env:
        observation, _ = env.reset(seed=109)
        assert env.spec.id == env_id
        assert env.spec.max_episode_steps == limit
        assert env.observation_space == model.observation_space
        assert env.action_space == model.action_space
        action, _ = model.predict(observation, deterministic=True)
        if algorithm == "dqn":
            batch, _ = model.policy.obs_to_tensor(observation)
            with torch.no_grad():
                assert int(action) == model.policy.q_net(batch).argmax(dim=1).item()
        assert env.action_space.contains(action)
        successor = env.step(action)[0]
        assert env.observation_space.contains(successor)
        audit = compare_inference(
            model, serialize_policy(model, env_id), np.stack([observation, successor, -observation]),
        )
        assert audit["action_mismatches"] == 0


def test_new_environments_reject_cartpole_interventions():
    interventions = Interventions(pole_length_scale=2)
    with pytest.raises(ValueError, match="unavailable for MountainCar-v0"):
        TrainingConfig(seed=42, env_id="MountainCar-v0", interventions=interventions)
    with pytest.raises(ValueError, match="unavailable for Acrobot-v1"):
        make_environment("Acrobot-v1", interventions)
