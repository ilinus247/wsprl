import json

import pytest
from stable_baselines3 import DQN

from wsprl import batch
from wsprl.training import file_sha256, load_checkpoint


def test_default_plan_covers_each_intervention_and_replicates():
    jobs = batch.build_plan()
    conditions = {
        "CartPole-v1": {
            "nominal", "masking", "action-perturbation", "position-noise",
            "velocity-noise", "gravity", "pole-length",
        },
        "MountainCar-v0": {"nominal", "masking", "action-perturbation"},
        "Acrobot-v1": {"nominal", "masking", "action-perturbation"},
    }
    expected = {
        f"{env}/{algorithm}/tanh/{condition}/seed-{seed}"
        for env, names in conditions.items()
        for algorithm in ("ppo", "a2c", "dqn")
        for condition in names
        for seed in (0, 1, 2)
    }
    assert set(jobs) == expected
    assert len(jobs) == 117
    for key, config in jobs.items():
        env, algorithm, policy, condition, seed = key.split("/")
        assert (config.env_id, config.algorithm, config.policy_type, config.seed) == (
            env, algorithm, policy, int(seed.removeprefix("seed-"))
        )
        changed_fields = config.interventions.model_dump(exclude_defaults=True)
        expected_fields = {
            "nominal": {},
            "masking": {"mask_probability": 0.1},
            "action-perturbation": {"action_flip_probability": 0.1},
            "position-noise": {"position_noise_std": 0.1},
            "velocity-noise": {"velocity_noise_std": 0.1},
            "gravity": {"gravity_scale": 1.25},
            "pole-length": {"pole_length_scale": 1.25},
        }
        assert changed_fields == expected_fields[condition]


def test_batch_retries_failed_and_corrupt_attempts(tmp_path, monkeypatch):
    condition = batch.Condition(name="nominal", environments=("CartPole-v1",))
    jobs = batch.build_plan((condition,), seeds=(7,), timesteps=128)
    first, second, third = jobs
    calls = []

    def fake_train(config, job_dir):
        calls.append(config.algorithm)
        attempt = job_dir / f"attempt-{len(calls)}"
        attempt.mkdir(parents=True)
        failed = len(calls) == 1
        record = {
            "checkpoint_id": attempt.name,
            "status": "failed" if failed else "complete",
            "training": config.model_dump(),
        }
        if not failed:
            checkpoint = attempt / "policy.zip"
            checkpoint.write_bytes(b"fixture checkpoint")
            record["checkpoint_sha256"] = file_sha256(checkpoint)
        (attempt / "metadata.json").write_text(json.dumps(record))
        if failed:
            raise RuntimeError("fixture training failure")
        return attempt

    monkeypatch.setattr(batch, "train", fake_train)
    assert batch.run_batch(jobs, tmp_path) == 1
    assert calls == ["ppo", "a2c", "dqn"]
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert manifest["jobs"] == {job: config.model_dump() for job, config in jobs.items()}

    corrupt = tmp_path / second / "attempt-2" / "policy.zip"
    corrupt.write_bytes(b"corrupt")
    assert batch.run_batch(jobs, tmp_path) == 0
    assert calls == ["ppo", "a2c", "dqn", "ppo", "a2c"]
    assert (tmp_path / first / "attempt-1" / "metadata.json").exists()
    assert corrupt.read_bytes() == b"corrupt"
    assert batch.completed_attempt(tmp_path / third, jobs[third]).name == "attempt-3"
    assert batch.run_batch(jobs, tmp_path) == 0
    assert len(calls) == 5
    with pytest.raises(ValueError, match="Batch plan differs"):
        batch.run_batch(batch.build_plan((condition,), seeds=(8,)), tmp_path)
    assert len(calls) == 5


def test_batch_saves_real_checkpoint(tmp_path):
    jobs = batch.build_plan(seeds=(7,), timesteps=128)
    key = "Acrobot-v1/dqn/tanh/masking/seed-7"
    config = jobs[key]
    assert batch.run_batch({key: config}, tmp_path) == 0
    attempt = batch.completed_attempt(tmp_path / key, config)
    assert attempt is not None
    model = load_checkpoint(attempt)
    assert isinstance(model, DQN)
    assert model.num_timesteps == 128
    assert model._n_updates > 0
