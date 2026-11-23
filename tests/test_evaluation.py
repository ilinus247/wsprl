import csv
import json

import pytest

from wsprl import evaluation
from wsprl.training import TrainingConfig


def test_summary_reports_paired_degradation_and_strict_failure_boundary():
    result = evaluation.summarize([90, 110, 100, 100], [70, 75, 100, 105], 0.25)
    assert result["nominal_mean"] == 100
    assert result["nominal_sd"] == pytest.approx((200 / 3) ** 0.5)
    assert result["perturbed_mean"] == 87.5
    assert result["absolute_degradation"] == 12.5
    assert result["normalized_degradation"] == 0.125
    assert result["failure_threshold_return"] == 75
    assert result["failure_frequency"] == 0.25
    assert result["absolute_ci95_low"] < 12.5 < result["absolute_ci95_high"]
    constant_drop = evaluation.summarize([90, 110], [80, 100], 0.25)
    assert constant_drop["absolute_ci95_low"] == constant_drop["absolute_ci95_high"] == 10


def test_negative_and_zero_baselines_have_explicit_behavior():
    negative = evaluation.summarize([-100, -100], [-130, -120], 0.25)
    assert negative["normalized_degradation"] == 0.25
    assert negative["failure_threshold_return"] == -125
    assert negative["failure_frequency"] == 0.5
    zero = evaluation.summarize([-1, 1], [-2, -2], 0.25)
    assert zero["absolute_degradation"] == 2
    assert zero["normalized_degradation"] is None
    assert zero["failure_frequency"] == 1


def test_seed_validation_rejects_contamination_and_duplicate_resets():
    with pytest.raises(ValueError, match="overlap"):
        evaluation.validate_seeds((20000, 20001), {20001}, set())
    with pytest.raises(ValueError, match="overlap"):
        evaluation.validate_seeds((20000, 20001), set(), {20001})
    with pytest.raises(ValueError, match="distinct"):
        evaluation.validate_seeds((20000, 20000), set(), set())
    with pytest.raises(ValueError, match="Pilot evaluation seeds"):
        evaluation.validate_seeds((30000, 30001), set(), set())


def test_matrix_uses_individual_interventions_supported_by_each_environment():
    for env, size in [("CartPole-v1", 13), ("MountainCar-v0", 5), ("Acrobot-v1", 5)]:
        cases = evaluation.intervention_matrix(env)
        assert len(cases) == len(set(cases)) == size
        assert cases[0].intervention == "nominal"
        for case in cases[1:]:
            settings = case.settings()
            settings.validate_for(env)
            assert settings.model_dump(exclude_defaults=True) == {case.intervention: case.severity}


def test_pilot_writes_paired_episodes_from_frozen_policy(tmp_path, monkeypatch):
    class FixedPolicy:
        def predict(self, observation, *, deterministic):
            assert deterministic
            return int(observation[2] > 0), None

    batch_root = tmp_path / "batch"
    batch_root.mkdir()
    config = TrainingConfig(seed=1, interventions={"mask_probability": 1})
    job = "CartPole-v1/ppo/tanh/full-mask/seed-1"
    (batch_root / "manifest.json").write_text(json.dumps({"jobs": {job: config.model_dump()}}))
    monkeypatch.setattr(evaluation, "PILOT_SEEDS", (20000, 20001))
    monkeypatch.setattr(evaluation, "completed_attempt", lambda *args: batch_root / "checkpoint")
    monkeypatch.setattr(evaluation, "load_checkpoint", lambda *args: FixedPolicy())

    first = tmp_path / "first"
    second = tmp_path / "second"
    evaluation.evaluate_pilot(batch_root, first)
    evaluation.evaluate_pilot(batch_root, second)
    path = "raw/episodes.csv"
    assert (first / path).read_bytes() == (second / path).read_bytes()
    with (first / path).open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    assert len(rows) == 26
    assert all(row["checkpoint_id"] == "checkpoint" for row in rows)
    assert [int(row["eval_seed"]) for row in rows] == [20000, 20001] * 13
    baseline = list(evaluation.collect_episodes(
        FixedPolicy(), "CartPole-v1", evaluation.Case("nominal", None), (20000, 20001)
    ))
    assert [float(row["return"]) for row in rows[:2]] == [episode["return"] for episode in baseline]
    with (first / "processed/vulnerability.csv").open(newline="") as stream:
        summary = list(csv.DictReader(stream))
    assert len(summary) == 13
    assert float(summary[0]["absolute_degradation"]) == 0
    protocol = json.loads((first / "protocol.json").read_text())
    assert protocol["cohort"] == "exploratory-pilot"
    assert protocol["matrices"]["CartPole-v1"][0]["settings"]["mask_probability"] == 0
