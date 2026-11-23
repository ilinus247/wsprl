"""Evaluate a frozen pilot cohort on a fixed matrix of interventions."""
import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path

import numpy as np
import torch

from wsprl.batch import completed_attempt
from wsprl.environment import EnvironmentId, Interventions, make_environment
from wsprl.training import TrainingConfig, load_checkpoint


# Exploratory levels fixed before collecting perturbed returns.
LEVELS = {
    "position_noise_std": (0.05, 0.20),
    "velocity_noise_std": (0.05, 0.20),
    "mask_probability": (0.10, 0.30),
    "gravity_scale": (1.25, 1.50),
    "pole_length_scale": (0.75, 0.50),
    "action_flip_probability": (0.05, 0.15),
}
SHARED_CASES = {"mask_probability", "action_flip_probability"}
PILOT_SEEDS = tuple(range(20_000, 20_024))
BOOTSTRAP_DRAWS = 2000
BOOTSTRAP_SEED = 41002


@dataclass(frozen=True)
class Case:
    intervention: str
    severity: float | None

    def settings(self) -> Interventions:
        if self.severity is None:
            return Interventions()
        return Interventions(**{self.intervention: self.severity})


def intervention_matrix(env_id: EnvironmentId) -> tuple[Case, ...]:
    return (Case("nominal", None),) + tuple(
        Case(name, level)
        for name, levels in LEVELS.items()
        if env_id == "CartPole-v1" or name in SHARED_CASES
        for level in levels
    )


def collect_episodes(model, env_id: EnvironmentId, case: Case, seeds: tuple[int, ...]):
    with make_environment(env_id, case.settings()) as env:
        for seed in seeds:
            observation, _ = env.reset(seed=seed)
            total = 0.0
            steps = 0
            terminated = truncated = False
            while not (terminated or truncated):
                action, _ = model.predict(observation, deterministic=True)
                observation, reward, terminated, truncated, _ = env.step(action)
                total += reward
                steps += 1
            yield {
                "eval_seed": seed, "return": total, "steps": steps,
                "terminated": terminated, "truncated": truncated,
            }


def summarize(nominal, perturbed, failure_fraction: float) -> dict:
    """Summarize matched episode returns and bootstrap over paired reset seeds."""
    baseline = np.asarray(nominal, dtype=float)
    returns = np.asarray(perturbed, dtype=float)
    if baseline.ndim != 1 or baseline.shape != returns.shape or len(baseline) < 2:
        raise ValueError("At least two paired episode returns are required")
    if not (np.isfinite(baseline).all() and np.isfinite(returns).all()):
        raise ValueError("Returns must be finite")
    if not 0 < failure_fraction < 1:
        raise ValueError("Failure fraction must be between 0 and 1")

    nominal_mean = baseline.mean()
    degradation = baseline - returns
    scale = abs(nominal_mean)
    failure_threshold = nominal_mean - failure_fraction * scale
    indices = np.random.default_rng(BOOTSTRAP_SEED).integers(
        len(returns), size=(BOOTSTRAP_DRAWS, len(returns))
    )
    lower, upper = np.quantile(degradation[indices].mean(axis=1), [0.025, 0.975])
    return {
        "episodes": len(returns),
        "nominal_mean": nominal_mean,
        "nominal_sd": baseline.std(ddof=1),
        "perturbed_mean": returns.mean(),
        "perturbed_sd": returns.std(ddof=1),
        "absolute_degradation": degradation.mean(),
        "normalized_degradation": degradation.mean() / scale if scale else None,
        "absolute_ci95_low": lower,
        "absolute_ci95_high": upper,
        "failure_threshold_return": failure_threshold,
        "failure_frequency": np.mean(returns < failure_threshold),
    }


def validate_seeds(seeds: tuple[int, ...], training_seeds: set[int], screen_seeds: set[int]):
    if len(seeds) < 2 or len(set(seeds)) != len(seeds):
        raise ValueError("Provide at least two distinct evaluation seeds")
    if any(not 20_000 <= seed < 30_000 for seed in seeds):
        raise ValueError("Pilot evaluation seeds must be in [20000, 30000)")
    if set(seeds) & (training_seeds | screen_seeds):
        raise ValueError("Evaluation seeds overlap training or nominal screening seeds")


def evaluate_pilot(
    batch_root: Path, output: Path, failure_fraction: float = 0.25,
    *, seeds: tuple[int, ...] | None = None,
):
    if not 0 < failure_fraction < 1:
        raise ValueError("Failure fraction must be between 0 and 1")
    manifest = json.loads((batch_root / "manifest.json").read_text())
    jobs = {job: TrainingConfig.model_validate(config) for job, config in manifest["jobs"].items()}
    if not jobs:
        raise ValueError("The cohort manifest has no jobs")
    screening = batch_root / "nominal-screen.json"
    screen_seeds = set(json.loads(screening.read_text())["seeds"]) if screening.exists() else set()
    seeds = PILOT_SEEDS if seeds is None else seeds
    validate_seeds(seeds, {config.seed for config in jobs.values()}, screen_seeds)

    attempts = {}
    for job, config in jobs.items():
        attempt = completed_attempt(batch_root / job, config)
        if attempt is None:
            raise ValueError(f"No verified checkpoint for {job}")
        attempts[job] = attempt

    matrices = {env_id: intervention_matrix(env_id) for env_id in sorted({c.env_id for c in jobs.values()})}
    (output / "raw").mkdir(parents=True, exist_ok=False)
    (output / "processed").mkdir()
    manifest = {
        "cohort": "exploratory-pilot",
        "batch": str(batch_root.resolve()),
        "evaluation_seeds": seeds,
        "failure_fraction": failure_fraction,
        "bootstrap_draws": BOOTSTRAP_DRAWS,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "matrices": {
            env: [{"intervention": case.intervention, "severity": case.severity,
                   "settings": case.settings().model_dump()} for case in cases]
            for env, cases in matrices.items()
        },
        "policies": {
            job: {"checkpoint_id": attempt.name, "training": jobs[job].model_dump()}
            for job, attempt in attempts.items()
        },
    }
    (output / "protocol.json").write_text(json.dumps(manifest, indent=2) + "\n")

    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        with (output / "raw" / "episodes.csv").open("x", newline="") as raw_file, (
            output / "processed" / "vulnerability.csv"
        ).open("x", newline="") as summary_file:
            common = ["job", "checkpoint_id", "environment", "training_seed", "intervention", "severity"]
            raw_writer = csv.DictWriter(
                raw_file, common + ["eval_seed", "return", "steps", "terminated", "truncated"]
            )
            raw_writer.writeheader()
            summary_writer = None
            for job, config in jobs.items():
                model = load_checkpoint(attempts[job])
                baseline = []
                for case in matrices[config.env_id]:
                    identity = {
                        "job": job, "checkpoint_id": attempts[job].name,
                        "environment": config.env_id, "training_seed": config.seed,
                        "intervention": case.intervention, "severity": case.severity,
                    }
                    episodes = list(collect_episodes(model, config.env_id, case, seeds))
                    raw_writer.writerows({**identity, **episode} for episode in episodes)
                    returns = [episode["return"] for episode in episodes]
                    if case.severity is None:
                        baseline = returns
                    summary = summarize(baseline, returns, failure_fraction)
                    if summary_writer is None:
                        summary_writer = csv.DictWriter(summary_file, common + list(summary))
                        summary_writer.writeheader()
                    summary_writer.writerow({**identity, **summary})
                raw_file.flush()
                summary_file.flush()
                print(f"Evaluated {job}", flush=True)
    finally:
        torch.set_num_threads(previous_threads)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch-root", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--failure-fraction", type=float, default=0.25)
    args = parser.parse_args()
    try:
        evaluate_pilot(args.batch_root, args.output, args.failure_fraction)
    except (ValueError, OSError) as error:
        parser.error(str(error))


if __name__ == "__main__":
    main()
