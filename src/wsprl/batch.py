"""Train and resume a grid of independently seeded policies."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
from typing import get_args

from pydantic import BaseModel, ConfigDict, Field, TypeAdapter

from wsprl.environment import EnvironmentId, Interventions
from wsprl.training import Algorithm, PolicyType, TrainingConfig, train, verify_checkpoint


class Condition(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str = Field(pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    environments: tuple[EnvironmentId, ...] = Field(
        default=get_args(EnvironmentId), min_length=1
    )
    interventions: Interventions = Field(default_factory=Interventions)


DEFAULT_CONDITIONS = (
    Condition(name="nominal"),
    Condition(name="masking", interventions=Interventions(mask_probability=0.1)),
    Condition(name="action-perturbation", interventions=Interventions(action_flip_probability=0.1)),
    Condition(name="position-noise", environments=("CartPole-v1",),
              interventions=Interventions(position_noise_std=0.1)),
    Condition(name="velocity-noise", environments=("CartPole-v1",),
              interventions=Interventions(velocity_noise_std=0.1)),
    Condition(name="gravity", environments=("CartPole-v1",),
              interventions=Interventions(gravity_scale=1.25)),
    Condition(name="pole-length", environments=("CartPole-v1",),
              interventions=Interventions(pole_length_scale=1.25)),
)


def build_plan(
    conditions: tuple[Condition, ...] = DEFAULT_CONDITIONS,
    seeds: tuple[int, ...] = (0, 1, 2),
    policies: tuple[PolicyType, ...] = ("tanh",),
    timesteps: int = 4096,
) -> dict[str, TrainingConfig]:
    if not conditions or not seeds or not policies:
        raise ValueError("Choose at least one condition, seed, and architecture")
    if len({condition.name for condition in conditions}) != len(conditions):
        raise ValueError("Condition names must be unique")
    if len(set(seeds)) != len(seeds) or len(set(policies)) != len(policies):
        raise ValueError("Seeds and architectures must be distinct")
    jobs = {}
    for condition in conditions:
        if len(set(condition.environments)) != len(condition.environments):
            raise ValueError(f"Duplicate environments in condition {condition.name}")
        for env_id in condition.environments:
            for algorithm in get_args(Algorithm):
                for policy in policies:
                    for seed in seeds:
                        job = f"{env_id}/{algorithm}/{policy}/{condition.name}/seed-{seed}"
                        jobs[job] = TrainingConfig(
                            seed=seed, timesteps=timesteps, env_id=env_id,
                            algorithm=algorithm, policy_type=policy,
                            interventions=condition.interventions,
                        )
    return jobs


def completed_attempt(job_dir: Path, config: TrainingConfig) -> Path | None:
    if not job_dir.exists():
        return None
    for attempt in sorted(job_dir.iterdir()):
        if not attempt.is_dir():
            continue
        try:
            metadata = verify_checkpoint(attempt)
        except (OSError, ValueError, KeyError, TypeError):
            continue
        if metadata.get("checkpoint_id") == attempt.name and metadata.get("training") == config.model_dump():
            return attempt
    return None


def run_batch(jobs: dict[str, TrainingConfig], output: Path) -> int:
    """Run pending jobs, preserving earlier attempts, and return the failure count."""
    plan = {"schema_version": 1, "jobs": {job: config.model_dump() for job, config in jobs.items()}}
    output.mkdir(parents=True, exist_ok=True)
    with (output / ".batch.lock").open("a") as lock:
        try:
            fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError(f"A runner is already using {output}") from error
        manifest = output / "manifest.json"
        if manifest.exists():
            if json.loads(manifest.read_text()) != plan:
                raise ValueError(f"Batch plan differs from {manifest}; choose a new output directory")
        else:
            pending = manifest.with_suffix(".tmp")
            pending.write_text(json.dumps(plan, indent=2) + "\n")
            pending.replace(manifest)

        failures = 0
        for job, config in jobs.items():
            job_dir = output / job
            existing = completed_attempt(job_dir, config)
            if existing is not None:
                print(f"Reused {existing}", flush=True)
                continue
            print(f"Training {job}", flush=True)
            try:
                attempt = train(config, job_dir)
            except Exception as error:
                failures += 1
                print(f"Failed {job}: {type(error).__name__}: {error}", file=sys.stderr, flush=True)
            else:
                print(f"Saved {attempt}", flush=True)
        return failures


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-root", type=Path, default=Path("runs/batch"))
    parser.add_argument("--conditions", type=Path, help="JSON array replacing the default conditions")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2])
    parser.add_argument("--policy", choices=get_args(PolicyType), action="append")
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--dry-run", action="store_true", help="Print the plan without training")
    args = parser.parse_args()
    try:
        conditions = DEFAULT_CONDITIONS
        if args.conditions:
            conditions = TypeAdapter(tuple[Condition, ...]).validate_json(args.conditions.read_bytes())
        jobs = build_plan(conditions, tuple(args.seeds), tuple(args.policy or ["tanh"]), args.timesteps)
        if args.dry_run:
            print(json.dumps({job: config.model_dump() for job, config in jobs.items()}, indent=2))
            print(f"{len(jobs)} planned runs", file=sys.stderr)
            return
        failures = run_batch(jobs, args.output_root)
    except (ValueError, OSError) as error:
        parser.error(str(error))
    print(f"{len(jobs) - failures}/{len(jobs)} jobs complete")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
