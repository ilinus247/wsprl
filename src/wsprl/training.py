"""Train a small CPU RL policy and retain its checkpoint and training record."""
import argparse
import hashlib
from importlib.metadata import version
import json
from pathlib import Path
import platform
import sys
from typing import Literal, get_args
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.base_class import BaseAlgorithm
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.monitor import Monitor
import torch

from wsprl.environment import EnvironmentId, Interventions, make_environment


PolicyType = Literal["tanh", "relu", "deep-tanh"]
Algorithm = Literal["ppo", "a2c", "dqn"]
ALGORITHMS = {
    "ppo": (PPO, {"n_steps": 128, "batch_size": 32}),
    "a2c": (A2C, {"n_steps": 128}),
    "dqn": (DQN, {"learning_starts": 100, "batch_size": 32, "train_freq": 4}),
}
POLICY_ARCHITECTURES = {
    "tanh": ((8,), torch.nn.Tanh),
    "relu": ((8,), torch.nn.ReLU),
    "deep-tanh": ((8, 8), torch.nn.Tanh),
}


class TrainingConfig(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")

    seed: int = Field(ge=0, le=2**32 - 1)
    timesteps: int = Field(default=4096, gt=0)
    env_id: EnvironmentId = "CartPole-v1"
    algorithm: Algorithm = "ppo"
    policy_type: PolicyType = "tanh"
    interventions: Interventions = Field(default_factory=Interventions)

    @model_validator(mode="after")
    def validate_interventions(self):
        self.interventions.validate_for(self.env_id)
        return self


class EpisodeLog(BaseCallback):
    def __init__(self, stream):
        super().__init__()
        self.stream = stream
        self.returns = []

    def _on_step(self) -> bool:
        for info in self.locals["infos"]:
            if "episode" in info:
                episode = info["episode"]
                self.returns.append(episode["r"])
                self.stream.write(json.dumps({
                    "timestep": self.num_timesteps,
                    "return": episode["r"],
                    "length": episode["l"],
                }) + "\n")
                self.stream.flush()
        return True


def file_sha256(path: Path) -> str:
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def train(config: TrainingConfig, output_root: Path) -> Path:
    algorithm, options = ALGORITHMS[config.algorithm]
    hidden_sizes, activation = POLICY_ARCHITECTURES[config.policy_type]
    widths = list(hidden_sizes)
    architecture = {"policy": "MlpPolicy", "activation": activation.__name__}
    if config.algorithm == "dqn":
        net_arch = widths
        architecture["q_hidden_sizes"] = widths
    else:
        net_arch = {"pi": widths, "vf": widths}
        architecture.update(actor_hidden_sizes=widths, critic_hidden_sizes=widths)
    checkpoint_id = str(uuid4())
    run = output_root / checkpoint_id
    run.mkdir(parents=True, exist_ok=False)
    metadata = {
        "checkpoint_id": checkpoint_id,
        "status": "running",
        "environment": config.env_id,
        "algorithm": config.algorithm,
        "training": config.model_dump(),
        "architecture": architecture,
        "algorithm_options": {**options, "device": "cpu", "torch_threads": 1},
        "runtime": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "dependencies": {
                name: version(name) for name in (
                    "gymnasium", "stable-baselines3", "torch", "numpy", "pandas",
                    "matplotlib", "pydantic",
                )
            },
        },
    }
    record = run / "metadata.json"

    def write_metadata():
        temporary = record.with_suffix(".tmp")
        temporary.write_text(json.dumps(metadata, indent=2) + "\n")
        temporary.replace(record)

    write_metadata()
    env = None
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        env = Monitor(make_environment(config.env_id, config.interventions))
        metadata["architecture"].update(
            observation_size=env.observation_space.shape[0],
            actions=int(env.action_space.n),
        )
        write_metadata()
        model = algorithm(
            "MlpPolicy",
            env,
            seed=config.seed,
            device="cpu",
            policy_kwargs={
                "net_arch": net_arch,
                "activation_fn": activation,
            },
            **options,
        )
        with (run / "training_episodes.jsonl").open("w") as stream:
            log = EpisodeLog(stream)
            model.learn(total_timesteps=config.timesteps, callback=log)
        model.save(run / "policy.tmp.zip")
        (run / "policy.tmp.zip").replace(run / "policy.zip")
        metadata.update(
            status="complete",
            actual_timesteps=model.num_timesteps,
            checkpoint_sha256=file_sha256(run / "policy.zip"),
            completed_training_episodes=len(log.returns),
            mean_training_return=(
                sum(log.returns) / len(log.returns) if log.returns else None
            ),
        )
    except (Exception, KeyboardInterrupt) as error:
        metadata.update(status="failed", error_type=type(error).__name__, error=str(error))
        print(f"Training run record: {run}", file=sys.stderr)
        raise
    finally:
        if env is not None:
            env.close()
        torch.set_num_threads(previous_threads)
        write_metadata()
    return run


def verify_checkpoint(run: Path) -> dict:
    """Read metadata and verify the associated checkpoint without deserializing it."""
    metadata = json.loads((run / "metadata.json").read_text())
    checkpoint = run / "policy.zip"
    if metadata["status"] != "complete" or file_sha256(checkpoint) != metadata["checkpoint_sha256"]:
        raise ValueError("Incomplete run or checkpoint checksum mismatch")
    return metadata


def load_checkpoint(run: Path) -> BaseAlgorithm:
    """Load a trusted local SB3 archive after checking its recorded digest."""
    metadata = verify_checkpoint(run)
    # Checkpoints written before algorithm selection were all PPO.
    name = metadata.get("algorithm", "ppo")
    if name not in ALGORITHMS:
        raise ValueError(f"Unsupported checkpoint algorithm: {name!r}")
    algorithm, _ = ALGORITHMS[name]
    return algorithm.load(run / "policy.zip", device="cpu")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--timesteps", type=int, default=4096)
    parser.add_argument("--env", choices=get_args(EnvironmentId), default="CartPole-v1")
    parser.add_argument("--algorithm", choices=get_args(Algorithm), default="ppo")
    parser.add_argument("--policy", choices=get_args(PolicyType), default="tanh")
    parser.add_argument("--interventions", type=Path, help="JSON file of intervention settings")
    parser.add_argument("--output-root", type=Path, default=Path("runs"))
    args = parser.parse_args()
    interventions = (
        Interventions.model_validate_json(args.interventions.read_bytes())
        if args.interventions else Interventions()
    )
    config = TrainingConfig(
        seed=args.seed,
        timesteps=args.timesteps,
        env_id=args.env,
        algorithm=args.algorithm,
        policy_type=args.policy,
        interventions=interventions,
    )
    run = train(config, args.output_root)
    print(run)


if __name__ == "__main__":
    main()
