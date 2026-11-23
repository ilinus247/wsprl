"""Check every saved PPO policy for parameter, distribution, and episode equivalence."""
import argparse
import json
from pathlib import Path

import numpy as np
from stable_baselines3 import PPO
import torch
from torch import nn

from wsprl.environment import make_environment
from wsprl.evaluation import Case, intervention_matrix
from wsprl.integrity_audit import compare_inference, near_cartpole_boundaries
from wsprl.serialization import canonical_json, reconstruct, select_actions, serialize_policy
from wsprl.training import load_checkpoint


AUDIT_SEEDS = tuple(range(42_000, 42_005))


def compare_actor_parameters(model: PPO, export: dict, restored: nn.Sequential) -> dict:
    originals = {
        name: tensor.detach().cpu()
        for name, tensor in model.policy.named_parameters()
        if name.startswith(("mlp_extractor.policy_net.", "action_net."))
    }
    recovered = {}
    for definition, layer in zip(export["architecture"]["layers"], list(restored)[1:], strict=True):
        for field, tensor in layer.named_parameters():
            recovered[f"{definition['name']}.{field}"] = tensor.detach()
    if set(originals) != set(recovered):
        raise AssertionError("Actor parameter names are not identical")
    errors = []
    for name, tensor in originals.items():
        other = recovered[name]
        # Exact float32 bit patterns are expected after a lossless JSON round trip.
        if tensor.shape != other.shape or not torch.equal(tensor.view(torch.int32), other.view(torch.int32)):
            raise AssertionError(f"Actor parameter differs after reconstruction: {name}")
        errors.append((tensor.double() - other.double()).abs().flatten())
    differences = torch.cat(errors)
    return {
        "values": differences.numel(),
        "max_absolute_error": differences.max().item(),
        "mean_absolute_error": differences.mean().item(),
    }


def paired_episode(model, restored, env_id: str, case: Case, seed: int):
    """Use distinct environments with identical reset and intervention random streams."""
    settings = case.settings()
    with make_environment(env_id, settings) as original, make_environment(env_id, settings) as replica:
        obs, _ = original.reset(seed=seed)
        cloned_obs, _ = replica.reset(seed=seed)
        np.testing.assert_array_equal(cloned_obs, obs, err_msg="Initial observations differ")
        observations = [obs.copy()]
        original_return = reconstructed_return = 0.0
        max_observation_error = max_reward_error = 0.0
        for step in range(1, original.spec.max_episode_steps + 1):
            action, _ = model.predict(obs, deterministic=True)
            scores = restored(torch.as_tensor(cloned_obs).unsqueeze(0))
            cloned_action = select_actions(scores, "categorical_logits").item()
            if int(action) != cloned_action:
                raise AssertionError(f"Selected actions differ at step {step}: {int(action)} vs {cloned_action}")
            obs, reward, terminated, truncated, _ = original.step(action)
            cloned_obs, cloned_reward, cloned_terminated, cloned_truncated, _ = replica.step(cloned_action)
            np.testing.assert_array_equal(cloned_obs, obs, err_msg=f"Observations differ at step {step}")
            if reward != cloned_reward:
                raise AssertionError(f"Rewards differ at step {step}")
            if (terminated, truncated) != (cloned_terminated, cloned_truncated):
                raise AssertionError(f"Termination or truncation differs at step {step}")
            observations.append(obs.copy())
            max_observation_error = max(
                max_observation_error, float(np.max(np.abs(obs.astype(float) - cloned_obs)))
            )
            max_reward_error = max(max_reward_error, abs(reward - cloned_reward))
            original_return += reward
            reconstructed_return += cloned_reward
            if terminated or truncated:
                if original_return != reconstructed_return:
                    raise AssertionError("Episodic returns differ")
                return np.stack(observations), {
                    "intervention": case.intervention, "severity": case.severity, "seed": seed,
                    "steps": step, "terminated": terminated, "truncated": truncated,
                    "original_return": original_return, "reconstructed_return": reconstructed_return,
                    "max_observation_error": max_observation_error, "max_reward_error": max_reward_error,
                }
    raise AssertionError("Episode failed to finish by its environment time limit")


def audit_runs(runs: Path, output: Path):
    inventory = []
    for record in sorted(runs.rglob("metadata.json")):
        metadata = json.loads(record.read_text())
        if metadata.get("status") == "complete" and metadata.get("algorithm", "ppo") == "ppo":
            inventory.append((record.parent, metadata))
    if not inventory:
        raise ValueError(f"No completed PPO checkpoints in {runs}")

    output.mkdir(parents=True)
    weights_dir = output / "weights"
    weights_dir.mkdir()
    reports = []
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for run, metadata in inventory:
            env_id = metadata["environment"]
            model = load_checkpoint(run)
            if not isinstance(model, PPO):
                raise ValueError(f"Expected PPO at {run}")
            path = weights_dir / f"{run.name}.json"
            with path.open("x") as stream:
                stream.write(canonical_json(serialize_policy(model, env_id)) + "\n")
            export = json.loads(path.read_text())
            restored = reconstruct(export)
            parameters = compare_actor_parameters(model, export, restored)
            trajectories = []
            episodes = []
            with torch.inference_mode():
                for case in intervention_matrix(env_id):
                    for seed in AUDIT_SEEDS:
                        try:
                            states, episode = paired_episode(model, restored, env_id, case, seed)
                        except AssertionError as error:
                            raise AssertionError(
                                f"{run}: {case.intervention}={case.severity}, seed {seed}: {error}"
                            ) from error
                        trajectories.append(states)
                        episodes.append(episode)
                observations = np.concatenate(trajectories)
                test = compare_inference(model, export, observations, reconstructed=restored)
                if env_id == "CartPole-v1":
                    limits = np.array([4.8, 8.0, 0.42, 8.0])
                    rng = np.random.default_rng(97354)
                    synthetic = rng.uniform(-limits, limits, size=(4096, 4))
                    boundaries = near_cartpole_boundaries(model, synthetic)
                    boundary = compare_inference(model, export, boundaries, reconstructed=restored)
                else:
                    boundary = None
            reports.append({
                "checkpoint": str(run.relative_to(runs)),
                "environment": env_id,
                "parameters": parameters,
                "trajectory_inference": test,
                "boundary_inference": boundary,
                "episodes": episodes,
            })
            print(f"Verified {run.name}: {len(episodes)} episode pairs, {len(observations)} trajectory observations",
                  flush=True)
    except Exception as error:
        (output / "failure.json").write_text(json.dumps({
            "completed_policies": reports, "error": str(error),
        }, indent=2) + "\n")
        raise
    finally:
        torch.set_num_threads(previous_threads)

    report = {
        "audit_seeds": AUDIT_SEEDS,
        "episode_observations_include_terminal": True,
        "output_tolerance": {"atol": 1e-7, "rtol": 1e-6},
        "policies": reports,
    }
    (output / "equivalence.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runs", type=Path, default=Path("runs"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    audit_runs(args.runs, args.output)


if __name__ == "__main__":
    main()
