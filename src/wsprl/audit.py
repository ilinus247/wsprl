"""Run the pre-Phase-5 audit locally without submitting any OpenAI requests."""
import argparse
import json
from pathlib import Path

import numpy as np
import torch

from wsprl.batch import completed_attempt
from wsprl.blinding_audit import audit_requests
from wsprl.diversity_audit import compare_banks, read_bank
from wsprl.integrity_audit import audit_cartpole
from wsprl.serialization import canonical_json, serialize_policy
from wsprl.training import TrainingConfig, load_checkpoint


def audit_pilot(batch: Path, original: Path, fresh: Path, output: Path) -> dict:
    protocols = [json.loads((bank / "protocol.json").read_text()) for bank in (original, fresh)]
    for key in ("matrices", "policies", "failure_fraction"):
        if protocols[0][key] != protocols[1][key]:
            raise ValueError(f"Audit banks differ in {key}")
    if any(protocol["cohort"] != "exploratory-pilot" for protocol in protocols):
        raise ValueError("Audit requires exploratory pilot data")

    checkpoints = {job: record["checkpoint_id"] for job, record in protocols[0]["policies"].items()}
    policies, cases, seeds, values = read_bank(original / "raw" / "episodes.csv", checkpoints)
    other_policies, other_cases, fresh_seeds, fresh_values = read_bank(
        fresh / "raw" / "episodes.csv", checkpoints,
    )
    if policies != other_policies or cases != other_cases or np.intersect1d(seeds, fresh_seeds).size:
        raise ValueError("Expected matching policies and cases with disjoint seed banks")
    for protocol, eval_seeds in zip(protocols, (seeds, fresh_seeds), strict=True):
        if protocol["evaluation_seeds"] != eval_seeds.tolist():
            raise ValueError("Observed seeds differ from the recorded protocol")

    screen = json.loads((batch / "nominal-screen.json").read_text())
    if set(screen["seeds"]) & (set(seeds) | set(fresh_seeds)):
        raise ValueError("An audit seed overlaps nominal screening")
    qualified = {
        entry["job"] for entry in screen["policies"] if entry["mean_return"] >= screen["minimum_mean"]
    }
    qualified_indices = [i for i, policy in enumerate(policies) if policy in qualified]
    diversity = {}
    for label, indices in [("all", list(range(len(policies)))), ("screen_passed", qualified_indices)]:
        for normalized in (True, False):
            scale = "normalized" if normalized else "return_points"
            diversity[f"{label}_{scale}"] = compare_banks(
                values[indices], fresh_values[indices], [policies[i] for i in indices],
                cases, normalized=normalized,
            )

    jobs = json.loads((batch / "manifest.json").read_text())["jobs"]
    if set(jobs) != set(policies):
        raise ValueError("Evaluation cohort differs from the training manifest")
    weights_dir = output / "weights"
    weights_dir.mkdir(parents=True)
    integrity = {}
    policy_weights = {}
    previous_threads = torch.get_num_threads()
    try:
        torch.set_num_threads(1)
        for job in policies:
            config = TrainingConfig.model_validate(jobs[job])
            run = completed_attempt(batch / job, config)
            expected = protocols[0]["policies"][job]
            if run is None or run.name != expected["checkpoint_id"] or config.model_dump() != expected["training"]:
                raise ValueError(f"Checkpoint for {job} differs from the evaluation protocol")
            if config.env_id != "CartPole-v1":
                raise ValueError("Observation stress bank is defined for the CartPole pilot")
            model = load_checkpoint(run)
            path = weights_dir / f"{run.name}.json"
            path.write_text(canonical_json(serialize_policy(model, config.env_id)) + "\n")
            exported = json.loads(path.read_text())
            policy_weights[run.name] = exported["parameters"]
            integrity[job] = audit_cartpole(model, exported)
            print(f"Verified inference for {job}", flush=True)
        blinding = audit_requests(batch, output, policy_weights)
    finally:
        torch.set_num_threads(previous_threads)
    report = {
        "first_bank_seeds": seeds.tolist(),
        "fresh_bank_seeds": fresh_seeds.tolist(),
        "diversity": diversity,
        "blinding": blinding,
        "weight_integrity": integrity,
    }
    (output / "audit.json").write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("batch", "original", "fresh", "output"):
        parser.add_argument(f"--{name}", type=Path, required=True)
    args = parser.parse_args()
    audit_pilot(args.batch, args.original, args.fresh, args.output)


if __name__ == "__main__":
    main()
