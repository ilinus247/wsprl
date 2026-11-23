"""Lossless JSON export of the network used for deterministic action selection."""
import argparse
import json
from pathlib import Path

from gymnasium.spaces import Box, Discrete
from stable_baselines3 import A2C, DQN, PPO
from stable_baselines3.common.distributions import CategoricalDistribution
from stable_baselines3.common.torch_layers import FlattenExtractor
import torch
from torch import nn

from wsprl.evaluation import intervention_matrix
from wsprl.training import load_checkpoint, verify_checkpoint


ENVIRONMENTS = {
    "CartPole-v1": {
        "task": "Balance the pole upright by pushing the cart. Higher return means longer balance.",
        "observations": [
            "cart position (m)", "cart velocity (m/s)",
            "pole angle from upright (rad)", "pole angular velocity (rad/s)",
        ],
        "actions": ["push left", "push right"],
    },
    "MountainCar-v0": {
        "task": "Build momentum to reach the flag on the right hill. Each step costs one reward point.",
        "observations": ["car position along the track", "car velocity"],
        "actions": ["accelerate left", "coast", "accelerate right"],
    },
    "Acrobot-v1": {
        "task": "Swing the tip of the two-link system above the target height. Higher return means reaching the target sooner.",
        "observations": [
            "cos(first joint angle)", "sin(first joint angle)",
            "cos(second joint angle)", "sin(second joint angle)",
            "first joint angular velocity (rad/s)", "second joint angular velocity (rad/s)",
        ],
        "actions": ["apply -1 torque", "apply zero torque", "apply +1 torque"],
    },
}
ACTIVATIONS = {"ReLU": nn.ReLU, "Tanh": nn.Tanh}
INTERVENTION_DESCRIPTIONS = {
    "position_noise_std": "Add independent zero-mean Gaussian noise to observed cart position, with this standard deviation in meters.",
    "velocity_noise_std": "Add independent zero-mean Gaussian noise to observed cart velocity, with this standard deviation in meters per second.",
    "mask_probability": "Independently replace each observed coordinate with zero with this probability.",
    "gravity_scale": "Multiply nominal gravity by this factor.",
    "pole_length_scale": "Multiply nominal pole half-length by this factor, holding pole mass fixed.",
    "action_flip_probability": "Replace the requested action with a uniformly sampled different action with this probability.",
}


def canonical_json(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False)


def serialize_policy(model, env_id: str) -> dict:
    description = ENVIRONMENTS[env_id]
    policy = model.policy
    observations, actions = model.observation_space, model.action_space
    if (
        not isinstance(observations, Box) or observations.shape != (len(description["observations"]),)
        or not isinstance(actions, Discrete) or actions.start != 0
        or actions.n != len(description["actions"])
    ):
        raise ValueError("Checkpoint spaces do not match the declared environment")

    if isinstance(model, DQN):
        extractor = policy.q_net.features_extractor
        modules = [(f"q_net.q_net.{name}", layer) for name, layer in policy.q_net.q_net.named_children()]
        output_type = "action_values"
    elif isinstance(model, (PPO, A2C)) and isinstance(policy.action_dist, CategoricalDistribution):
        extractor = policy.pi_features_extractor
        modules = [
            (f"mlp_extractor.policy_net.{name}", layer)
            for name, layer in policy.mlp_extractor.policy_net.named_children()
        ]
        modules.append(("action_net", policy.action_net))
        output_type = "categorical_logits"
    else:
        raise ValueError("Expected a supported categorical actor or DQN policy")

    if type(extractor) is not FlattenExtractor:
        raise ValueError("Only the standard SB3 vector flattening extractor is supported")

    layers = []
    parameters = {}
    for name, layer in modules:
        if type(layer) is nn.Linear:
            layers.append({
                "name": name, "type": "Linear",
                "in_features": layer.in_features, "out_features": layer.out_features,
            })
            for field in ("weight", "bias"):
                tensor = getattr(layer, field)
                if tensor is None or tensor.dtype != torch.float32 or not torch.isfinite(tensor).all():
                    raise ValueError(f"Expected finite float32 tensor for {name}.{field}")
                parameters[f"{name}.{field}"] = {
                    "shape": list(tensor.shape),
                    "values": tensor.detach().cpu().reshape(-1).tolist(),
                }
        elif type(layer) in ACTIVATIONS.values():
            layers.append({"name": name, "type": type(layer).__name__})
        else:
            raise ValueError(f"Unsupported inference layer: {type(layer).__name__}")

    return {
        "schema_version": 2,
        "architecture": {
            "environment": env_id,
            "observations": description["observations"].copy(),
            "actions": description["actions"].copy(),
            "input_preprocessing": "Cast each observation to float32 and flatten the observation dimensions in order, keeping the batch dimension. No normalization.",
            "output_type": output_type,
            "deterministic_action": (
                "Construct a float32 categorical distribution from the logits and choose the index "
                "of the largest probability, breaking ties by the lowest index."
                if output_type == "categorical_logits" else
                "Choose the index of the largest Q-value, breaking ties by the lowest index."
            ),
            "parameter_dtype": "float32",
            "tensor_order": "row-major",
            "layers": layers,
        },
        "parameters": parameters,
    }


class VectorInput(nn.Module):
    def forward(self, observation: torch.Tensor) -> torch.Tensor:
        return observation.float().flatten(start_dim=1)


def reconstruct(serialized: dict) -> nn.Sequential:
    """Reconstruct the batched observation-to-scores function on CPU."""
    architecture = serialized["architecture"]
    if serialized["schema_version"] not in (1, 2) or architecture["parameter_dtype"] != "float32":
        raise ValueError("Unsupported weight schema or dtype")
    modules = [VectorInput()]
    parameters = dict(serialized["parameters"])
    with torch.no_grad(), torch.random.fork_rng(devices=[]):
        for definition in architecture["layers"]:
            kind = definition["type"]
            if kind in ACTIVATIONS:
                modules.append(ACTIVATIONS[kind]())
                continue
            if kind != "Linear":
                raise ValueError(f"Unsupported layer: {kind}")
            layer = nn.Linear(definition["in_features"], definition["out_features"], dtype=torch.float32)
            for field, target in layer.named_parameters():
                name = f"{definition['name']}.{field}"
                parameter = parameters.pop(name, None)
                if parameter is None or parameter["shape"] != list(target.shape):
                    raise ValueError(f"Missing or mismatched parameter: {name}")
                values = torch.tensor(parameter["values"], dtype=torch.float32)
                if values.shape != (target.numel(),) or not torch.isfinite(values).all():
                    raise ValueError(f"Invalid parameter values: {name}")
                target.copy_(values.reshape_as(target))
            modules.append(layer)
    if parameters:
        raise ValueError("Unexpected parameters in weight export")
    return nn.Sequential(*modules).eval()


def action_probabilities(logits: torch.Tensor) -> torch.Tensor:
    return torch.distributions.Categorical(logits=logits).probs


def select_actions(scores: torch.Tensor, output_type: str) -> torch.Tensor:
    if output_type == "categorical_logits":
        return action_probabilities(scores).argmax(dim=1)
    if output_type == "action_values":
        return scores.argmax(dim=1)
    raise ValueError(f"Unsupported output type: {output_type}")


def prompt_payload(
    serialized: dict, *, include_weights: bool = True, failure_fraction: float = 0.25,
) -> dict:
    """Build a local message payload from the inference export and public task definitions."""
    architecture = serialized["architecture"]
    env_id = architecture["environment"]
    cases = [
        {"intervention": case.intervention, "severity": case.severity,
         "definition": INTERVENTION_DESCRIPTIONS[case.intervention]}
        for case in intervention_matrix(env_id)
        if case.severity is not None
    ]
    content = {
        "task": (
            "Predict the frozen policy's vulnerability to each listed intervention relative to the nominal environment. "
            "Predict normalized return degradation (nominal mean minus perturbed mean, divided by the magnitude "
            f"of the nominal mean), the probability an episode returns strictly below the nominal mean minus {100 * failure_fraction:.12g}% "
            "of its magnitude, and your confidence. Include a brief reason for each prediction and rank the "
            "settings from least harmful to most harmful. Use only the supplied information."
        ),
        "environment_task": ENVIRONMENTS[env_id]["task"],
        "architecture": architecture,
        "intervention_cases": cases,
    }
    if include_weights:
        content["parameters"] = serialized["parameters"]
    return {"messages": [{"role": "user", "content": canonical_json(content)}]}


def export_checkpoint(checkpoint: Path, output: Path) -> None:
    metadata = verify_checkpoint(checkpoint)
    serialized = serialize_policy(load_checkpoint(checkpoint), metadata["environment"])
    output.mkdir(parents=True)
    exports = {
        "weights.json": serialized,
        "prompt-with-weights.json": prompt_payload(serialized),
        "prompt-without-weights.json": prompt_payload(serialized, include_weights=False),
    }
    for filename, artifact in exports.items():
        (output / filename).write_text(canonical_json(artifact) + "\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True, help="A trusted local run directory")
    parser.add_argument("--output", type=Path, required=True, help="A new export directory")
    args = parser.parse_args()
    export_checkpoint(args.checkpoint, args.output)
    print(args.output)


if __name__ == "__main__":
    main()
