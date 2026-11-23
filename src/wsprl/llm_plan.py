"""Build blind, reproducible LLM trials from verified policy checkpoints."""
from collections import defaultdict
from copy import copy
from dataclasses import dataclass
import json
from pathlib import Path
import random
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field
import torch

from wsprl.batch import completed_attempt
from wsprl.serialization import canonical_json, prompt_payload, serialize_policy
from wsprl.training import TrainingConfig, load_checkpoint


CONDITIONS = ("correct", "no_weights", "incorrect", "random")
Task = Literal["individual", "paired"]
CaseKey = tuple[str, float]


class ResponseModel(BaseModel):
    model_config = ConfigDict(extra="forbid", strict=True, allow_inf_nan=False)


class Setting(ResponseModel):
    intervention: str
    severity: float

    def key(self) -> CaseKey:
        return self.intervention, self.severity


class Prediction(Setting):
    normalized_degradation: float
    failure_probability: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(min_length=1)


class IndividualResponse(ResponseModel):
    predictions: list[Prediction]
    least_to_most_harmful: list[Setting]


class PairedResponse(ResponseModel):
    more_vulnerable: Literal["A", "B", "tie"]
    confidence: float = Field(ge=0, le=1)
    reasoning: str = Field(min_length=1)


@dataclass(frozen=True)
class Trial:
    task: Task
    replicate: int
    condition: str
    target_ids: tuple[str, ...]
    sources: tuple[str | int, ...]
    cases: tuple[CaseKey, ...]
    messages: list[dict]

    def assignment(self) -> dict:
        return {
            "task": self.task, "replicate": self.replicate, "condition": self.condition,
            "target_ids": list(self.target_ids), "sources": list(self.sources),
            "cases": [list(case) for case in self.cases],
        }

    def validate(self, content: str) -> ResponseModel:
        if self.task == "paired":
            return PairedResponse.model_validate_json(content)
        result = IndividualResponse.model_validate_json(content)
        expected = sorted(self.cases)
        if (
            sorted(item.key() for item in result.predictions) != expected
            or sorted(item.key() for item in result.least_to_most_harmful) != expected
        ):
            raise ValueError("Each intervention setting must appear exactly once in each list")
        return result

    def schema(self) -> dict:
        response = IndividualResponse if self.task == "individual" else PairedResponse
        return response.model_json_schema()


def randomized_policy(model, env_id: str, seed: int) -> dict:
    """Generate untrained weights with the same SB3 policy constructor."""
    with torch.random.fork_rng(devices=[]):
        torch.random.default_generator.manual_seed(seed)
        untrained = copy(model)
        untrained.policy = model.policy_class(
            model.observation_space, model.action_space, model.lr_schedule, **model.policy_kwargs,
        )
    return serialize_policy(untrained, env_id)


def paired_message(first: dict, second: dict, case: dict, include_weights: bool) -> list[dict]:
    context = json.loads(prompt_payload(first, include_weights=False)["messages"][0]["content"])
    body = {
        "task": (
            "Which frozen policy, A or B, will experience a larger normalized return degradation "
            "under the specified intervention relative to its own nominal environment? "
            "Normalized degradation is (nominal mean return minus perturbed mean return) divided by "
            "the magnitude of the nominal mean. Choose A, B, or tie, provide confidence and brief reasoning. "
            "Use only the supplied information."
        ),
        "environment_task": context["environment_task"],
        "architecture": context["architecture"],
        "intervention_case": case,
        "policies": {"A": {}, "B": {}},
    }
    if include_weights:
        body["policies"]["A"]["parameters"] = first["parameters"]
        body["policies"]["B"]["parameters"] = second["parameters"]
    return [{"role": "user", "content": canonical_json(body)}]


def build_trials(
    batch_root: Path,
    *,
    replicates: int,
    order_seed: int,
    tasks: tuple[Task, ...],
    paired_case: CaseKey,
    failure_fraction: float = 0.25,
) -> list[Trial]:
    if replicates < 1 or not tasks or set(tasks) - {"individual", "paired"}:
        raise ValueError("Choose a positive replicate count and at least one supported task")
    if not 0 < failure_fraction < 1:
        raise ValueError("Failure fraction must be between 0 and 1")
    manifest = json.loads((batch_root / "manifest.json").read_text())
    groups = defaultdict(list)
    seen_checkpoints = set()
    weight_rng = random.Random(order_seed)
    for job, record in sorted(manifest["jobs"].items()):
        config = TrainingConfig.model_validate(record)
        checkpoint = completed_attempt(batch_root / job, config)
        if checkpoint is None:
            raise ValueError(f"No verified checkpoint for {job}")
        if checkpoint.name in seen_checkpoints:
            raise ValueError("The cohort contains duplicate checkpoint identifiers")
        seen_checkpoints.add(checkpoint.name)
        model = load_checkpoint(checkpoint)
        weights = serialize_policy(model, config.env_id)
        group = (config.algorithm, canonical_json(weights["architecture"]))
        random_controls = {}
        for replicate in range(replicates):
            for task in tasks:
                seed = weight_rng.getrandbits(63)
                random_controls[replicate, task] = (
                    seed, randomized_policy(model, config.env_id, seed)
                )
        groups[group].append((checkpoint.name, weights, random_controls))
        del model
    if not groups:
        raise ValueError("The cohort has no checkpoints")

    rng = random.Random(order_seed)
    trials = []
    for group in groups.values():
        if len(group) < 2:
            raise ValueError("Every environment, algorithm, and architecture group needs two checkpoints")
        catalog = json.loads(prompt_payload(group[0][1], include_weights=False)["messages"][0]["content"])
        public_cases = catalog["intervention_cases"]
        cases = tuple((case["intervention"], case["severity"]) for case in public_cases)
        env_id = group[0][1]["architecture"]["environment"]
        for replicate in range(replicates):
            ordered = rng.sample(group, len(group))
            # A nonzero cyclic shift is a one-to-one assignment without self-matches.
            shift = rng.randrange(1, len(group))
            controls = ordered[shift:] + ordered[:shift]
            for index, (target, correct, random_controls) in enumerate(ordered):
                if "individual" not in tasks:
                    break
                incorrect_id, incorrect, _ = controls[index]
                seed, random_weights = random_controls[replicate, "individual"]
                assignments = (
                    ((target,), correct), ((), correct), ((incorrect_id,), incorrect), ((seed,), random_weights),
                )
                for condition, (sources, weights) in zip(CONDITIONS, assignments, strict=True):
                    messages = prompt_payload(
                        weights, include_weights=bool(sources), failure_fraction=failure_fraction,
                    )["messages"]
                    trials.append(
                        Trial("individual", replicate, condition, (target,), sources, cases, messages)
                    )
            if "paired" not in tasks:
                continue
            if paired_case not in cases:
                raise ValueError(f"Paired intervention {paired_case!r} is unavailable for {env_id}")
            case = public_cases[cases.index(paired_case)]
            for first, second in zip(ordered[::2], ordered[1::2]):
                first_id, first_weights, first_random = first
                second_id, second_weights, second_random = second
                first_seed, first_control = first_random[replicate, "paired"]
                second_seed, second_control = second_random[replicate, "paired"]
                assignments = (
                    ((first_id, second_id), first_weights, second_weights),
                    ((), first_weights, second_weights),
                    ((second_id, first_id), second_weights, first_weights),
                    ((first_seed, second_seed), first_control, second_control),
                )
                for condition, (sources, a, b) in zip(CONDITIONS, assignments, strict=True):
                    messages = paired_message(a, b, case, include_weights=bool(sources))
                    trials.append(
                        Trial("paired", replicate, condition, (first_id, second_id), sources,
                              (paired_case,), messages)
                    )
    rng.shuffle(trials)
    return trials
