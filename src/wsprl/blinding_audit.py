"""Inspect all pilot requests as emitted by the SDK's in-process mock transport."""
from collections import Counter
import json
from pathlib import Path

import httpx
from openai import OpenAI

from wsprl.llm_plan import build_trials
from wsprl.llm_runner import RequestConfig, execute_requests, prepare_requests


ARCHITECTURE_KEYS = {
    "environment", "observations", "actions", "input_preprocessing", "output_type",
    "deterministic_action", "parameter_dtype", "tensor_order", "layers",
}
TRANSPORT_KEYS = {"model", "messages", "response_format", "max_completion_tokens", "store"}
FORBIDDEN_KEYS = {
    "seed", "training", "training_seed", "training_intervention", "training_config",
    "checkpoint_id", "checkpoint_sha256", "job", "policy_id", "filename", "condition",
    "replicate", "nominal_mean", "perturbed_mean", "absolute_degradation",
    "failure_frequency", "eval_seed", "evaluation_results", "ground_truth",
}


def require(condition, message):
    if not condition:
        raise ValueError(message)


def inspect_body(raw: str, task: str, condition: str, private_markers: tuple[str, ...]) -> dict:
    body = json.loads(raw)
    require(set(body) == TRANSPORT_KEYS, "Unexpected API request fields")
    require(body["store"] is False, "Unexpected response storage setting")
    require(len(body["messages"]) == 1, "Unexpected conversation history")
    message, = body["messages"]
    require(set(message) == {"role", "content"} and message["role"] == "user",
            "Unexpected message role or metadata")
    public = json.loads(message["content"])
    serialized_public = json.dumps(public)
    require(
        not any(marker in raw or marker in serialized_public for marker in private_markers),
        "Private identifier or metadata marker in request",
    )

    def inspect_keys(value):
        if isinstance(value, dict):
            require(not (set(value) & FORBIDDEN_KEYS), "Private metadata key in request")
            for child in value.values():
                inspect_keys(child)
        elif isinstance(value, list):
            for child in value:
                inspect_keys(child)

    inspect_keys(public)
    require(set(public["architecture"]) == ARCHITECTURE_KEYS, "Unexpected architecture metadata")
    common = {"task", "environment_task", "architecture"}
    weighted = condition != "no_weights"
    if task == "individual":
        allowed = common | {"intervention_cases"}
        require(set(public) == allowed | ({"parameters"} if weighted else set()),
                "Unexpected individual prompt fields")
        public.pop("parameters", None)
    else:
        require(set(public) == common | {"intervention_case", "policies"},
                "Unexpected paired prompt fields")
        require(set(public["policies"]) == {"A", "B"}, "Unexpected policy aliases")
        for policy in public["policies"].values():
            require(set(policy) == ({"parameters"} if weighted else set()),
                    "Unexpected policy alias metadata")
            policy.pop("parameters", None)
    return public


def audit_requests(batch_root: Path, output: Path, policy_weights: dict) -> dict:
    trials = build_trials(
        batch_root, replicates=1, order_seed=44,
        tasks=("individual", "paired"), paired_case=("mask_probability", 0.3),
    )
    config = RequestConfig(
        model="gpt-5.4-mini-2026-03-17", input_price="0.75", output_price="4.50",
    )
    requests = prepare_requests(trials, config)
    captured = []
    markers = (
        "/Users/", str(batch_root), "policy.zip", "metadata.json", "manifest.json",
        "training_episodes.jsonl", "nominal-screen.json", "vulnerability.csv",
        "raw/episodes.csv", "seed-0", "seed-1", "seed-2",
        "mock-audit-credential", *policy_weights,
    )

    def mock_transport(request):
        if request.method == "GET":
            require(request.url.path == f"/v1/models/{config.model}", "Unexpected model check")
            return httpx.Response(200, json={
                "id": config.model, "object": "model", "created": 1, "owned_by": "mock",
            })
        require(request.url.path == "/v1/chat/completions", "Unexpected completion endpoint")
        trial = trials[len(captured)]
        captured.append(request.content.decode())
        if trial.task == "paired":
            forecast = {"more_vulnerable": "tie", "confidence": 0.5, "reasoning": "Mock only"}
        else:
            cases = [{"intervention": name, "severity": severity} for name, severity in trial.cases]
            forecast = {
                "least_to_most_harmful": cases,
                "predictions": [
                    {**case, "normalized_degradation": 0, "failure_probability": 0,
                     "confidence": 0.5, "reasoning": "Mock only"} for case in cases
                ],
            }
        return httpx.Response(200, json={
            "id": f"mock-{len(captured)}", "object": "chat.completion", "created": 1,
            "model": config.model,
            "choices": [{"index": 0, "finish_reason": "stop", "message": {
                "role": "assistant", "content": json.dumps(forecast),
            }}],
            "usage": {"prompt_tokens": 1500, "completion_tokens": 600, "total_tokens": 2100},
        })

    def client_factory(**options):
        return OpenAI(
            api_key="mock-audit-credential",
            http_client=httpx.Client(transport=httpx.MockTransport(mock_transport)),
            **options,
        )

    counts = execute_requests(
        requests, config, output / "mock-ledger.sqlite", len(requests), client_factory=client_factory,
    )
    require(counts["valid"] == len(requests), "Mock responses did not validate")
    comparable = {}
    captures = output / "mock-request-bodies.jsonl"
    with captures.open("x") as stream:
        for request, raw in zip(requests, captured, strict=True):
            trial = request.trial
            require(json.loads(raw) == request.body, "SDK changed the planned API request")
            public = inspect_body(raw, trial.task, trial.condition, markers)
            key = (trial.task, trial.cases, public["architecture"]["environment"])
            require(public == comparable.setdefault(key, public),
                    "Task context changed by condition or checkpoint")
            content = json.loads(json.loads(raw)["messages"][0]["content"])
            if trial.condition in {"correct", "incorrect"}:
                sent = ([content["parameters"]] if trial.task == "individual" else
                        [content["policies"][alias]["parameters"] for alias in ("A", "B")])
                require(sent == [policy_weights[source] for source in trial.sources],
                        "Sent parameters did not equal source checkpoint tensors")
            if trial.condition == "random":
                sent = ([content["parameters"]] if trial.task == "individual" else
                        [content["policies"][alias]["parameters"] for alias in ("A", "B")])
                require(all(parameters not in policy_weights.values() for parameters in sent),
                        "Random control was a trained checkpoint")
            stream.write(json.dumps({"trial_key": request.key, "wire_body": raw}) + "\n")

    require(b"mock-audit-credential" not in (output / "mock-ledger.sqlite").read_bytes(),
            "Mock credential in ledger")
    return {
        "mock_requests": len(captured),
        "requests_by_task": dict(Counter(trial.task for trial in trials)),
        "requests_by_condition": dict(Counter(trial.condition for trial in trials)),
        "private_marker_count": len(markers),
        "context_matched_across_conditions": True,
        "correct_and_incorrect_parameters_matched_checkpoint": True,
        "random_controls_distinct_from_trained_checkpoints": True,
    }
