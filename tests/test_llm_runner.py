from contextlib import closing
from decimal import Decimal
import json
import sqlite3

import httpx
from openai import OpenAI
import pytest
from stable_baselines3 import PPO
import torch

from wsprl.environment import make_environment
from wsprl.llm_plan import CONDITIONS, Trial, build_trials
from wsprl.llm_runner import (
    BudgetExceeded, RequestConfig, UsageUncertain, execute_requests, prepare_requests,
)
from wsprl.serialization import serialize_policy
from wsprl.training import TrainingConfig, file_sha256


def test_trial_plan_uses_correct_blind_assignments_and_independent_replicates(tmp_path):
    jobs = {}
    target_weights = {}
    with make_environment("CartPole-v1") as env:
        for seed in range(3):
            config = TrainingConfig(seed=seed)
            job = f"PRIVATE_TRAINING_NAME/seed-{seed}"
            jobs[job] = config.model_dump()
            name = f"PRIVATE_CHECKPOINT_{seed}"
            folder = tmp_path / job / name
            folder.mkdir(parents=True)
            model = PPO(
                "MlpPolicy", env, seed=seed, device="cpu",
                policy_kwargs={"net_arch": {"pi": [8], "vf": [8]}, "activation_fn": torch.nn.Tanh},
            )
            target_weights[name] = serialize_policy(model, "CartPole-v1")["parameters"]
            model.save(folder / "policy.zip")
            (folder / "metadata.json").write_text(json.dumps({
                "checkpoint_id": name, "environment": "CartPole-v1", "status": "complete",
                "training": config.model_dump(), "checkpoint_sha256": file_sha256(folder / "policy.zip"),
                "evaluation_profile": "PRIVATE_GROUND_TRUTH",
            }))
    (tmp_path / "manifest.json").write_text(json.dumps({"jobs": jobs}))
    plan = dict(replicates=2, order_seed=3, tasks=("individual", "paired"),
                paired_case=("mask_probability", 0.3))
    trials = build_trials(tmp_path, **plan)
    assert trials == build_trials(tmp_path, **plan)
    assert len(trials) == (3 + 1) * 4 * 2
    assert list(trial.condition for trial in trials) != sorted(trial.condition for trial in trials)
    individual = {}
    paired = {}
    for trial in trials:
        content, = trial.messages
        assert content["role"] == "user" and "PRIVATE_" not in content["content"]
        body = json.loads(content["content"])
        if trial.task == "individual":
            individual.setdefault((trial.replicate, trial.target_ids), {})[trial.condition] = body
            if trial.condition == "correct":
                assert body["parameters"] == target_weights[trial.target_ids[0]]
            elif trial.condition == "incorrect":
                assert trial.sources[0] != trial.target_ids[0]
                assert body["parameters"] == target_weights[trial.sources[0]]
            elif trial.condition == "random":
                assert body["parameters"] not in target_weights.values()
        else:
            paired.setdefault((trial.replicate, trial.target_ids), {})[trial.condition] = body
            if trial.condition == "incorrect":
                assert trial.sources == trial.target_ids[::-1]
                assert body["policies"]["A"]["parameters"] == target_weights[trial.target_ids[1]]
                assert body["policies"]["B"]["parameters"] == target_weights[trial.target_ids[0]]
    for positions in individual.values():
        assert set(positions) == set(CONDITIONS)
        for condition, body in positions.items():
            assert ("parameters" in body) == (condition != "no_weights")
            assert {k: v for k, v in body.items() if k != "parameters"} == positions["no_weights"]
    for positions in paired.values():
        assert set(positions) == set(CONDITIONS)
        for condition, body in positions.items():
            assert ("parameters" in body["policies"]["A"]) == (condition != "no_weights")
            assert {k: v for k, v in body.items() if k != "policies"} == {
                k: v for k, v in positions["no_weights"].items() if k != "policies"
            }
    requests = prepare_requests(trials, config_for_tests())
    no_weights = [request for request in requests if request.trial.condition == "no_weights"]
    assert len({request.key for request in no_weights}) == len(no_weights)


def test_prediction_validation_rejects_missing_duplicate_and_invalid_settings():
    trial = Trial("individual", 0, "no_weights", ("X",), (), (("gravity_scale", 1.25),), [])
    setting = {"intervention": "gravity_scale", "severity": 1.25}
    prediction = {**setting, "normalized_degradation": -0.2,
                  "failure_probability": 0.1, "confidence": 0.5, "reasoning": "Test"}
    valid = {"predictions": [prediction], "least_to_most_harmful": [setting]}
    assert trial.validate(json.dumps(valid)).predictions[0].normalized_degradation == -0.2
    alternatives = [
        {**valid, "predictions": []},
        {**valid, "least_to_most_harmful": [setting, setting]},
        {**valid, "predictions": [{**prediction, "severity": 0.25}]},
        {**valid, "predictions": [{**prediction, "confidence": 2.0}]},
        {**valid, "predictions": [{**prediction, "failure_probability": "0.1"}]},
    ]
    for alternative in alternatives:
        with pytest.raises(ValueError):
            trial.validate(json.dumps(alternative))


def config_for_tests(**changes):
    return RequestConfig(**{
        "model": "mock-model", "input_price": "1", "output_price": "5",
        "max_output_tokens": 200, **changes,
    })


def paired_requests(config):
    trials = [
        Trial("paired", i, "no_weights", ("X", "Y"), (), (("mask_probability", 0.3),),
              [{"role": "user", "content": "Test anonymous pair"}])
        for i in range(2)
    ]
    return prepare_requests(trials, config)


def mock_client_factory(events, completions):
    def handle(request):
        body = json.loads(request.content) if request.content else None
        events.append((request.method, request.url.path, body))
        if request.method == "GET":
            return httpx.Response(200, json={"id": "mock-model", "object": "model",
                                            "created": 1, "owned_by": "mock"})
        result = completions.pop(0)
        if isinstance(result, Exception):
            raise result
        return httpx.Response(200, json=result, headers={"x-request-id": "req_test"})
    def create(**options):
        assert options["max_retries"] == 0
        return OpenAI(api_key="TEST_KEY_NEVER_LOGGED", http_client=httpx.Client(
            transport=httpx.MockTransport(handle)), **options)
    return create


def completion(*, prediction="A", usage=True):
    content = json.dumps({"more_vulnerable": prediction, "confidence": 0.4, "reasoning": "Test"})
    return {
        "id": "test-completion", "object": "chat.completion", "created": 1, "model": "mock-model",
        "choices": [{"index": 0, "finish_reason": "stop", "message": {
            "role": "assistant", "content": content,
        }}],
        "usage": {"prompt_tokens": 100, "completion_tokens": 20, "total_tokens": 120} if usage else None,
    }


def rows(path):
    with closing(sqlite3.connect(path)) as db:
        db.row_factory = sqlite3.Row
        return list(db.execute("SELECT * FROM requests ORDER BY rowid"))


def test_mock_sdk_cache_resumption_and_raw_records(tmp_path):
    config = config_for_tests()
    requests = paired_requests(config)
    assert requests[0].body == requests[1].body and requests[0].key != requests[1].key
    events = []
    factory = mock_client_factory(events, [completion(), completion(prediction="invalid")])
    path = tmp_path / "ledger.sqlite"
    first = execute_requests(requests, config, path, 1, client_factory=factory)
    assert first == {"sent": 1, "valid": 1}
    second = execute_requests(requests, config, path, 1, client_factory=factory)
    assert second == {"cached_valid": 1, "sent": 1, "invalid": 1}
    third = execute_requests(requests, config, path, 2, client_factory=factory)
    assert third == {"cached_valid": 1, "cached_invalid": 1}
    assert [(method, endpoint) for method, endpoint, _ in events] == [
        ("GET", "/v1/models/mock-model"), ("POST", "/v1/chat/completions"),
        ("GET", "/v1/models/mock-model"), ("POST", "/v1/chat/completions"),
    ]
    first_record, invalid_record = rows(path)
    assert json.loads(first_record["request_json"]) == events[1][2] == requests[0].body
    assert json.loads(first_record["response_body"]) == completion()
    assert json.loads(first_record["prediction_json"])["more_vulnerable"] == "A"
    assert first_record["request_id"] == "req_test"
    assert first_record["input_tokens"] == 100
    assert first_record["output_tokens"] == 20
    assert first_record["billed_nusd"] == 200_000
    assert invalid_record["status"] == "invalid" and invalid_record["prediction_json"] is None
    assert b"TEST_KEY_NEVER_LOGGED" not in path.read_bytes()
    with pytest.raises(ValueError, match="differs"):
        execute_requests(requests[::-1], config, path, 1, client_factory=factory)


def test_budget_blocks_second_call_after_actual_cost(tmp_path):
    reference = config_for_tests()
    request = paired_requests(reference)[0]
    config = config_for_tests(budget=Decimal(request.estimated_cost + 199_999) / 10**9)
    assert config.cost(100, 20) == 200_000
    events = []
    factory = mock_client_factory(events, [completion()])
    path = tmp_path / "ledger.sqlite"
    with pytest.raises(BudgetExceeded):
        execute_requests(paired_requests(config), config, path, 2, client_factory=factory)
    assert len(events) == 2 and len(rows(path)) == 1
    with pytest.raises(BudgetExceeded):
        execute_requests(paired_requests(config), config, path, 1,
                         client_factory=factory, experiment_id="another-experiment")
    assert len(events) == 2 and len(rows(path)) == 1


@pytest.mark.parametrize("result", [completion(usage=False), httpx.ReadTimeout("timed out")])
def test_uncertain_billing_retains_reservation_and_stops(tmp_path, result):
    events = []
    factory = mock_client_factory(events, [result])
    path = tmp_path / "ledger.sqlite"
    config = config_for_tests()
    requests = paired_requests(config)
    with pytest.raises(UsageUncertain):
        execute_requests(requests, config, path, 2, client_factory=factory)
    assert len(events) == 2
    row, = rows(path)
    assert row["status"] == "uncertain"
    assert row["billed_nusd"] is None and row["reserved_nusd"] > 0
    with pytest.raises(UsageUncertain):
        execute_requests(requests, config, path, 2, client_factory=factory)
    assert len(events) == 2
