import json

import numpy as np
import pytest

from wsprl.blinding_audit import ARCHITECTURE_KEYS, inspect_body
from wsprl.diversity_audit import compare_banks, profile_interactions
from wsprl.llm_plan import Trial
from wsprl.llm_runner import RequestConfig, prepare_requests


def test_additive_policy_offsets_do_not_count_as_profile_interactions():
    shared = np.array([[100, 105, 110], [50, 52, 54], [80, 79, 78]])
    values = np.stack([shared, shared - 25, shared + 20])
    _, gaps, bound = profile_interactions(values, normalized=False, draws=100)
    np.testing.assert_allclose(gaps, 0, atol=1e-12)
    assert bound < 1e-12


def test_replicated_interactions_are_detected_after_familywise_noise_adjustment():
    rng = np.random.default_rng(22)
    first = np.array([[100, 60, 90], [100, 90, 60]])[:, :, None] + rng.integers(
        -2, 3, size=(2, 3, 20)
    )
    second = np.array([[100, 60, 90], [100, 90, 60]])[:, :, None] + rng.integers(
        -2, 3, size=(2, 3, 30)
    )
    cases = [("a", 0.1), ("b", 0.1)]
    result = compare_banks(first, second, ["one", "two"], cases, normalized=True)
    assert result["replicated_nonzero_contrasts"] == 2
    assert result["replicated_with_lower_bound_above"]["0.2"] == 2
    assert result["fresh_simultaneous_95_half_width"] < 0.05
    reversed_result = compare_banks(first, second[::-1], ["one", "two"], cases, normalized=True)
    assert reversed_result["replicated_nonzero_contrasts"] == 0


def test_wire_auditor_rejects_metadata_hidden_in_allowed_and_nested_fields():
    public = {
        "task": "Predict vulnerability", "environment_task": "Control the cart",
        "architecture": dict.fromkeys(ARCHITECTURE_KEYS, "public"),
        "intervention_cases": [{"intervention": "mask_probability", "severity": 0.3}],
    }
    config = RequestConfig(model="mock", input_price="1", output_price="1")
    def body(content):
        trial = Trial("individual", 0, "no_weights", ("id",), (), (("mask_probability", 0.3),),
                      [{"role": "user", "content": json.dumps(content)}])
        return json.dumps(prepare_requests([trial], config)[0].body)

    marker = "PRIVATE_CHECKPOINT_123"
    assert inspect_body(body(public), "individual", "no_weights", (marker,)) == public
    try_string = {**public, "task": f"Predict using {marker}"}
    with pytest.raises(ValueError, match="identifier"):
        inspect_body(body(try_string), "individual", "no_weights", (marker,))
    try_numeric = {**public, "intervention_cases": [{"nominal_mean": 499.9}]}
    with pytest.raises(ValueError, match="Private metadata key"):
        inspect_body(body(try_numeric), "individual", "no_weights", (marker,))
