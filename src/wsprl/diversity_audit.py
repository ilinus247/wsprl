"""Quantify pilot policy-by-intervention effects against matched reset-seed noise."""
from itertools import combinations
from pathlib import Path

import numpy as np
import pandas as pd


def read_bank(path: Path, checkpoints: dict[str, str]):
    rows = pd.read_csv(path)
    observed = rows[["job", "checkpoint_id"]].drop_duplicates()
    if len(observed) != len(checkpoints) or dict(observed.itertuples(index=False, name=None)) != checkpoints:
        raise ValueError("Episode checkpoint identities differ from the protocol")
    cases = sorted({
        (kind, float(level))
        for kind, level in rows.loc[rows.severity.notna(), ["intervention", "severity"]]
        .itertuples(index=False, name=None)
    })
    rows["case"] = list(zip(rows.intervention, rows.severity.fillna(0)))
    wide = rows.pivot(index="eval_seed", columns=["job", "case"], values="return")
    policies = sorted(rows.job.unique())
    channels = [("nominal", 0), *cases]
    values = np.stack([
        np.stack([wide[(policy, case)].to_numpy() for case in channels])
        for policy in policies
    ])
    if not np.isfinite(values).all() or values.shape[-1] < 2:
        raise ValueError("Expected one finite return for each policy, setting, and reset seed")
    return policies, cases, wide.index.to_numpy(), values


def profile(vectors: np.ndarray, *, normalized: bool) -> np.ndarray:
    # Axis -2 indexes settings, and axis -1 indexes reset seeds.
    means = vectors.mean(axis=-1)
    baseline = means[..., :1]
    drops = baseline - means[..., 1:]
    if normalized:
        if np.any(baseline == 0):
            raise ValueError("Cannot normalize a zero nominal mean")
        drops = drops / np.abs(baseline)
    return drops - drops.mean(axis=-1, keepdims=True)


def profile_interactions(
    values: np.ndarray, *, normalized: bool, draws: int = 3000, seed: int = 9215,
):
    """Bootstrap whole reset-seed blocks and bound the family of all pair-by-case contrasts."""
    if values.ndim != 3 or min(values.shape) < 2 or draws < 2:
        raise ValueError("Expected at least two policies, settings, seeds, and bootstrap draws")
    pairs = list(combinations(range(len(values)), 2))
    first, second = np.asarray(pairs).T
    centers = profile(values, normalized=normalized)
    observed = centers[first] - centers[second]
    rng = np.random.default_rng(seed)
    maxima = np.empty(draws)
    # Chunk resampling so a larger audit does not allocate every sampled trajectory at once.
    for start in range(0, draws, 100):
        size = min(100, draws - start)
        indices = rng.integers(values.shape[-1], size=(size, values.shape[-1]))
        sampled = np.moveaxis(values[..., indices], -2, 0)
        bootstrap = profile(sampled, normalized=normalized)
        errors = bootstrap[:, first] - bootstrap[:, second] - observed
        maxima[start:start + size] = np.abs(errors).max(axis=(1, 2))
    half_width = float(np.quantile(maxima, 0.95))
    return pairs, observed, half_width


def compare_banks(first: np.ndarray, second: np.ndarray, policies, cases, *, normalized: bool):
    pairs, a, width_a = profile_interactions(first, normalized=normalized)
    _, b, width_b = profile_interactions(second, normalized=normalized)
    replicated = (np.sign(a) == np.sign(b)) & (np.abs(a) > width_a) & (np.abs(b) > width_b)
    lower = np.minimum(np.abs(a) - width_a, np.abs(b) - width_b)
    cutoff = (0.05, 0.10, 0.20) if normalized else (25, 50, 100)
    if np.std(a) and np.std(b):
        correlation = float(np.corrcoef(a.ravel(), b.ravel())[0, 1])
    else:
        correlation = None
    order = np.argsort(np.where(replicated, lower, -np.inf).ravel())[::-1]
    examples = []
    for index in order[:5]:
        pair, case = np.unravel_index(index, a.shape)
        if not replicated[pair, case]:
            break
        i, j = pairs[pair]
        examples.append({
            "policies": [policies[i], policies[j]],
            "intervention": cases[case][0],
            "severity": cases[case][1],
            "first_gap": float(a[pair, case]),
            "fresh_gap": float(b[pair, case]),
        })
    return {
        "policy_count": len(policies),
        "policy_pairs": len(pairs),
        "contrasts": a.size,
        "profile_gap_correlation": correlation,
        "first_rms_gap": float(np.sqrt(np.mean(a**2))),
        "fresh_rms_gap": float(np.sqrt(np.mean(b**2))),
        "first_simultaneous_95_half_width": width_a,
        "fresh_simultaneous_95_half_width": width_b,
        "replicated_nonzero_contrasts": int(replicated.sum()),
        "replicated_policy_pairs": int(replicated.any(axis=1).sum()),
        "replicated_with_lower_bound_above": {
            str(level): int((replicated & (lower > level)).sum()) for level in cutoff
        },
        "strongest_replicated_contrasts": examples,
    }
