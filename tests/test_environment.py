from contextlib import ExitStack

import gymnasium as gym
import numpy as np
from pydantic import ValidationError
import pytest

from wsprl.environment import Interventions, make_environment


@pytest.mark.parametrize("env_id,observation_size,action_count", [
    ("CartPole-v1", 4, 2),
    ("MountainCar-v0", 2, 3),
    ("Acrobot-v1", 6, 3),
])
def test_environment_factory_dispatches_to_requested_simulator(
    env_id, observation_size, action_count,
):
    with make_environment(env_id) as env, gym.make(env_id) as stock:
        assert env.spec.id == env_id
        assert env.action_space.n == action_count
        observation, _ = env.reset(seed=11)
        assert observation.shape == (observation_size,)
        np.testing.assert_array_equal(observation, stock.reset(seed=11)[0])
        for action in range(action_count):
            actual = env.step(action)
            expected = stock.step(action)
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:] == expected[1:]


@pytest.mark.parametrize("env_id", ["MountainCar-v0", "Acrobot-v1"])
def test_shared_interventions_on_three_action_environments(env_id):
    flipped = Interventions(action_flip_probability=1)
    both = Interventions(action_flip_probability=1, mask_probability=1)
    with ExitStack() as stack:
        env = stack.enter_context(make_environment(env_id, flipped))
        masked = stack.enter_context(make_environment(env_id, both))
        stock = stack.enter_context(gym.make(env_id))
        replacements = set()
        for seed in range(12):
            env.reset(seed=seed)
            np.testing.assert_array_equal(masked.reset(seed=seed)[0], 0)
            observation, *outcome = env.step(1)
            masked_observation, *masked_outcome = masked.step(1)
            np.testing.assert_array_equal(masked_observation, 0)
            np.testing.assert_array_equal(env.unwrapped.state, masked.unwrapped.state)
            assert masked_outcome == outcome
            matches = []
            for action in range(3):
                stock.reset(seed=seed)
                candidate, *candidate_outcome = stock.step(action)
                if np.array_equal(observation, candidate) and outcome == candidate_outcome:
                    matches.append(action)
            assert len(matches) == 1 and matches[0] != 1
            replacements.add(matches[0])
        assert replacements == {0, 2}


@pytest.mark.parametrize("field,value", [
    ("position_noise_std", -1),
    ("velocity_noise_std", float("nan")),
    ("mask_probability", 1.01),
    ("action_flip_probability", -0.01),
    ("gravity_scale", 0),
    ("pole_length_scale", float("inf")),
    ("unknown", 1),
])
def test_invalid_settings(field, value):
    with pytest.raises(ValidationError):
        Interventions(**{field: value})


def test_default_matches_stock_including_time_limit():
    with gym.make("CartPole-v1") as stock, make_environment("CartPole-v1") as wrapped:
        # With zero initial state and no force, the system reaches the time limit.
        for env in (stock, wrapped):
            env.unwrapped.force_mag = 0
        options = {"low": 0, "high": 0}
        np.testing.assert_array_equal(
            stock.reset(seed=8, options=options)[0],
            wrapped.reset(seed=8, options=options)[0],
        )
        for _ in range(500):
            expected = stock.step(0)
            actual = wrapped.step(0)
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:] == expected[1:]
        assert actual[1:4] == (1.0, False, True)


@pytest.mark.parametrize("field,index", [
    ("position_noise_std", 0),
    ("velocity_noise_std", 1),
])
def test_noise_changes_only_the_selected_observation(field, index):
    with make_environment("CartPole-v1") as base, make_environment("CartPole-v1", Interventions(**{field: 20})) as noisy:
        ordinary, _ = base.reset(seed=42)
        perturbed, _ = noisy.reset(seed=42)
        assert noisy.observation_space.contains(perturbed)
        assert perturbed[index] != ordinary[index]
        np.testing.assert_array_equal(np.delete(perturbed, index), np.delete(ordinary, index))
        for action in [0, 1] * 4:
            expected, *expected_outcome = base.step(action)
            actual, *actual_outcome = noisy.step(action)
            assert actual_outcome == expected_outcome
            np.testing.assert_array_equal(np.delete(actual, index), np.delete(expected, index))
            np.testing.assert_array_equal(noisy.unwrapped.state, base.unwrapped.state)


def test_full_mask_is_observation_only_and_applied_after_noise():
    config = Interventions(position_noise_std=1, velocity_noise_std=1, mask_probability=1)
    with make_environment("CartPole-v1", config) as masked, make_environment("CartPole-v1") as nominal:
        np.testing.assert_array_equal(masked.reset(seed=2)[0], np.zeros(4))
        nominal.reset(seed=2)
        actual = masked.step(1)
        expected = nominal.step(1)
        np.testing.assert_array_equal(actual[0], np.zeros(4))
        np.testing.assert_array_equal(masked.unwrapped.state, nominal.unwrapped.state)
        assert actual[1:] == expected[1:]


def test_full_action_flip_matches_opposite_actions():
    with make_environment("CartPole-v1", Interventions(action_flip_probability=1)) as flipped, make_environment("CartPole-v1") as base:
        flipped.reset(seed=15)
        base.reset(seed=15)
        assert flipped.action_space == base.action_space
        for action in [0, 1] * 4:
            actual = flipped.step(action)
            expected = base.step(1 - action)
            np.testing.assert_array_equal(actual[0], expected[0])
            assert actual[1:] == expected[1:]
        with pytest.raises(ValueError):
            flipped.step(2)


@pytest.mark.parametrize("gravity_scale,pole_length_scale", [(1.5, 1), (1, 0.7)])
def test_dynamics_match_one_euler_step(gravity_scale, pole_length_scale):
    config = Interventions(gravity_scale=gravity_scale, pole_length_scale=pole_length_scale)
    with make_environment("CartPole-v1", config) as env:
        env.reset(seed=1)
        env.unwrapped.state = (0.05, 0.1, 0.13, -0.15)
        x, x_dot, theta, theta_dot = env.unwrapped.state
        half_length = 0.5 * pole_length_scale
        polemass_length = 0.1 * half_length
        temp = (10 + polemass_length * theta_dot**2 * np.sin(theta)) / 1.1
        angular_acceleration = (
            9.8 * gravity_scale * np.sin(theta) - np.cos(theta) * temp
        ) / (half_length * (4 / 3 - 0.1 * np.cos(theta)**2 / 1.1))
        acceleration = temp - polemass_length * angular_acceleration * np.cos(theta) / 1.1
        expected = [
            x + 0.02 * x_dot,
            x_dot + 0.02 * acceleration,
            theta + 0.02 * theta_dot,
            theta_dot + 0.02 * angular_acceleration,
        ]
        np.testing.assert_allclose(env.step(1)[0], expected, rtol=1e-6)


def test_reseed_replays_stochastic_interventions_and_unseeded_resets_continue():
    config = Interventions(
        position_noise_std=0.2, velocity_noise_std=0.3,
        mask_probability=0.25, action_flip_probability=0.4,
    )

    def trace(env, seed):
        observations = [env.reset(seed=seed)[0]]
        for _ in range(6):
            observations.append(env.step(1)[0])
        return np.stack(observations)

    with make_environment("CartPole-v1", config) as first, make_environment("CartPole-v1", config) as second:
        original = trace(first, 22)
        np.testing.assert_array_equal(trace(second, 22), original)
        continuation = trace(first, None)
        np.testing.assert_array_equal(trace(second, None), continuation)
        assert not np.array_equal(continuation, original)
        np.testing.assert_array_equal(trace(first, 22), original)
        assert not np.array_equal(trace(first, 23), original)


def test_stochastic_streams_are_independent():
    common = dict(velocity_noise_std=0.3, action_flip_probability=0.4)
    with ExitStack() as stack:
        plain, noisy, masked, velocity_only = [
            stack.enter_context(make_environment("CartPole-v1", config))
            for config in (
                Interventions(**common),
                Interventions(**common, position_noise_std=0.3),
                Interventions(**common, mask_probability=0.25),
                Interventions(velocity_noise_std=0.3),
            )
        ]
        for env in (plain, noisy, masked, velocity_only):
            env.reset(seed=22)
        for _ in range(8):
            ordinary = plain.step(1)[0]
            with_noise = noisy.step(1)[0]
            with_mask = masked.step(1)[0]
            velocity_only.step(1)
            assert ordinary[1] == with_noise[1]
            np.testing.assert_array_equal(plain.unwrapped.state, noisy.unwrapped.state)
            np.testing.assert_array_equal(plain.unwrapped.state, masked.unwrapped.state)
            np.testing.assert_array_equal(with_mask[with_mask != 0], ordinary[with_mask != 0])
        assert not np.array_equal(plain.unwrapped.state, velocity_only.unwrapped.state)
