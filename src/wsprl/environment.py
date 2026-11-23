"""Classic-control environments and independently seeded interventions."""
from typing import Literal, get_args

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field


EnvironmentId = Literal["CartPole-v1", "MountainCar-v0", "Acrobot-v1"]


class Interventions(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", allow_inf_nan=False)

    position_noise_std: float = Field(default=0, ge=0, le=1e30)
    velocity_noise_std: float = Field(default=0, ge=0, le=1e30)
    mask_probability: float = Field(default=0, ge=0, le=1)
    gravity_scale: float = Field(default=1, gt=0)
    pole_length_scale: float = Field(default=1, gt=0)
    action_flip_probability: float = Field(default=0, ge=0, le=1)

    def validate_for(self, env_id: EnvironmentId) -> None:
        if env_id == "CartPole-v1":
            return
        fields = ("position_noise_std", "velocity_noise_std", "gravity_scale", "pole_length_scale")
        defaults = Interventions()
        unsupported = [name for name in fields if getattr(self, name) != getattr(defaults, name)]
        if unsupported:
            raise ValueError(f"Interventions unavailable for {env_id}: {', '.join(unsupported)}")


class ClassicControlInterventions(gym.Wrapper):
    def __init__(self, env: gym.Env, config: Interventions):
        super().__init__(env)
        self.config = config
        if env.spec.id == "CartPole-v1":
            # Keep the same declared bounds across nominal and noisy CartPole runs.
            low = env.observation_space.low.copy()
            high = env.observation_space.high.copy()
            low[0], high[0] = -np.inf, np.inf
            self.observation_space = gym.spaces.Box(low, high, dtype=np.float32)
        self._seed_streams(None)

    def _seed_streams(self, seed: int | None):
        children = np.random.SeedSequence(seed).spawn(4)
        self._position_rng, self._velocity_rng, self._mask_rng, self._action_rng = (
            np.random.default_rng(child) for child in children
        )

    def reset(self, *, seed=None, options=None):
        observation, info = super().reset(seed=seed, options=options)
        if seed is not None:
            self._seed_streams(seed)
        return self._observe(observation), info

    def step(self, action):
        if not self.action_space.contains(action):
            raise ValueError(f"Invalid {self.spec.id} action: {action!r}")
        probability = self.config.action_flip_probability
        if probability and self._action_rng.random() < probability:
            if self.action_space.n == 2:
                action = 1 - int(action)
            else:
                # Draw from the remaining actions by skipping the requested index.
                replacement = int(self._action_rng.integers(self.action_space.n - 1))
                action = replacement + (replacement >= int(action))
        observation, reward, terminated, truncated, info = super().step(action)
        return self._observe(observation), reward, terminated, truncated, info

    def _observe(self, observation):
        observed = observation.copy()
        config = self.config
        if config.position_noise_std:
            observed[0] += self._position_rng.normal(scale=config.position_noise_std)
        if config.velocity_noise_std:
            observed[1] += self._velocity_rng.normal(scale=config.velocity_noise_std)
        if config.mask_probability:
            observed[self._mask_rng.random(observed.shape) < config.mask_probability] = 0
        return observed


def _configure_cartpole_physics(env: gym.Env, config: Interventions) -> None:
    physics = env.unwrapped
    gravity = physics.gravity * config.gravity_scale
    length = physics.length * config.pole_length_scale
    polemass_length = physics.masspole * length
    if not (np.isfinite(gravity) and length > 0 and np.isfinite(length * 2)
            and polemass_length > 0):
        raise ValueError("Gravity or pole length cannot be represented safely")
    physics.gravity = gravity
    physics.length = length
    physics.polemass_length = polemass_length


def make_environment(env_id: EnvironmentId, config: Interventions | None = None) -> gym.Env:
    if env_id not in get_args(EnvironmentId):
        raise ValueError(f"Unsupported environment: {env_id}")
    config = config or Interventions()
    config.validate_for(env_id)
    env = gym.make(env_id)
    try:
        if env_id == "CartPole-v1":
            _configure_cartpole_physics(env, config)
        return ClassicControlInterventions(env, config)
    except Exception:
        env.close()
        raise
