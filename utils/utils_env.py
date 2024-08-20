import abc
from dataclasses import dataclass
from typing import Any, List

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from worldllm_envs.envs.base import BaseRule, BaseRuleEnv


class BaseAgent(abc.ABC):
    """Base class for the agents."""

    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    @abc.abstractmethod
    def __call__(self, obs):
        """Generate action"""


class RandomAgent(BaseAgent):
    """The agent that samples actions uniformly."""

    def __call__(self, obs):
        return self.action_space.sample()


class AllAgent(BaseAgent):
    """The agent that samples all actions."""

    def __init__(self, action_space: gym.Space):
        super().__init__(action_space)
        assert isinstance(
            action_space, gym.spaces.MultiDiscrete
        ), "Only implemented for MultiDiscrete action space."
        _arr_actions = np.indices(self.action_space.nvec)
        self._arr_actions = np.stack(_arr_actions, axis=-1)
        self.flat_index = 0

    def __call__(self, obs):
        if self.flat_index >= np.prod(self._arr_actions.shape[:-1]):
            raise ValueError(
                f"All actions have been sampled, lower the number of trajectories to {np.prod(self._arr_actions.shape[:-1])}."
            )
        action = np.unravel_index(self.flat_index, self._arr_actions.shape[:-1])
        self.flat_index += 1
        return action


@dataclass
class Trajectory:
    """Save information on some rollout for the llms."""

    text: List[str]
    obs: List[Any]

    def __len__(self):
        return len(self.text)

    def get_full_text(self) -> str:
        return " ".join(self.text)


def build_env(cfg: DictConfig):
    """Build the environment."""
    # Add seed to kwargs
    kwargs = OmegaConf.to_container(cfg.environment.kwargs, resolve=True)
    kwargs["seed"] = cfg.seed
    env = gym.make(cfg.environment.id, **kwargs)
    if not isinstance(env.unwrapped, BaseRuleEnv):
        raise ValueError("The environment must be rule based.")
    return env


def generate_text_trajectories(
    env: BaseRuleEnv, agent: BaseAgent, rule: BaseRule, nb_trajectories: int
) -> List[Trajectory]:
    """Generate random trajectories for the environment."""
    # Set rule
    obs, _ = env.reset(options={"rule": rule})
    lst_trajectory = []
    for _ in tqdm(range(nb_trajectories), desc="Generating trajectories"):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        lst_trajectory.append(
            Trajectory(info["text_trajectory"], info["obs_trajectory"])
        )
    return lst_trajectory
