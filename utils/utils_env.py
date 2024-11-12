import abc
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm

from worldllm_envs.base import BaseRule, BaseRuleEnv


class BaseAgent(abc.ABC):
    """Base class for the agents."""

    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    @abc.abstractmethod
    def __call__(self, obs, **kwargs) -> Tuple[str, bool]:
        """Generate action"""

    def reset(self, info: Dict[str, Any]):
        """Reset the agent."""
        pass


class RandomAgent(BaseAgent):
    """The agent that samples actions uniformly while respecting the action mask."""

    def __call__(self, obs, **kwargs):
        if "action_mask" in kwargs:
            action_mask = kwargs["action_mask"]
            possible_actions = np.arange(len(action_mask))[action_mask]
        return possible_actions[np.random.randint(len(possible_actions))], False


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

    lst_obs: List[str]
    lst_act: List[str]
    lst_diff: List[str]

    def __len__(self):
        return len(self.lst_diff)


def build_env(cfg: DictConfig, rule: Optional[str] = None) -> BaseRuleEnv:
    """Build the environment."""
    # Add seed to kwargs
    kwargs = OmegaConf.to_container(cfg.environment.kwargs, resolve=True)
    kwargs["seed"] = cfg.seed
    env = gym.make(cfg.environment.id, **kwargs)
    if not isinstance(env.unwrapped, BaseRuleEnv):
        raise ValueError("The environment must be rule based.")
    if rule is not None:
        env.reset(options={"rule": rule})
    return env


def generate_text_trajectories(
    env: BaseRuleEnv,
    agent: BaseAgent,
    nb_trajectories: int,
    progression: float,
) -> Tuple[List[Trajectory], Set[str]]:
    """Generate random trajectories for the environment."""
    # Set rule
    lst_trajectory = []
    set_discovered_transitions = set()
    for _ in tqdm(
        range(nb_trajectories),
        desc="Generating trajectories",
        leave=False,
    ):
        obs, info = env.reset()
        info["pipeline_progression"] = (
            progression  # Add progression to info for curriculum learning
        )
        agent.reset(info)
        done = False
        while not done:
            action, agent_done = agent(obs, **info)
            obs, _, terminated, truncated, info = env.step(action)
            set_discovered_transitions.add(info["transition_type"])
            done = terminated or truncated or agent_done
        lst_trajectory.append(
            Trajectory(
                info["trajectory_obs_text"],
                info["trajectory_act_text"],
                info["trajectory_diff_text"],
            )
        )
    return lst_trajectory, set_discovered_transitions
