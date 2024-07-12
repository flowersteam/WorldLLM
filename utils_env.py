import abc
from dataclasses import dataclass
from typing import Any, List

from worldllm_envs.envs.base import BaseRule, BaseRuleEnv


class BaseAgent(abc.ABC):
    def __init__(self, action_space):
        self.action_space = action_space

    @abc.abstractmethod
    def __call__(self, obs):
        """Generate action"""


class RandomAgent(BaseAgent):
    def __call__(self, obs):
        return self.action_space.sample()


@dataclass
class Trajectory:
    """Save information on some rollout for the llms."""

    text: List[str]
    obs: List[Any]

    def __len__(self):
        return len(self.text)

    def get_full_text(self) -> str:
        return " ".join(self.text)


def generate_text_trajectories(
    env: BaseRuleEnv, agent: BaseAgent, rule: BaseRule, nb_trajectories: int
) -> List[Trajectory]:
    """Generate random trajectories for the environment."""
    # Set rule
    obs, _ = env.reset(options={"rule": rule})
    lst_trajectory = []
    for _ in range(nb_trajectories):
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
