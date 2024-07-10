import abc
from typing import List

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


def generate_text_trajectories(
    env: BaseRuleEnv, agent: BaseAgent, rule: BaseRule, nb_trajectories: int
) -> List[str]:
    """Generate random trajectories for the environment."""
    # Set rule
    obs, _ = env.reset(options={"rule": rule})
    trajectories = []
    for _ in range(nb_trajectories):
        obs, _ = env.reset()
        done = False
        while not done:
            action = agent(obs)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        trajectories.append(info["trajectory"])
    return trajectories
