import abc
from typing import List

from worldllm_envs.envs.base import BaseRuleEnv


class BaseAgent(abc.ABC):
    def __init__(self, action_space):
        self.action_space = action_space

    @abc.abstractmethod
    def __call__(self):
        """Generate action"""


class RandomAgent(BaseAgent):
    def __call__(self):
        return self.action_space.sample()


def generate_text_trajectories(
    env: BaseRuleEnv, agent: BaseAgent, nb_trajectories: int
) -> List[str]:
    """Generate random trajectories for the environment."""
    trajectories = []
    for _ in range(nb_trajectories):
        new_rule = env.generate_rule()
        obs, info = env.reset(options={"rule": new_rule})
        trajectory = []
        for _ in range(env.nb_steps):
            action = agent()
            trajectory.append(env.action_to_text(action))
        trajectories.append(" ".join(trajectory))
    return trajectories
