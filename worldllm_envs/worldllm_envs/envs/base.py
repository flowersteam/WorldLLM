import abc
from typing import Any, Dict, Optional, Tuple

import gymnasium as gym


class BaseRule(abc.ABC):
    """Base Class for the rules"""

    pass


class BaseRuleEnv(gym.Env, abc.ABC):
    """Base Class for the world llm environments."""

    def __init__(self, initial_config: Dict[str, Any]) -> None:
        self.observation_space = initial_config["observation_space"]
        self.action_space = initial_config["action_space"]
        self.rule: BaseRule

    @staticmethod
    @abc.abstractmethod
    def generate_rule() -> BaseRule:
        """Generate Rule"""

    def change_rule(self, rule: BaseRule) -> None:
        """Change the rule of the environment."""
        self.rule = rule

    @abc.abstractmethod
    def action_to_text(self, action) -> str:
        """Return text associated with the action"""

    @abc.abstractmethod
    def observation_to_text(self, observation) -> str:
        """Return text associated with the observation"""

    @abc.abstractmethod
    def mapping_action(self, action) -> Tuple[Any, ...]:
        """Map action to the actual objects"""

    def get_rule(self) -> BaseRule:
        return self.rule

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if "rule" not in options:
            raise ValueError("You must provide a rule to reset the environment.")
        return self._reset(options)

    @abc.abstractmethod
    def _reset(self, options) -> Tuple[Any, Dict[str, Any]]:
        """Reset the environment"""

    @abc.abstractmethod
    def step(self, action):
        """Called each step to update the environment"""

    def render(self, mode="human"):
        print("Not Implemented for text environment")
        pass

    def close(self):
        pass


class TextWrapper(gym.Wrapper):
    def __init__(self, env: BaseRuleEnv):
        super().__init__(env)

    def action_to_text(self, action):
        return self.env.unwrapped.action_to_text(action)

    def observation_to_text(self, observation):
        return self.env.unwrapped.observation_to_text(observation)

    def generate_rule(self):
        return self.env.unwrapped.generate_rule()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        return (
            self.observation_to_text(observation),
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        return self.observation_to_text(observation), info
