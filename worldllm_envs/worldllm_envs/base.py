import abc
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym
import numpy as np


class BaseRule:
    """Base Class for the rules"""

    def __init__(self, condition_text: str) -> None:
        self.condition_text = condition_text

    def get_prompt(self) -> str:
        """get the prompt of the rule"""
        return self.condition_text


@dataclass
class EnvPromptInfo:
    """Prompting info to give to the LLM"""

    stat_prompt: str
    th_prompt: str
    stat_template: Callable[[str], str]
    th_template: Callable[[List[str], Optional[str], Optional[List[str]]], str]


class BaseRuleEnv(gym.Env, abc.ABC):
    """Base Class for the world llm environments."""

    def __init__(self, **kwargs) -> None:
        self.stat_prompt: str
        self.th_prompt: str
        self.stat_template: Callable[[str], str]
        self.th_template: Callable[[List[str], Optional[str], Optional[List[str]]], str]
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
        self.rule: BaseRule
        # Set the seed
        self.set_seed(kwargs.get("seed", None))

    def set_seed(self, seed: Optional[int]) -> None:
        """Set the seed of the environment"""
        self.observation_space.seed(seed)
        self.action_space.seed(seed)
        np.random.seed(seed)
        random.seed(seed)

    def get_message_info(self):
        """Return prompting information for the theorist and statistician llms"""
        return EnvPromptInfo(
            self.stat_prompt,
            self.th_prompt,
            self.stat_template,
            self.th_template,
        )

    @staticmethod
    @abc.abstractmethod
    def generate_rule(rule: Optional[Any]) -> BaseRule:
        """Generate Rule from argument or randomly"""

    def change_rule(self, rule: BaseRule) -> None:
        """Change the rule of the environment."""
        self.rule = rule

    @abc.abstractmethod
    def action_to_text(self, action) -> str:
        """Return text associated with the action"""

    @abc.abstractmethod
    def observation_to_text(self, observation) -> str:
        """Return text associated with the observation"""

    def get_rule(self) -> BaseRule:
        """Return the rule of the environment."""
        return self.rule

    def reset(self, seed=None, options=None):
        # We need the following line to seed self.np_random
        super().reset(seed=seed)
        if seed is not None:
            # Do not change seed if not provided
            self.set_seed(seed)
        if (options is None or "rule" not in options) and not hasattr(self, "rule"):
            raise ValueError("You must provide a rule to init the environment.")
        return self._reset(options)

    @abc.abstractmethod
    def _reset(self, options: Optional[Dict[str, Any]]) -> Tuple[Any, Dict[str, Any]]:
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
        self.text_trajectory: List[str]
        self.obs_trajectory: List[Any]

    # TODO: remove unwrapped as it takes longer to take the function
    def action_to_text(self, action):
        return self.env.unwrapped.action_to_text(action)

    def observation_to_text(self, observation):
        return self.env.unwrapped.observation_to_text(observation)

    def generate_rule(self, rule: Optional[Any] = None):
        return self.env.unwrapped.generate_rule(rule)

    def step(self, action):
        act_text = self.action_to_text(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs_text = self.observation_to_text(observation)
        self.text_trajectory.extend([act_text, obs_text])
        self.obs_trajectory.append(observation)
        info["action_text"] = act_text
        info["text_trajectory"] = self.text_trajectory
        info["obs_trajectory"] = self.obs_trajectory
        return (
            obs_text,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        text_obs = self.observation_to_text(observation)
        self.text_trajectory = [text_obs]
        self.obs_trajectory = [observation]
        info["text_trajectory"] = self.text_trajectory
        info["obs_trajectory"] = self.obs_trajectory
        return text_obs, info
