import abc
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import gymnasium as gym


class BaseRule(abc.ABC):
    """Base Class for the rules"""

    pass


@dataclass
class EnvPromptInfo:
    """Prompting info to give to the LLM"""

    tokens: List[str]
    stat_prompt: str
    th_prompt: str
    stat_template: Callable[[str, str], str]
    th_template: Callable[[List[str]], str]


class BaseRuleEnv(gym.Env, abc.ABC):
    """Base Class for the world llm environments."""

    def __init__(self, initial_config: Dict[str, Any], **kwargs) -> None:
        self.observation_space = initial_config["observation_space"]
        self.action_space = initial_config["action_space"]
        for attr in [
            "tokens",
            "stat_prompt",
            "stat_template",
            "th_prompt",
            "th_template",
        ]:
            if attr in kwargs:
                setattr(self, attr, kwargs[attr])
            else:
                setattr(self, attr, initial_config[attr])
        self.rule: BaseRule

    def get_message_info(self):
        """Return prompting information for the theorist and statistician llms"""
        return EnvPromptInfo(
            self.tokens,
            self.stat_prompt,
            self.th_prompt,
            self.stat_template,
            self.th_template,
        )

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
        self.trajectory: str

    def action_to_text(self, action):
        return self.env.unwrapped.action_to_text(action)

    def observation_to_text(self, observation):
        return self.env.unwrapped.observation_to_text(observation)

    def generate_rule(self):
        return self.env.unwrapped.generate_rule()

    def step(self, action):
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs_text, act_text = self.observation_to_text(observation), self.action_to_text(
            action
        )
        self.trajectory += f" {act_text} {obs_text}"
        info["action_text"] = act_text
        info["trajectory"] = self.trajectory
        return (
            obs_text,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        observation, info = self.env.reset(seed=seed, options=options)
        self.trajectory = self.observation_to_text(observation)
        info["trajectory"] = self.trajectory
        return self.observation_to_text(observation), info
