import abc
import json
import random
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np
from omegaconf import DictConfig, OmegaConf
from tqdm import tqdm


class BaseRule:
    """Base Class for the rules"""

    def __init__(self, condition_text: str) -> None:
        self.condition_text = condition_text

    def get_prompt(self) -> str:
        """get the prompt of the rule"""
        return self.condition_text


@dataclass
class Trajectory:
    """Save information on some rollout for the llms."""

    lst_obs: List[str]
    lst_act: List[str]
    lst_diff: List[str]

    def __len__(self):
        return len(self.lst_diff)

    def to_dict(self) -> Dict[str, List[str]]:
        """Convert the trajectory to a dictionary."""
        return {
            "obs": self.lst_obs,
            "act": self.lst_act,
            "diff": self.lst_diff,
        }

    @staticmethod
    def from_dict(traj_dict: Dict[str, List[str]]) -> "Trajectory":
        """Convert the dictionary to a trajectory."""
        return Trajectory(traj_dict["obs"], traj_dict["act"], traj_dict["diff"])


@dataclass
class EnvPromptInfo:
    """Prompting info to give to the LLM"""

    stat_prompt: str
    th_prompt: str
    exp_prompt: str
    stat_template: Callable[[str], str]
    th_template: Callable[[List[str], Optional[str], Optional[List[str]]], str]
    exp_template: Callable[[str, List[str], str, str], str]


class BaseRuleEnv(gym.Env, abc.ABC):
    """Base Class for the world llm environments."""

    def __init__(self, **kwargs) -> None:
        self.stat_prompt: str
        self.th_prompt: str
        self.exp_prompt: str
        self.stat_template: Callable[[str], Any]
        self.th_template: Callable[[List[str], Optional[str], Optional[List[str]]], str]
        self.exp_template: Callable[[str, List[str], str, str], str]
        self.test_dataset_path: Optional[str]
        self.rule: BaseRule
        self.all_transition_to_prompt: Dict[str, str]
        for attr, value in kwargs.items():
            if hasattr(self, attr):
                setattr(self, attr, value)
        # Set the seed
        self.set_seed(kwargs.get("seed", None))
        # Load the test_dataset
        if self.test_dataset_path is not None:
            self.test_dataset = self.load_test_dataset(self.test_dataset_path)

    @staticmethod
    def load_test_dataset(test_dataset_path: str) -> List[Trajectory]:
        """Load test dataset from the given path."""
        try:
            with open(test_dataset_path, "r", encoding="utf-8") as f:
                test_dataset = json.load(f)
            return [Trajectory.from_dict(traj) for traj in test_dataset]
        except FileNotFoundError as e:
            # Customize the error message and include the original error message
            msg = f"Test dataset not found at {test_dataset_path}. Generate the dataset by running the main script of the chosen environment."
            raise FileNotFoundError(msg) from e

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
            self.exp_prompt,
            self.stat_template,
            self.th_template,
            self.exp_template,
        )

    @staticmethod
    @abc.abstractmethod
    def generate_rule(rule: Optional[Any]) -> BaseRule:
        """Generate Rule from argument or randomly"""

    def change_rule(self, rule: BaseRule) -> None:
        """Change the rule of the environment."""
        self.rule = rule

    def get_all_transition_to_prompt(self) -> Set[str]:
        return set(self.all_transition_to_prompt.keys())

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


class BaseAgent(abc.ABC):
    """Base class for the agents."""

    def __init__(self, action_space: gym.Space):
        self.action_space = action_space

    @abc.abstractmethod
    def __call__(self, obs, **kwargs) -> Tuple[str, bool]:
        """Generate action and return if the agent is done."""

    def reset(self, info: Dict[str, Any]):
        """Reset the agent."""

    def generate_trajectories(
        self,
        env: BaseRuleEnv,
        nb_trajectories: int,
        reset_info: Dict[str, Any],
        n_steps: Optional[int] = None,
    ) -> Tuple[List[Trajectory], Set[str]]:
        """
        Generate text-based trajectories from the given environment.
        Args:
            env (BaseRuleEnv): The environment to generate trajectories from.
            nb_trajectories (int): The number of trajectories to generate.
            reset_info (Dict[str,Any]): Additional information to pass to the agent for reseting.
            n_steps (Optional[int], optional): Gather a number of steps instead of trajectories. Used in derived class
        Returns:
            Tuple[List[Trajectory], Set[str]]: A tuple containing a list of generated trajectories and a set of discovered transition types.
        """
        # Set rule
        lst_trajectory = []
        set_discovered_transitions = set()
        for _ in tqdm(
            range(nb_trajectories),
            desc="Generating trajectories",
            leave=False,
        ):
            obs, info = env.reset()
            info.update(reset_info)
            self.reset(info)
            done = False
            while not done:
                action, agent_done = self(obs, **info)
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


class RandomAgent(BaseAgent):
    """The agent that samples actions uniformly while respecting the action mask."""

    def __call__(self, obs, **kwargs):
        if "action_mask" in kwargs:
            action_mask = kwargs["action_mask"]
            possible_actions = np.arange(len(action_mask))[action_mask]
            return possible_actions[np.random.randint(len(possible_actions))], False
        return self.action_space.sample(), False


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

    def __call__(self, obs, **kwargs):
        if self.flat_index >= np.prod(self._arr_actions.shape[:-1]):
            raise ValueError(
                f"All actions have been sampled, lower the number of trajectories to {np.prod(self._arr_actions.shape[:-1])}."
            )
        action = np.unravel_index(self.flat_index, self._arr_actions.shape[:-1])
        self.flat_index += 1
        return action, self.flat_index >= np.prod(self._arr_actions.shape[:-1])
