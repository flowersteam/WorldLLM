import argparse
import json
import os
import random
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Set, Tuple, Union

import gymnasium as gym
import numpy as np
from tqdm import tqdm

from worldllm_envs.base import BaseAgent, BaseRule, BaseRuleEnv, RandomAgent, Trajectory


class Sizes(Enum):
    SMALL = "small"
    MEDIUM = "medium"
    LARGE = "large"


class Colors(Enum):
    RED = "red"
    GREEN = "green"
    BLUE = "blue"


class Shapes(Enum):
    KEY = "key"
    CARD = "card"
    BALL = "ball"


class Combination:
    """Handle a combination of color, size and shape."""

    def __init__(self, size: Sizes, color: Colors, shape: Shapes):
        self.size = size
        self.color = color
        self.shape = shape

    @staticmethod
    def generate() -> "Combination":
        return Combination(*generate_comb(3))

    def __str__(self):
        return f"color: {self.color}, size: {self.size}, shape: {self.shape}"

    @staticmethod
    def from_prompt(prompt: str) -> "Combination":
        # Split prompt into words, remove the dot
        words = prompt[:-1].split(" ")
        size = Sizes(words[-3])
        color = Colors(words[-2])
        shape = Shapes(words[-1])
        return Combination(Sizes(size), Colors(color), Shapes(shape))

    @staticmethod
    def get_all() -> Iterator["Combination"]:
        """Return all possible combinations of color, size and shape.

        Yields:
            _type_: iterator of Combination
        """
        for color in Colors:
            for size in Sizes:
                for shape in Shapes:
                    yield Combination(size, color, shape)

    def return_prompt(self):
        return f"You are holding a {self.size.value} {self.color.value} {self.shape.value}."

    def __repr__(self) -> str:
        return f"Combination({self.color}, {self.size}, {self.shape})"


def generate_comb(
    number_component: int = 2,
) -> Union[
    Tuple[Optional[Sizes], Optional[Colors], Optional[Shapes]],
    Tuple[Sizes, Colors, Shapes],
]:
    """Generate a condition on the combination of color, size and shape."""
    if number_component == 0:
        return (None, None, None)
    elif number_component == 1:
        # Choose an attribute between color, size and shape
        attr_ind = random.randint(0, 2)
        attribute = [Sizes, Colors, Shapes][attr_ind]
        lst_comb = [None, None, None]
        lst_comb[attr_ind] = random.choice(list(attribute))
        return tuple(lst_comb)
    elif number_component == 2:
        # Choose an attribute between color, size and shape
        attr_ind = random.randint(0, 2)
        attribute = [Sizes, Colors, Shapes][attr_ind]
        lst_comb = [
            random.choice(list(Sizes)),
            random.choice(list(Colors)),
            random.choice(list(Shapes)),
        ]
        lst_comb[attr_ind] = None
        return tuple(lst_comb)

    elif number_component == 3:
        return (
            random.choice(list(Sizes)),
            random.choice(list(Colors)),
            random.choice(list(Shapes)),
        )
    else:
        raise ValueError("number_component must be integer between 0 and 3.")


class Rule(BaseRule):
    """Rule wrapper for the Door Environment."""

    def __init__(
        self,
        condition: Callable[[Combination], bool],
        condition_text: str,
    ):
        super().__init__(condition_text=condition_text)
        self.condition = condition

    @staticmethod
    def from_tuple(
        size: Optional[Sizes] = None,
        color: Optional[Colors] = None,
        shape: Optional[Shapes] = None,
    ) -> "Rule":
        """Generate condition from tuple of color, size and shape."""

        def condition(prompt: Combination) -> bool:
            for cond_attr, prompt_attr in zip(
                [size, color, shape], [prompt.size, prompt.color, prompt.shape]
            ):
                if cond_attr is not None and cond_attr != prompt_attr:
                    return False
            return True

        condition_text = "opens with "
        condition_text += f"{size.value} " if size is not None else ""
        condition_text += f"{color.value} " if color is not None else ""
        condition_text += f"{shape.value}." if shape is not None else "objects."
        return Rule(
            condition,
            condition_text,
        )

    def is_opened(self, combinaison: Combination) -> bool:
        """Apply condition on combination"""
        return self.condition(combinaison)

    def __repr__(self) -> str:
        return f"Rule({self.condition_text})"

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, Rule):
            return NotImplemented
        # Considered equal if prompt is the same
        return self.condition_text == other.condition_text


class DoorEnv(BaseRuleEnv):
    """Basic Door Environment."""

    def __init__(self, **kwargs) -> None:
        self.all_transition_to_prompt = {
            "opened": "The door is opened.",
            "closed": "The door is closed.",
        }

        def statisitician_template(
            trajectory: Trajectory,
            discovered_transition: Set[str],
            rule: Optional[str] = None,
        ):
            """template given to the llm to compute the likelihood of a rule given a trajectory"""
            base_user_prompt = (
                "You are in an environment in front of a door. You have several objects at your disposal. "
                + "You have access to all combinations of the possible objects: key, card and ball, with the possible colors: red, green and blue, and the possible sizes: small, medium and large. "
            )
            if rule is not None:
                base_user_prompt += f"You know that: \n{rule}\n"
            base_user_prompt += "Your objective is to predict whether the door will open or close given the object you are holding. "
            base_user_prompt += "In the current space:\n"
            base_user_prompt += trajectory.lst_act[0] + "\n"

            base_user_prompt += (
                "Do you think the door will open ? You must answer only by saying "
            )
            if "opened" in discovered_transition:
                base_user_prompt += "'The door is opened.'"
                if "closed" in discovered_transition:
                    base_user_prompt += " or 'The door is closed.'."
            if "closed" in discovered_transition:
                base_user_prompt += "'The door is closed.'."

            all_user_prompts = [base_user_prompt]
            all_assistant_prompts = [trajectory.lst_diff[0]]
            return all_user_prompts, all_assistant_prompts

        def _format_trajectory_for_theorist(trajectory: Trajectory) -> str:
            """Format trjaectory for theorist"""
            msg = (
                trajectory.lst_obs[0]
                + " "
                + trajectory.lst_act[0]
                + " "
                + trajectory.lst_diff[0]
                + "\n"
            )
            return msg

        # When designing the template keep in mind that the text generated should be only the rule
        def theorist_template(
            trajectories: List[Trajectory],
            previous_rule: Optional[str] = None,
            worst_trajectories: Optional[List[Trajectory]] = None,
        ):
            """Template given to the theorist to sample new rules given trajectories"""
            msg = (
                "You are in environment with a door. You have several objects at your disposal. "
                + "There are all the combinations of the possible objects: key, card and ball with the possible colors: red, green and blue and the possible sizes: small, medium and large. "
                + "You have these information: \n"
            )
            for trajectory in trajectories:
                msg += _format_trajectory_for_theorist(trajectory)
            if previous_rule is None:
                msg += "\nFrom these, can you find the rule for the door? It should respect all the trajectories while still being as general as possible. Answer with just the rule."
            else:
                if worst_trajectories is not None:
                    msg += f"\nFrom these, can you find the rule for the door? You can take inspiration from the previous rule:'{previous_rule}' You also know that the previous rule failed the most on those trajectories:\n"
                    for trajectory in worst_trajectories:
                        msg += _format_trajectory_for_theorist(trajectory)
                    msg += "\nAnswer with just the rule."
                else:
                    msg += f"\nFrom these, can you find the rule for the door? You can take inspiration from the previous rule:'{previous_rule}' Answer with just the rule."
            return msg

        def experimenter_template(
            obs: str,
            possible_actions: List[str],
            rule: Optional[str],
            goal: str,
        ) -> str:
            """Template given to the experimenter to ask for a new action"""
            msg = (
                "You are in environment with a door. You have several objects at your disposal."
                + "There are all the combinations of the possible objects: key, card and ball with the possible colors: red, green and blue and the possible sizes: small, medium and large."
            )
            msg += "Your objective is to pick the best combination to open the door. "
            if rule is not None:
                msg += f"You know that: \n{rule}\n"
            msg += "The possible actions are: "
            for action in possible_actions:
                msg += f"\n{action} "
            msg += " \n\nWhat is the best action to take? Answer with just the action."
            return msg

        # Default config:
        self.observation_space = gym.spaces.Text(int(1e6))
        self.action_space = gym.spaces.Text(int(1e6))
        self.stat_prompt = ""
        self.stat_template = statisitician_template
        self.th_prompt = ""
        self.th_template = theorist_template
        self.exp_prompt = ""
        self.exp_template = experimenter_template
        self.test_dataset_path = os.path.join(
            os.path.dirname(__file__), "data/test_dataset.json"
        )
        super().__init__(**kwargs)

        # DoorEnv attributes
        self.door_state_to_text = {
            0: "The door is closed.",
            1: "The door is opened.",
        }

    @staticmethod
    def generate_rule(custom_rule: Optional[Union[str, List[str]]] = None) -> Rule:
        """Generate Rule from custom rule or random combination of color, size and shape."""
        if custom_rule is None:
            return Rule.from_tuple(*generate_comb(2))
        if (
            isinstance(custom_rule, str)
            and hasattr(CustomRules, custom_rule)
            and isinstance(getattr(CustomRules, custom_rule), Callable)
        ):
            return getattr(CustomRules, custom_rule)()
        elif isinstance(custom_rule, list):
            return Rule.from_tuple(
                Sizes(custom_rule[0]) if custom_rule[0] is not None else None,
                Colors(custom_rule[1]) if custom_rule[1] is not None else None,
                Shapes(custom_rule[2]) if custom_rule[2] is not None else None,
            )
        raise ValueError(f"Rule {custom_rule} could not be found or built.")

    def action_to_text(self, action: str) -> str:
        return action

    def observation_to_text(self, observation: str):
        return observation, {}

    def mapping_action(self, action: Tuple[int, int, int]) -> Combination:
        """Map action to combination of color, size and shape."""
        return Combination(
            *[list(enum)[i] for enum, i in zip([Sizes, Colors, Shapes], action)]
        )

    def _reset(self, options: Optional[Dict[str, Any]]):
        # Change the rule if new one is presented
        if options is not None and "rule" in options:
            self.rule = options["rule"]
        return self.door_state_to_text[0], {
            "rule": self.rule,
            "possible_actions": [
                combi.return_prompt() for combi in Combination.get_all()
            ],
        }

    def step(self, action: str):
        combination = Combination.from_prompt(action)
        is_open = self.rule.is_opened(combination)
        return (
            self.door_state_to_text[1] if is_open else self.door_state_to_text[0],
            self.door_state_to_text[1] if is_open else self.door_state_to_text[0],
            True,
            False,
            {"rule": self.rule, "transition_type": "opened" if is_open else "closed"},
        )


class CustomRules:
    """List of Custom Rules to be used in the Door Environment"""

    @staticmethod
    def not_blue() -> Rule:
        return Rule(
            condition=lambda x: x.color != Colors.BLUE,
            condition_text="The door opens with every objects that isn't blue.",
        )

    @staticmethod
    def not_blue_key() -> Rule:
        return Rule(
            condition=lambda x: x.color != Colors.BLUE or x.shape != Shapes.KEY,
            condition_text="The door opens with every objects that isn't a blue key.",
        )

    @staticmethod
    def nor_green_large() -> Rule:
        return Rule(
            condition=lambda x: x.color != Colors.GREEN and x.size != Sizes.LARGE,
            condition_text="The door opens with every objects that is nor green or large.",
        )

    @staticmethod
    def all_true() -> Rule:
        return Rule(
            condition=lambda x: True,
            condition_text="The door opens with every objects.",
        )

    @staticmethod
    def all_false() -> Rule:
        return Rule(
            condition=lambda x: False,
            condition_text="The door opens with no objects.",
        )


class AllAgent(BaseAgent):
    """Agent that takes all the possible actions."""

    def __init__(self, action_space: gym.Space):
        super().__init__(action_space)
        self.action_iterator: Iterator
        self.next_action: Optional[Combination]

    def reset(self, info: Dict[str, Any]):
        self.action_iterator = Combination.get_all()
        self.next_action = next(self.action_iterator, None)

    def __call__(self, obs, **kwargs):
        action = self.next_action.return_prompt()
        self.next_action = next(self.action_iterator, None)
        return action, self.next_action is None

    def generate_trajectories(
        self,
        env: DoorEnv,
        nb_trajectories: int,
        reset_info: Dict[str, Any],
        n_steps: Optional[int] = None,
    ) -> Tuple[List[Trajectory], Set[str], List[List[str]]]:
        """
        Generate text-based trajectories from the given environment.
        Args:
            env (BaseWrapper): The environment to generate trajectories from.
            nb_trajectories (int): The number of trajectories to generate.
            reset_info (Dict[str,Any]): Additional information to pass to the agent for reseting.
            n_steps (Optional[int], optional): Gather a number of steps instead of trajectories. Used in derived class
        Returns:
            Tuple[List[Trajectory], Set[str], List[List[str]]]: A tuple containing a list of generated trajectories, a set of discovered transition types and the collected transition.
        """
        # Set rule
        self.reset(reset_info)
        lst_trajectory = []
        lst_transitions = []
        for _ in tqdm(
            range(nb_trajectories),
            desc="Generating trajectories",
            leave=False,
        ):
            lst_transitions_episode = []
            obs, info = env.reset()
            info.update(reset_info)
            done = False
            while not done:
                action, agent_done = self(obs, **info)
                obs, _, terminated, truncated, info = env.step(action)
                lst_transitions_episode.append(info["transition_type"])
                done = terminated or truncated or agent_done
            lst_trajectory.append(
                Trajectory(
                    info["trajectory_obs_text"],
                    info["trajectory_act_text"],
                    info["trajectory_diff_text"],
                )
            )
            lst_transitions.append(lst_transitions_episode)
        return (
            lst_trajectory,
            set(
                [
                    transition
                    for transition_episode in lst_transitions
                    for transition in transition_episode
                ]
            ),
            lst_transitions,
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument(
        "--filepath",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data/test_dataset.json"),
        help="Path to save the dataset",
    )

    # Argument for the environment
    parser.add_argument(
        "--nb-trajectories",
        type=int,
        default=27,
        help="Number of trajectories for the dataset.",
    )
    parser.add_argument(
        "--rule",
        type=str,
        default="not_blue",
        help="The rule that the environment follows.",
    )
    args = parser.parse_args()

    env: BaseWrapper = gym.make(
        "worldllm_envs/Door-v0",
        **{
            "seed": args.seed,
            "test_dataset_path": None,
        },
    )
    # Set the seed
    np.random.seed(args.seed)
    random.seed(args.seed)
    all_agent = AllAgent(env.action_space)
    env.reset(options={"rule": env.generate_rule(args.rule)})
    # Collect all trajectories
    trajectories: List[Trajectory] = []
    new_trajectories, new_discovered_transitions, lst_transition = (
        all_agent.generate_trajectories(
            env,
            args.nb_trajectories,
            {},
            0,
        )
    )
    trajectories.extend(new_trajectories)
    # Save the trajectories
    with open(args.filepath, "w", encoding="utf-8") as f:
        json.dump([t.to_dict() for t in trajectories], f)

    print("Done.")
