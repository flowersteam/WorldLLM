import random
from enum import Enum
from typing import Any, Callable, Dict, Iterator, List, Optional, Tuple, Union

import gymnasium as gym

from worldllm_envs.base import BaseRule, BaseRuleEnv


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
    def __init__(
        self,
        condition: Callable[[Combination], bool],
        condition_text: str,
    ):
        super().__init__(condition_text=condition_text)
        self.condition = condition

    @staticmethod
    def from_combination(
        size: Optional[Sizes] = None,
        color: Optional[Colors] = None,
        shape: Optional[Shapes] = None,
    ) -> "Rule":
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

    def is_opened(self, prompt: Combination) -> bool:
        """Apply condition on combination"""
        return self.condition(prompt)

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
        def statisitician_template(rule: str, trajectory: str):
            """template given to the llm to compute the likelihood of a rule given a trajectory"""
            return (
                "You are in an environment in front of a door. You have several objects at your disposal."
                + "You have access to all combinations of the possible objects: key, card and ball, with the possible colors: red, green and blue, and the possible sizes: small, medium and large."
                + f"You know that: {rule} and {trajectory}\nDo you think the door will open ? You must answer in lower case only by saying 'opened' or 'closed'."
            )

        # When designing the template keep in mind that the text generated should be only the rule
        def theorist_template(
            trajectories: List[str],
            previous_rule: Optional[str] = None,
            worst_trajectories: Optional[List[str]] = None,
        ):
            """Template given to the theorist to sample new rules given trajectories"""
            msg = (
                "You are in environment with a door. You have several objects at your disposal."
                + "There are all the combinations of the possible objects: key, card and ball with the possible colors: red, green and blue and the possible sizes: small, medium and large."
                + "You have these information: \n"
            )
            for trajectory in trajectories:
                msg += f"{trajectory}\n"
            if previous_rule is None:
                msg += "\nFrom these, can you find the rule for the door? It should respect all the trajectories while still being as general as possible. Answer with just the rule"
            else:
                if worst_trajectories is not None:
                    msg += f"\nFrom these, can you find the rule for the door? You can take inspiration from the previous rule:'{previous_rule}' You also know that the previous rule failed the most on those trajectories:\n"
                    for trajectory in worst_trajectories:
                        msg += f"{trajectory}\n"
                    msg += "\nAnswer with just the rule."
                else:
                    msg += f"\nFrom these, can you find the rule for the door? You can take inspiration from the previous rule:'{previous_rule}' Answer with just the rule"
            return msg

        config = {
            "observation_space": gym.spaces.Discrete(2),
            "action_space": gym.spaces.MultiDiscrete([3, 3, 3]),
            "tokens": [
                "closed",
                "opened",
            ],  # List must be same size as observation_space and in same order
            "stat_prompt": "You must answer only by saying 'opened' or 'closed'.",
            "stat_template": statisitician_template,
            "th_prompt": "",
            "th_template": theorist_template,
        }

        super().__init__(config, **kwargs)

    @staticmethod
    def generate_rule(custom_rule: Optional[Union[str, List[str]]] = None) -> Rule:
        """Generate Rule from custom rule or random combination of color, size and shape."""
        if custom_rule is None:
            return Rule.from_combination(*generate_comb(2))
        if (
            isinstance(custom_rule, str)
            and getattr(CustomRules, custom_rule)
            and isinstance(getattr(CustomRules, custom_rule), Callable)
        ):
            return getattr(CustomRules, custom_rule)()
        elif isinstance(custom_rule, list):
            return Rule.from_combination(
                Sizes(custom_rule[0]) if custom_rule[0] is not None else None,
                Colors(custom_rule[1]) if custom_rule[1] is not None else None,
                Shapes(custom_rule[2]) if custom_rule[2] is not None else None,
            )
        raise ValueError(f"Rule {custom_rule} could not be found or built.")

    def action_to_text(self, action: Tuple[int, int, int]) -> str:
        combination = self.mapping_action(action)
        return combination.return_prompt()

    def observation_to_text(self, observation) -> str:
        return "The door is opened." if observation == 1 else "The door is closed."

    def mapping_action(self, action: Tuple[int, int, int]) -> Combination:
        return Combination(
            *[list(enum)[i] for enum, i in zip([Sizes, Colors, Shapes], action)]
        )

    def _reset(self, options: Optional[Dict[str, Any]]) -> Tuple[int, Dict[str, Any]]:
        # Change the rule if new one is presented
        if options is not None and "rule" in options:
            self.rule = options["rule"]
        return 0, {"rule": self.rule}

    def step(self, action):
        is_open = self.rule.is_opened(self.mapping_action(action))
        return (
            1 if is_open else 0,
            1 if is_open else 0,
            True,
            False,
            {"rule": self.rule},
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
