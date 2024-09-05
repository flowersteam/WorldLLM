import random
import re
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np

from utils.utils_env import BaseAgent
from worldllm_envs.base import BaseRuleEnv, TextWrapper
from worldllm_envs.playground.descriptions import generate_all_descriptions
from worldllm_envs.playground.env_params import get_env_params
from worldllm_envs.playground.playgroundnavv1 import PlayGroundNavigationV1
from worldllm_envs.playground.reward_function import (
    get_reward_from_state,
    sample_descriptions_from_state,
)


def rm_trailing_number(input_str):
    return re.sub(r"\d+$", "", input_str)


class PlayGroundText(BaseRuleEnv):  # Transformer en wrapper
    """
    PlayGroundText is a wrapper for the PlayGroundNavigation environment.
    It convert natural language commands into actions for the PlayGroud environment
    and convert observations into natural language descriptions.
    """

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

        # WorldLLM parameters
        self.observation_space = gym.spaces.Text(int(1e6))
        self.action_space = gym.spaces.Text(int(1e6))
        self.stat_prompt = ""
        self.stat_template = statisitician_template
        self.th_prompt = ""
        self.th_template = theorist_template

        # Playground parameters
        self.max_steps = 64
        self.train = True
        self.mask_growgrow = False
        self.grasp_only = False
        self.goal_sampler = None
        self.random_type = True

        super().__init__(**kwargs)
        # The playground environment
        self.playground = PlayGroundNavigationV1(**kwargs.get("playground_config", {}))

        # Dict containing all the playground environment parameters
        self.env_params = get_env_params(admissible_attributes=("categories", "types"))

        # Generate all the descriptions/goals for the environment
        train_descriptions_non_seq, test_descriptions_non_seq, _ = (
            generate_all_descriptions(self.env_params)
        )

        # Remove all the 'Go to <position>' goals
        train_descriptions_non_seq = [
            s for s in train_descriptions_non_seq if not s.startswith("Go")
        ]
        test_descriptions_non_seq = [
            s for s in test_descriptions_non_seq if not s.startswith("Go")
        ]

        self.train_descriptions = train_descriptions_non_seq.copy()
        self.test_descriptions = test_descriptions_non_seq.copy()

        # Generate sequential goals
        # for desc_a in train_descriptions_non_seq:
        #     for desc_b in train_descriptions_non_seq:
        #         if desc_a.startswith('Grow') and (desc_a.split(' ')[-1] != desc_b.split(' ')[-1] or desc_a.split(' ')[-2] != desc_b.split(' ')[-2]):
        #             self.train_descriptions.append(desc_a + ' then ' + desc_b.lower())

        # for desc_a in test_descriptions_non_seq:
        #     for desc_b in test_descriptions_non_seq:
        #         if desc_a.startswith('Grow') and (desc_a.split(' ')[-1] != desc_b.split(' ')[-1] or desc_a.split(' ')[-2] != desc_b.split(' ')[-2]):
        #             self.test_descriptions.append(desc_a + ' then ' + desc_b.lower())

        # If unseen type is True we remove "grow then grow" goals from the train set
        if self.mask_growgrow:
            self.train_descriptions = [
                desc for desc in self.train_descriptions if "then grow" not in desc
            ]

        if self.grasp_only:
            self.train_descriptions = [
                desc for desc in self.train_descriptions if desc.startswith("Grasp")
            ]

    @staticmethod
    def generate_rule(custom_rule: Optional[str] = None) -> str:
        # print("WARNING: no other rule than the default one is available")
        return "Default rule"

    def action_to_text(self, action: str):
        action_type, action_obj = self._split_action(action)
        if action_type == "go to":
            return f"You go to the {action_obj}."
        elif action_type == "grasp":
            return "You grasp the object."
        elif action_type == "release":
            return f"You release the {action_obj}."
        raise ValueError("The action " + action + " has not been recognized")

    def observation_to_text(self, observation: str):
        "Format the observation into more of a sentence"
        obs_obj, obs_stand, obs_hold = self._split_description(observation)
        output = "You see the "
        for i, obj in enumerate(obs_obj):
            output += obj
            if i == len(obs_obj) - 2:
                output += " and the "
            elif i != len(obs_obj) - 1:
                output += ", the "
            else:
                output += "."
        output += f"\nYou are standing on {obs_stand[0]}."
        if len(obs_hold) == 2:
            output += f"\nYou are holding {obs_hold[0]} and {obs_hold[1]}."
        elif len(obs_hold) == 1:
            if obs_hold[0] == "empty":
                output += "\nYour are holding nothing."
            else:
                output += f"\nYou are holding {obs_hold[0]}."
        return output

    def get_diff_description(
        self, last_observation: str, observation: str, action: str
    ) -> str:
        """Compute the difference between two observations"""
        # Split text description
        last_obs_obj, last_obs_stand, last_obs_hold = self._split_description(
            last_observation
        )
        obs_obj, obs_stand, obs_hold = self._split_description(observation)
        action_type, action_obj = self._split_action(action)
        if (
            Counter(last_obs_obj) == Counter(obs_obj)
            and Counter(last_obs_stand) == Counter(obs_stand)
            and Counter(last_obs_hold) == Counter(obs_hold)
        ):
            return "Nothing has changed."
        elif action_type == "go to":
            return f"You are standing on {action_obj}"
        elif action_type == "grasp":
            counter_diff = Counter(obs_hold) - Counter(last_obs_hold)
            assert (
                len(counter_diff) == 1
            ), "There should be only one object grasped at a time"
            if last_obs_hold[0] == "empty":
                return f"You are holding {list(counter_diff.keys())[0]}."
            if len(last_obs_hold) == 1:
                return f"You are holding {list(counter_diff.keys())[0]} and {list(counter_diff.keys())[0]}."
            raise ValueError("Inventory cannot contain more than 2 objects")
        elif action_type == "release":
            new_obj = Counter(obs_obj) - Counter(last_obs_obj)
            assert len(new_obj) == 1, "There should be only one new object emerging"
            old_obj = Counter(last_obs_obj) - Counter(obs_obj)
            return f"The {list(old_obj)[0]} transforms into the {list(new_obj)[0]}."
        raise ValueError(
            f"The difference between the two observations: \n{last_observation} \n and: \n{observation} \nis not recognized"
        )

    def _split_action(self, action: str) -> Tuple[str, str]:
        """Split the action into type and object"""
        if "go to" in action:
            return "go to", action.split("go to ")[1]
        elif "grasp" in action:
            return "grasp", ""
        elif "release" in action:
            return "release", action.split("release ")[1]
        raise ValueError("The action " + action + " has not been recognized")

    def _split_description(
        self, description: str
    ) -> Tuple[List[str], List[str], List[str]]:
        """Split the description into what you see, stand on and hold"""
        split_description = description.split("\n")
        lst_objects = split_description[0].split(": ")[1].split(", ")
        lst_standing = split_description[1].split(": ")[1].split(", ")
        lst_holding = split_description[2].split(": ")[1].split(", ")
        return lst_objects, lst_standing, lst_holding

    def _reset(self, options: Optional[Dict[str, Any]]):
        # Change the rule if new one is presented
        if options is not None and "rule" in options:
            self.rule = options["rule"]

        # Old playground reset
        self.lp = None
        self.sr = None
        self.sr_delayed = None
        self.temperature = None
        # Get goal
        if "goal_str" in options:
            self.goal_str = options["goal_str"]
        elif self.train:
            if self.goal_sampler is not None:
                self.goal_str, self.lp, self.sr, self.sr_delayed, self.temperature = (
                    self.goal_sampler.sample_goal()
                )
            else:
                lst_goal_possible = []
                for goal in self.train_descriptions:
                    lst_components = goal.split(" ")
                    if not (
                        lst_components[0] != "Grow"
                        or lst_components[2] in {"living_thing", "animal"}
                    ):
                        lst_goal_possible.append(goal)
                self.goal_str = random.choice(lst_goal_possible)
        else:
            raise NotImplementedError("Test mode not supported yet")
            # If we are in test mode, we want to test the model on unseen data
            # Sample goal uniformly for the type 'Grasp', 'Grow', 'Grow then grasp', 'Grow then grow'
            goal_type = random.choice(
                [
                    r"^Grow.*\b(lion|grizzly|shark|fox|bobcat|coyote|small_carnivorous|big_carnivorous)\b",
                    r"^Grow.*\b(elephant|giraffe|rhinoceros|pig|cow|sheep|small_herbivorous|big_herbivorous)\b",
                    r"^Grow.*\b(carrot|potato|beet|berry|pea)\b",
                ]
            )
            self.goal_str = random.choice(
                [goal for goal in self.test_descriptions if re.match(goal_type, goal)]
            )
        self.goals = self.goal_str.split(" then ")
        self.goals = [goal.capitalize() for goal in self.goals]
        self.goals_reached = [False for _ in self.goals]
        # Reset playground engine and start a new episode
        self.playground.reset()
        self.playground.reset_with_goal(self.goal_str)
        o, _, _, _, _ = self.playground.step(np.array([0, 0, 0]))  # Init step
        self.current_step = 0

        self.update_obj_info()
        observation, info = self.generate_description()

        self.hindsights_list = []  # Used to construct sequential hindsights
        self.hindsights_mem = (
            []
        )  # Used to make sure we don't repeat the same hindsights
        hindsight = self.get_hindsight(o)
        if len(hindsight) != 0:
            self.hindsights_list.extend(hindsight.copy())

        info["obj_dict"] = self.obj_dict
        info["hindsight"] = hindsight
        if self.lp is not None:
            info["lp"] = self.lp
            info["sr"] = self.sr
            info["sr_delayed"] = self.sr_delayed
            info["temperature"] = self.temperature

        self.inventory = info["inventory"]

        return observation, info

    def step(self, action_str):
        # Old playground step
        if action_str[:5].lower() == "go to":
            action = self.go_to(action_str[6:])
        elif action_str.lower() == "grasp":
            action = self.grasp()
        elif action_str[:7].lower() == "release":
            if "all" in action_str:
                release_id = 4
            else:
                obj_to_release = action_str[8:]
                if obj_to_release == self.inventory[0]:
                    release_id = 2
                else:
                    release_id = 3

            action = self.release(release_id=release_id)
        else:
            raise ValueError("The action " + action_str + " is incorrect")

        # Used for the hindsight method
        grasp = (
            action_str.lower() == "grasp"
            and self.playground.unwrapped.gripper_state != 1
        )

        # Take a step in the playgroud environment
        o, _, _, _, _ = self.playground.step(action)

        # There is a problem if you move directly to one of the objects, the state of the object is not updated
        # So we need to take a step with no action to update the state of the objects
        if action[:2].sum() != 0:  # If we moved
            o, _, _, _, _ = self.playground.step(
                np.array([0, 0, self.playground.unwrapped.gripper_state])
            )

        self.current_step += 1

        self.goals_reached[0] = (
            get_reward_from_state(o, self.goals[0], self.env_params)
            or self.goals_reached[0]
        )
        if len(self.goals_reached) == 2:
            self.goals_reached[1] = (
                get_reward_from_state(o, self.goals[1], self.env_params)
                and self.goals_reached[0]
                or self.goals_reached[1]
            )
        elif len(self.goals_reached) > 2:
            raise ValueError(
                "The number of sequential goals is greater than 2. It is not supported"
            )

        r = all(self.goals_reached)

        done = self.current_step == self.max_steps or r

        self.update_obj_info()
        observation, info = self.generate_description()

        # Gather hindsights for the current state
        hindsights = self.get_hindsight(o, grasp)
        if len(hindsights) != 0:
            seq_hindsight = [
                h2 + " then " + h1.lower()
                for h2 in self.hindsights_list
                for h1 in hindsights
                if (h2 + " then " + h1.lower()) in self.train_descriptions
            ]

            self.hindsights_list.extend(hindsights.copy())
            hindsights.extend(seq_hindsight)

        hindsights = list(set(hindsights) - set(self.hindsights_mem))

        info["hindsight"] = hindsights
        if self.lp is not None:
            info["lp"] = self.lp
            info["sr"] = self.sr
            info["sr_delayed"] = self.sr_delayed
            info["temperature"] = self.temperature

        self.hindsights_mem.extend(hindsights)

        self.inventory = info["inventory"]

        # Reset the size of obj help to find the obj grown in the current step
        self.playground.unwrapped.reset_size()

        return observation, float(r), done, False, info

    def render(self):
        if self.playground.unwrapped.render_mode == "human":
            self.playground.render()
        else:
            raise NotImplementedError(
                "To use render you have to instanciate playground using gym.make('PlaygroundNavigationRender-v1')"
            )

    # --- Utils ---

    def get_hindsight(self, o, grasp=False):
        hindsights = [
            hindsight
            for hinsights in sample_descriptions_from_state(o, self.env_params)
            for hindsight in hinsights
        ]
        return [
            hindsight
            for hindsight in hindsights
            if not hindsight.startswith("Go")
            and (grasp or not hindsight.startswith("Grasp"))
            and hindsight in self.train_descriptions
        ]

    def update_obj_info(self):
        """
        Store in a dict the position and grasped state of all the environment objects
        Ex: {'red cow': {'position': (0.1, 0.2), 'grasped': False}, ...}
        """
        self.obj_dict = {}
        i = 1
        for obj in self.playground.unwrapped.objects:
            if obj is None:
                continue
            agent_on = (
                np.linalg.norm(obj.position - self.playground.unwrapped.agent_pos)
                < (obj.size + obj.agent_size) / 2
                and not obj.grasped
            )
            category = obj.object_descr["categories"]
            if category == "plant" and not obj.grown_once:
                key = (
                    obj.object_descr["colors"]
                    + " "
                    + obj.object_descr["types"]
                    + " seed"
                )
            elif (
                "carnivorous" in category or "herbivorous" in category
            ) and not obj.grown_once:
                key = (
                    "baby "
                    + obj.object_descr["colors"]
                    + " "
                    + obj.object_descr["types"]
                )
            else:
                key = obj.object_descr["colors"] + " " + obj.object_descr["types"]
            if key not in self.obj_dict.keys():
                self.obj_dict[key] = {
                    "position": obj.position,
                    "grasped": obj.grasped,
                    "agent_on": agent_on,
                    "grown": obj.grown_once,
                    "category": category,
                }
            else:  # If there are multiple objects with the same description
                self.obj_dict[key + str(i)] = {
                    "position": obj.position,
                    "grasped": obj.grasped,
                    "agent_on": agent_on,
                    "grown": obj.grown_once,
                    "category": category,
                }
                i += 1

    def generate_description(self):
        """
        Return a natural language description of the scene
        """
        desc = "You see: "
        desc += ", ".join(
            rm_trailing_number(obj)
            for obj in self.obj_dict.keys()
            if not self.obj_dict[obj]["grasped"]
        )
        agent_on = [
            rm_trailing_number(obj)
            for obj in self.obj_dict.keys()
            if self.obj_dict[obj]["agent_on"]
        ]
        desc += (
            f'\nYou are on: {", ".join(agent_on) if len(agent_on) > 0 else "nothing"}'
        )
        obj_held = [
            rm_trailing_number(obj)
            for obj in self.obj_dict.keys()
            if self.obj_dict[obj]["grasped"]
        ]
        nb_held = 0
        for obj in self.obj_dict.keys():
            if self.obj_dict[obj]["grasped"]:
                nb_held += 1
        desc += f'\nInventory ({nb_held}/2): {", ".join(obj_held) if len(obj_held) > 0 else "empty"}'

        possible_actions = (
            ["Grasp"]
            + [
                "Go to " + rm_trailing_number(obj)
                for obj in self.obj_dict.keys()
                if not self.obj_dict[obj]["grasped"]
            ]
            + ["Release " + obj for obj in obj_held]
        )
        if len(obj_held) == 2:
            possible_actions.append("Release all")

        info = {
            "goal": self.goal_str,
            "possible_actions": possible_actions,
            "inventory": obj_held,
        }

        return desc, info

    # --- Actions ---

    def go_to(self, obj_desc):
        """
        Return the action to move to the object described by obj_desc
        """
        for obj in self.obj_dict.keys():
            if obj.startswith(obj_desc) and not self.obj_dict[obj]["grasped"]:
                target_pos = self.obj_dict[obj]["position"]
                return np.array(
                    [
                        target_pos[0] - self.playground.unwrapped.agent_pos[0],
                        target_pos[1] - self.playground.unwrapped.agent_pos[1],
                        -1,
                    ]
                )

        else:
            raise ValueError(obj_desc + " not in the environment")

    def grasp(self):
        """
        Return the action to grasp an object
        """
        return np.array([0, 0, 1])

    def release(self, release_id=-1):
        """
        Return the action to release an object
        """
        return np.array([0, 0, release_id])


class PlaygroundWrapper(TextWrapper):
    """Text wrapper for the Playground environment"""

    def __init__(self, env: BaseRuleEnv):
        super().__init__(env)
        self.last_obs = None
        self.last_action = None

    def action_to_text(self, action):
        act_text = self.env.unwrapped.action_to_text(action)
        self.last_action = act_text
        return act_text

    def observation_to_text(self, observation):
        if self.last_obs is None:
            text_obs = self.env.unwrapped.observation_to_text(observation)
            self.last_obs = observation
            return text_obs
        text_obs = self.env.unwrapped.get_diff_description(
            self.last_obs, observation, self.last_action
        )
        self.last_obs = observation
        return text_obs


class PerfectAgent(BaseAgent):
    def __init__(
        self, action_space: gym.Space, obj_dict: Dict[str, Dict[str, Any]], goal: str
    ):
        super().__init__(action_space)
        self.obj_dict = obj_dict
        self.goal = goal

        # Split goal
        goal_type, goal_obj = self.split_goal(goal)

        # Define plan
        self.dtree = PlaygroundDecisionTree(obj_dict, goal_type, goal_obj)
        self.lst_actions = self.dtree.get_plan()
        self.index = 0
        self.is_done = False

    def split_goal(self, goal: str) -> Tuple[str, str]:
        """Split the goal into objects"""
        lst_goal = goal.split(" ")
        assert len(lst_goal) == 3
        if lst_goal[0] == "Grow":
            goal_type = "grow"
        else:
            raise ValueError(f"Unrecognized {lst_goal[0]} as a goal")
        goal_obj = lst_goal[2]
        return goal_type, goal_obj

    def __call__(self, obs):
        """Take action according to plan"""
        action = self.lst_actions[self.index]
        self.index += 1
        if self.index == len(self.lst_actions):
            self.is_done = True
        return action


class PlaygroundDecisionTree:
    def __init__(
        self,
        obj_dict: Dict[str, Dict[str, Any]],
        goal_type: str,
        goal_obj: str,
    ) -> None:
        self.obj_dict = obj_dict
        self.goal_type = goal_type
        self.goal_obj = goal_obj
        self.lst_action: List[str] = []

        # Clean obj_dict
        self.category: Dict[str, Dict[str, str]] = {}
        for k, v in self.obj_dict.items():
            if v["category"] not in self.category:
                self.category[v["category"]] = OrderedDict({k: ""})
            else:
                self.category[v["category"]][k] = ""
        self.obj_cat = {k: v["category"] for k, v in self.obj_dict.items()}
        # Compute plan
        if goal_type == "grow":
            if goal_obj not in {
                "small_herbivorous",
                "big_herbivorous",
                "small_carnivorous",
                "big_carnivorous",
                "plant",
            }:
                for obj in self.obj_dict.keys():
                    if goal_obj in obj:
                        goal_obj = obj
                        break
                goal_cat = self.obj_cat[goal_obj]
            else:
                goal_cat = goal_obj
                goal_obj = None
            if goal_cat == "plant":
                _, success = self._grow_plant(goal_obj)
            elif goal_cat == "small_herbivorous":
                _, success = self._grow_herbivore(goal_obj)
            elif goal_cat == "big_herbivorous":
                _, success = self._grow_herbivore(goal_obj, big=True)
            elif goal_cat == "small_carnivorous":
                _, success = self._grow_carnivore(goal_obj)
            elif goal_cat == "big_carnivorous":
                _, success = self._grow_carnivore(goal_obj, big=True)
            else:
                raise ValueError(f"Unrecognized goal category {goal_cat}")
            if not success:
                raise ValueError(f"Could not find a plan for {goal_cat} and {goal_obj}")
        else:
            raise ValueError(f"Unrecognized goal type {goal_type}")

    def get_plan(self) -> List[str]:
        """Return list of actions to take"""
        return self.lst_action

    def _find_object(self, obj: str) -> Tuple[str, bool]:
        if obj in self.obj_cat:
            return obj, True
        return "", False

    def _find_object_category(self, category: str) -> Tuple[str, bool]:
        if category in self.category:
            obj, _ = self.category[category].popitem()
            self.category[category][obj] = ""
            return obj, True
        return "", False

    def _grow_plant(self, obj: Optional[str] = None) -> Tuple[str, bool]:
        if obj is None:
            obj, has_found = self._find_object_category("plant")
            if not has_found:
                return "", False
        water_obj, has_found = self._find_object_category("supply")
        if not has_found:
            return "", False
        self._go_to(water_obj)
        self._grasp(water_obj)
        self._go_to(obj)
        obj, _ = self._release(obj, water_obj)
        return obj, True

    def _grow_herbivore(
        self, obj: Optional[str] = None, big: bool = False
    ) -> Tuple[str, bool]:
        if obj is None:
            if big:
                obj, has_found = self._find_object_category("big_herbivorous")
                if not has_found:
                    return "", False
            else:
                obj, has_found = self._find_object_category("small_herbivorous")
                if not has_found:
                    return "", False
        plant, success = self._grow_plant()
        if not success:
            return "", False
        self._grasp(plant)
        if big:
            second_plant, success = self._grow_plant()
            if not success:
                return "", False
            self._grasp(second_plant)
        self._go_to(obj)
        if big:
            obj, _ = self._release(obj, plant, second_plant)
        else:
            obj, _ = self._release(obj, plant)
        return obj, True

    def _grow_carnivore(
        self, obj: Optional[str] = None, big: bool = False
    ) -> Tuple[str, bool]:
        if big:
            if obj is None:
                obj, has_found = self._find_object_category("big_carnivorous")
                if not has_found:
                    return "", False

            herbivorous, success = self._grow_herbivore(big=True)
            if success:
                self._grasp(herbivorous)
                self._go_to(obj)
                obj, _ = self._release(obj, herbivorous)
            else:
                first_herbivorous, first_success = self._grow_herbivore(big=False)
                if not first_success:
                    return "", False
                self._grasp(first_herbivorous)
                second_herbivorous, second_success = self._grow_herbivore(big=False)
                if not second_success:
                    return "", False
                self._grasp(second_herbivorous)
                self._go_to(obj)
                obj, _ = self._release(obj, first_herbivorous, second_herbivorous)
        else:
            if obj is None:
                obj, has_found = self._find_object_category("small_carnivorous")
                if not has_found:
                    return "", False
            herbivorous, success = self._grow_herbivore()
            if not success:
                return "", False
            self._grasp(herbivorous)
            self._go_to(obj)
            obj, _ = self._release(obj, herbivorous)
        return obj, True

    def _remove_obj(self, obj: str):
        cat = self.obj_cat[obj]
        del self.obj_cat[obj]
        del self.category[cat][obj]
        if len(self.category[cat]) == 0:
            del self.category[cat]

    # Low level actions
    def _go_to(self, obj: str):
        # Remove integers at the end of objects
        self.lst_action.append("go to " + rm_trailing_number(obj))

    def _grasp(self, obj: str):
        # Remove integers at the end of objects
        self.lst_action.append("grasp")
        self._remove_obj(obj)

    def _release(
        self, obj: str, release_obj: str, release_obj2: str = ""
    ) -> Tuple[str, bool]:
        # Remove integers at the end of objects
        if release_obj2 == "":
            self.lst_action.append("release " + rm_trailing_number(release_obj))
        else:
            self.lst_action.append("release all")
        # Modify object
        obj_cat = self.obj_cat[obj]
        if obj_cat == "plant":
            new_obj = rm_trailing_number(obj)[:-5]  # Remove "seed"
        else:
            new_obj = rm_trailing_number(obj)[5:]
        self._remove_obj(obj)
        self.obj_cat[new_obj] = "grown " + obj_cat
        if "grown " + obj_cat not in self.category:
            self.category["grown " + obj_cat] = OrderedDict({new_obj: ""})
        else:
            self.category["grown " + obj_cat][new_obj] = ""
        return new_obj, True
