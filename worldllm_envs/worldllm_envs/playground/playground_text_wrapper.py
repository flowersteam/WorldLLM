import random
import re
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np

from utils.utils_env import BaseAgent, Trajectory
from worldllm_envs.base import BaseRuleEnv
from worldllm_envs.playground.descriptions import generate_all_descriptions
from worldllm_envs.playground.env_params import get_env_params
from worldllm_envs.playground.playgroundnavv1 import PlayGroundNavigationV1
from worldllm_envs.playground.reward_function import (
    get_reward_from_state,
    sample_descriptions_from_state,
)


def rm_trailing_number(input_str):
    return re.sub(r"\d+$", "", input_str)


class DiverseAgent(BaseAgent):
    """Generate diverse trajectories. Just used as a flag"""

    def __call__(self, obs: str, **kwargs) -> str:
        raise NotImplementedError("DiverseAgent does not generate actions")


class RandomAgent(BaseAgent):
    """Random agent for the Playground environment"""

    def __call__(self, obs: str, **kwargs) -> str:
        """Take action according to plan"""
        return random.choice(kwargs["possible_actions"])


class PerfectAgent(BaseAgent):
    """Heuristic agent for the Playground environment"""

    def __init__(self, action_space: gym.Space):
        super().__init__(action_space)
        self.obj_dict: Dict[str, Any]
        self.goal: str
        self.dtree: PlaygroundDecisionTree
        self.lst_actions: List[str]
        self.index: int
        self.is_done: bool

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

    def __call__(self, obs: str, **kwargs) -> str:
        """Take action according to plan"""
        if getattr(self, "is_done", False) or not hasattr(self, "obj_dict"):
            raise ValueError("You need to call reset first")
        action = self.lst_actions[self.index]
        self.index += 1
        if self.index == len(self.lst_actions):
            self.is_done = True
        return action

    def reset(self, info: Dict[str, Any]):
        if not getattr(self, "is_done", False) and hasattr(self, "obj_dict"):
            raise ValueError("You need to finish the plan before resetting")
        self.obj_dict = info["obj_dict"]
        self.goal = info["goal"]
        # Split goal
        goal_type, goal_obj = self.split_goal(self.goal)

        # Define plan
        self.dtree = PlaygroundDecisionTree(self.obj_dict, goal_type, goal_obj)
        self.lst_actions = self.dtree.get_plan()
        self.index = 0
        self.is_done = False


class PlaygroundDecisionTree:
    """Decision tree to find the plan to reach a goal in the Playground environment"""

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


class PlayGroundText(BaseRuleEnv):  # Transformer en wrapper
    """
    PlayGroundText is a wrapper for the PlayGroundNavigation environment.
    It convert natural language commands into actions for the PlayGroud environment
    and convert observations into natural language descriptions.
    For this environment, the rule is the goal.
    """

    def __init__(self, **kwargs) -> None:
        def statisitician_template(trajectory: Trajectory, rule: Optional[str] = None):
            """template given to the llm to compute the likelihood of a rule given a trajectory"""
            user_prompt = (
                "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). "
                + "I can move an object, a plant or a herbivore and place it on another object to make them interact. "
            )
            if rule is not None:
                user_prompt += f"You know that: \n{rule}\n"
            user_prompt += "Your objective is to predict the next observation in the sequence given the past actions and observations. The sequence will be under this form, with x,y, z and w 4 objects and action an action:\n\n In the current space:\nYou see x, y, and z. You are standing on the y. Your are holding nothing. \na: action. \no: You are standing on x. \na: action. \no: You are holding y. \na: action. \no: You are holding y and z. \na: action. \no: x and y transform into z. \na: action. \no: x, y and z transform into w. \na: action. \no: Nothing has changed."
            # Give example trajectory
            user_prompt += "\n\nNow please complete the sequence:\n\n"
            user_prompt += "In the current space:\n"
            # Add initialisation observation and first action
            user_prompt += trajectory.text[0] + " "
            user_prompt += f"\na: {trajectory.text[1]} "
            user_prompt += "\no:"
            # Compute the prompt for the assistant
            assistant_prompt = trajectory.text[2]
            for i in range(3, len(trajectory.text)):
                if i % 2 == 1:
                    # It is an action
                    assistant_prompt += f" \na: {trajectory.text[i]}"
                else:
                    # It is an observation
                    assistant_prompt += f" \no: {trajectory.text[i]}"
            # Compute the list of tokens for the assistant
            assitant_token_lst = [trajectory.text[2]]
            for i in range(3, len(trajectory.text)):
                if i % 2 == 1:
                    # It is an action
                    assitant_token_lst.append("\na:")
                    assitant_token_lst.append(trajectory.text[i])
                else:
                    assitant_token_lst.append("\no:")
                    assitant_token_lst.append(trajectory.text[i])
            return user_prompt, assistant_prompt, assitant_token_lst

        def _format_trajectory_for_theorist(trajectory: Trajectory) -> str:
            """Format trjaectory for theorist"""
            msg = f"In the current space: {trajectory.text[0]} The sequence of actions and observations is: "
            for i in range(1, len(trajectory.text)):
                if i % 2 == 1:
                    # It is an action
                    msg += f" a: {trajectory.text[i]}"
                else:
                    # It is an observation
                    msg += f" o: {trajectory.text[i]}"
            msg += "\n"
            return msg

        # When designing the template keep in mind that the text generated should be only the rule
        def theorist_template(
            trajectories: List[Trajectory],
            previous_rule: Optional[str] = None,
            worst_trajectories: Optional[List[Trajectory]] = None,
        ):
            """Template given to the theorist to sample new rules given trajectories"""
            msg = (
                "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). "
                + "I can move an object, a plant or a herbivore and place it on another object to make them interact. "
                + "Your previous experiences were: \n\n"
            )
            for trajectory in trajectories:
                msg += _format_trajectory_for_theorist(trajectory)
            if previous_rule is None:
                msg += "\nCan you find a set of easily undestandable and concise rules to predict how the environment will change based on these trajectories? They should respect all the trajectories while still being as general as possible. Answer with just the rules."
            else:
                if worst_trajectories is not None:
                    msg += f"\nCan you find a set of easily undestandable and concise rules to predict how the environment will change based on these trajectories? You can take inspiration from the previous rules:\n{previous_rule}\nYou also know that the previous set of rules failed the most on those trajectories:\n\n"
                    for trajectory in worst_trajectories:
                        msg += _format_trajectory_for_theorist(trajectory)
                    msg += "\nAnswer with just the rules."
                else:
                    msg += f"\nCan you find a set of easily undestandable and concise rules to predict how the environment will change based on these trajectories? You can take inspiration from the previous rules:\n{previous_rule}\nAnswer with just the rules."
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

    def generate_rule(self, custom_rule: Optional[List[str]] = None) -> str:
        # print("WARNING: no other rule than the default one is available")
        if custom_rule is not None:
            return custom_rule
        if self.train:
            if self.goal_sampler is not None:
                raise ValueError("Goal sampler not supported")
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
                return random.choice(lst_goal_possible)
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

            raise ValueError("Test rule generation not implemented")

    def action_to_text(self, action: str):
        action_type, action_obj = self._split_action(action)
        if action_type == "go to":
            return f"You go to the {action_obj}."
        elif action_type == "grasp":
            return "You pick up the object."
        elif action_type == "release":
            if action_obj == "all":
                return "You give all the objects you hold."
            return f"You give the {action_obj}."
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
                output += ". "
        output += f"You are standing on {obs_stand[0]}. "
        if len(obs_hold) == 2:
            output += f"You are holding {obs_hold[0]} and {obs_hold[1]}."
        elif len(obs_hold) == 1:
            if obs_hold[0] == "empty":
                output += "Your are holding nothing."
            else:
                output += f"You are holding {obs_hold[0]}."
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
            if action_obj == "nothing":
                return "You are standing on nothing."
            return f"You are standing on the {action_obj}."
        elif action_type == "grasp":
            counter_diff = Counter(obs_hold) - Counter(last_obs_hold)
            assert (
                len(counter_diff) == 1
            ), "There should be only one object grasped at a time"
            if last_obs_hold[0] == "empty":
                return f"You are holding the {list(counter_diff.keys())[0]}."
            if len(last_obs_hold) == 1:
                return f"You are holding the {last_obs_hold[0]} and the {list(counter_diff.keys())[0]}."
            raise ValueError("Inventory cannot contain more than 2 objects")
        elif action_type == "release":
            new_obj = Counter(obs_obj) - Counter(last_obs_obj)
            assert len(new_obj) == 1, "There should be only one new object emerging"
            old_obj = Counter(last_obs_obj) - Counter(obs_obj)
            if action_obj == "all":
                return f"The {last_obs_hold[0]}, {last_obs_hold[1]} and {list(old_obj)[0]} transform into the {list(new_obj)[0]}."
            return f"The {action_obj} and {list(old_obj)[0]} transform into the {list(new_obj)[0]}."
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

        # Old playground reset
        self.lp = None
        self.sr = None
        self.sr_delayed = None
        self.temperature = None
        # Set goal as rule
        # Change the rule if new one is presented
        if options is not None and "rule" in options:
            self.rule = options["rule"]

        self.goal_str = self.rule

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
                key = obj.object_descr["types"] + " seed"
            elif (
                "carnivorous" in category or "herbivorous" in category
            ) and not obj.grown_once:
                key = "baby " + obj.object_descr["types"]
            else:
                key = obj.object_descr["types"]
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
            ["grasp"]
            + [
                "go to " + rm_trailing_number(obj)
                for obj in self.obj_dict.keys()
                if not self.obj_dict[obj]["grasped"]
            ]
            + ["release " + obj for obj in obj_held]
        )
        if len(obj_held) == 2:
            possible_actions.append("release all")

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

    @staticmethod
    def get_test_dataset() -> List[Trajectory]:
        """Return the test dataset"""
        # Small Herbivores
        trajectories = [
            Trajectory(
                [
                    "You see the baby cow, the water, the berry seed, the baby giraffe and the baby giraffe. You are standing on nothing. Your are holding nothing.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the water.",
                    "You go to the berry seed.",
                    "You are standing on the berry seed.",
                    "You give the water.",
                    "The water and berry seed transform into the berry.",
                    "You pick up the object.",
                    "You are holding the berry.",
                    "You go to the baby cow.",
                    "You are standing on the baby cow.",
                    "You give the berry.",
                    "The berry and baby cow transform into the cow.",
                ]
            )
        ]
        trajectories.append(
            Trajectory(
                [
                    "You see the baby pig, the water, the potato seed, the pea seed and the baby giraffe. You are standing on nothing. Your are holding nothing.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the water.",
                    "You go to the pea seed.",
                    "You are standing on the pea seed.",
                    "You give the water.",
                    "The water and pea seed transform into the pea.",
                    "You pick up the object.",
                    "You are holding the pea.",
                    "You go to the baby pig.",
                    "You are standing on the baby pig.",
                    "You give the pea.",
                    "The pea and baby pig transform into the pig.",
                ]
            )
        )
        trajectories.append(
            Trajectory(
                [
                    "You see the baby sheep, the water, the pea seed, the carrot seed and the berry seed. You are standing on nothing. Your are holding nothing.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the water.",
                    "You go to the berry seed.",
                    "You are standing on the berry seed.",
                    "You give the water.",
                    "The water and berry seed transform into the berry.",
                    "You pick up the object.",
                    "You are holding the berry.",
                    "You go to the baby sheep.",
                    "You are standing on the baby sheep.",
                    "You give the berry.",
                    "The berry and baby sheep transform into the sheep.",
                ]
            )
        )
        # Big Herbivores
        trajectories.append(
            Trajectory(
                [
                    "You see the baby giraffe, the water, the carrot seed, the water and the beet seed. You are standing on nothing. Your are holding nothing.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the water.",
                    "You go to the beet seed.",
                    "You are standing on the beet seed.",
                    "You give the water.",
                    "The water and beet seed transform into the beet.",
                    "You pick up the object.",
                    "You are holding the beet.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the beet and the water.",
                    "You go to the carrot seed.",
                    "You are standing on the carrot seed.",
                    "You give the water.",
                    "The water and carrot seed transform into the carrot.",
                    "You pick up the object.",
                    "You are holding the beet and the carrot.",
                    "You go to the baby giraffe.",
                    "You are standing on the baby giraffe.",
                    "You give all the objects you hold.",
                    "The carrot, beet and baby giraffe transform into the giraffe.",
                ]
            )
        )
        trajectories.append(
            Trajectory(
                [
                    "You see the baby rhinoceros, the water, the potato seed, the water and the berry seed. You are standing on nothing. Your are holding nothing.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the water.",
                    "You go to the berry seed.",
                    "You are standing on the berry seed.",
                    "You give the water.",
                    "The water and berry seed transform into the berry.",
                    "You pick up the object.",
                    "You are holding the berry.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the berry and the water.",
                    "You go to the potato seed.",
                    "You are standing on the potato seed.",
                    "You give the water.",
                    "The water and potato seed transform into the potato.",
                    "You pick up the object.",
                    "You are holding the berry and the potato.",
                    "You go to the baby rhinoceros.",
                    "You are standing on the baby rhinoceros.",
                    "You give all the objects you hold.",
                    "The potato, berry and baby rhinoceros transform into the rhinoceros.",
                ]
            )
        )
        trajectories.append(
            Trajectory(
                [
                    "You see the baby elephant, the water, the potato seed, the water and the pea seed. You are standing on nothing. Your are holding nothing.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the water.",
                    "You go to the pea seed.",
                    "You are standing on the pea seed.",
                    "You give the water.",
                    "The water and pea seed transform into the pea.",
                    "You pick up the object.",
                    "You are holding the pea.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the pea and the water.",
                    "You go to the potato seed.",
                    "You are standing on the potato seed.",
                    "You give the water.",
                    "The water and potato seed transform into the potato.",
                    "You pick up the object.",
                    "You are holding the pea and the potato.",
                    "You go to the baby elephant.",
                    "You are standing on the baby elephant.",
                    "You give all the objects you hold.",
                    "The potato, pea and baby elephant transform into the elephant.",
                ]
            )
        )
        # Random Trajectories
        trajectories.append(
            Trajectory(
                [
                    "You see the baby rhinoceros, the water, the berry seed, the water and the berry seed. You are standing on nothing. Your are holding nothing.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You go to the baby rhinoceros.",
                    "You are standing on the baby rhinoceros.",
                    "You go to the baby rhinoceros.",
                    "Nothing has changed.",
                    "You pick up the object.",
                    "You are holding the baby rhinoceros.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You go to the berry seed.",
                    "You are standing on the berry seed.",
                    "You go to the berry seed.",
                    "Nothing has changed.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the baby rhinoceros and the water.",
                    "You give the water.",
                    "Nothing has changed.",
                    "You go to the berry seed.",
                    "You are standing on the berry seed.",
                    "You give the baby rhinoceros.",
                    "Nothing has changed.",
                    "You give all the objects you hold.",
                    "Nothing has changed.",
                    "You pick up the object.",
                    "Nothing has changed.",
                    "You give the water.",
                    "The water and berry seed transform into the berry.",
                    "You go to the berry.",
                    "Nothing has changed.",
                    "You go to the berry.",
                    "Nothing has changed.",
                    "You go to the berry seed.",
                    "You are standing on the berry seed.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the baby rhinoceros and the water.",
                ]
            )
        )
        trajectories.append(
            Trajectory(
                [
                    "You see the baby rhinoceros, the water, the beet seed, the water and the carrot seed. You are standing on nothing. Your are holding nothing.",
                    "You go to the baby rhinoceros.",
                    "You are standing on the baby rhinoceros.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You go to the beet seed.",
                    "You are standing on the beet seed.",
                    "You pick up the object.",
                    "You are holding the beet seed.",
                    "You go to the baby rhinoceros.",
                    "You are standing on the baby rhinoceros.",
                    "You give the beet seed.",
                    "Nothing has changed.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You go to the water.",
                    "Nothing has changed.",
                    "You go to the water.",
                    "Nothing has changed.",
                    "You pick up the object.",
                    "You are holding the beet seed and the water.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You give the water.",
                    "Nothing has changed.",
                    "You go to the carrot seed.",
                    "You are standing on the carrot seed.",
                    "You give the beet seed.",
                    "Nothing has changed.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You give the beet seed.",
                    "The beet seed and water transform into the beet.",
                    "You pick up the object.",
                    "You are holding the water and the beet.",
                    "You go to the baby rhinoceros.",
                    "You are standing on the baby rhinoceros.",
                    "You give the beet.",
                    "Nothing has changed.",
                    "You pick up the object.",
                    "Nothing has changed.",
                ]
            )
        )
        trajectories.append(
            Trajectory(
                [
                    "You see the baby elephant, the water, the pea seed, the water and the pea seed. You are standing on nothing. Your are holding nothing.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the water.",
                    "You go to the baby elephant.",
                    "You are standing on the baby elephant.",
                    "You go to the water.",
                    "You are standing on the water.",
                    "You pick up the object.",
                    "You are holding the water and the water.",
                    "You go to the pea seed.",
                    "You are standing on the pea seed.",
                    "You give all the objects you hold.",
                    "Nothing has changed.",
                    "You pick up the object.",
                    "Nothing has changed.",
                    "You give all the objects you hold.",
                    "Nothing has changed.",
                    "You pick up the object.",
                    "Nothing has changed.",
                    "You give the water.",
                    "The water and pea seed transform into the pea.",
                    "You go to the pea seed.",
                    "You are standing on the pea seed.",
                    "You go to the baby elephant.",
                    "You are standing on the baby elephant.",
                    "You go to the pea.",
                    "You are standing on the pea.",
                    "You go to the pea seed.",
                    "You are standing on the pea seed.",
                    "You go to the pea.",
                    "You are standing on the pea.",
                    "You give the water.",
                    "Nothing has changed.",
                    "You go to the pea seed.",
                    "You are standing on the pea seed.",
                    "You give the water.",
                    "The water and pea seed transform into the pea.",
                    "You go to the pea.",
                    "Nothing has changed.",
                ]
            )
        )
        return trajectories


def generate_diverse_trajectories(env: PlayGroundText) -> List[Trajectory]:
    """Generate 2 Small Herbivores, 2 Big Herbivores and 2 Random trajectories"""
    obs, info = env.reset(options={"rule": "Grow any small_herbivorous"})
    trajectories = []
    perfect_agent = PerfectAgent(env.action_space)
    for _ in range(2):
        obs, info = env.reset()
        perfect_agent.reset(info)
        done = False
        while not done:
            action = perfect_agent(obs, **info)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        trajectories.append(Trajectory(info["text_trajectory"]))

    obs, info = env.reset(options={"rule": "Grow any big_herbivorous"})
    for _ in range(2):
        obs, info = env.reset()
        perfect_agent.reset(info)
        done = False
        while not done:
            action = perfect_agent(obs, **info)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        trajectories.append(Trajectory(info["text_trajectory"]))

    random_agent = RandomAgent(env.action_space)
    obs, info = env.reset(options={"rule": "Grow any big_herbivorous"})
    for _ in range(2):
        obs, info = env.reset()
        random_agent.reset(info)
        done = False
        while not done:
            action = random_agent(obs, **info)
            obs, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
        trajectories.append(Trajectory(info["text_trajectory"]))
    return trajectories


class PlayGroundDiscrete(PlayGroundText):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # 4 because 1 for what you see, 1 for what you stand on, 2 for what you hold
        # 13 because we have 12 objects and 1 for the empty space
        # 2 because we have 2 states for the object (evolved or not)
        # 4 because we have 4 categories of objects
        self.observation_space = gym.spaces.MultiDiscrete(
            np.ones(
                4 * 13 * 2 * 4,
            )
            * 4
        )
        self.action_space = gym.spaces.Discrete(30)

        self.types_to_id = {
            k: (i + 1) for i, k in enumerate(self.env_params["attributes"]["types"])
        }
        self.types_to_id["empty"] = 0
        self.id_to_types = {
            (i + 1): k for i, k in enumerate(self.env_params["attributes"]["types"])
        }
        self.id_to_types[0] = "empty"
        self.types_to_categories = {}
        sorted_categories = sorted(self.env_params["categories"].keys())
        for cat in sorted_categories:
            if cat == "living_thing":
                continue  # Skip living things there are just to make all work
            for k in self.env_params["categories"][cat]:
                self.types_to_categories[k] = cat

        self.categories_to_id = {}
        i = 0
        for cat in sorted_categories:
            if cat == "living_thing":
                continue
            self.categories_to_id[cat] = i
            i += 1
        # Add mask for the actions
        self.action_mask = np.ones(30, dtype=bool)
        # Keep track of the inventory
        self.inventory = []

        # WE save last observation to compute the difference
        self._last_text_obs = None

    def obj_to_index(self, incr: int, obj_name: str) -> Tuple[int, int, int, int]:
        """Return the index of the object in the observation"""
        obs_name = rm_trailing_number(obj_name)
        split_obs = obs_name.split(" ")
        if len(split_obs) == 1:
            obs_name = split_obs[0]
            is_evolved = 1
        else:
            if split_obs[0] == "baby":
                obs_name = split_obs[1]
            elif split_obs[1] == "seed":
                obs_name = split_obs[0]
            else:
                raise ValueError("The object name " + obj_name + " is not recognized")
            is_evolved = 0
        return (
            incr,
            self.types_to_id[obs_name],
            is_evolved,
            self.categories_to_id[self.types_to_categories[obs_name]],
        )

    def dict_to_feature(self, obj_dict: Dict[str, Dict[str, Any]]) -> np.ndarray:
        """Convert the object dictionary to the observation"""
        # TODO: Sort the obj dict to make sur the order is always the same
        sorted_objs = sorted(obj_dict.keys())
        self.inventory = []
        standing_object = None
        obs = np.zeros(
            (
                4,
                13,
                2,
                4,
            )
        )
        i = 0
        for obj_name in sorted_objs:
            if obj_dict[obj_name]["grasped"]:
                self.inventory.append(obj_name)
                continue  # We add it at the end
            elif obj_dict[obj_name]["agent_on"]:
                standing_object = obj_name
            index = self.obj_to_index(i, obj_name)
            obs[index] += 1
        # We leave 0 if there are no more seen objects
        i = 1
        if standing_object is not None:
            index = self.obj_to_index(i, standing_object)
            obs[index] = 1
        assert (
            len(self.inventory) <= 2
        ), "The inventory cannot contain more than 2 objects"
        i += 1
        for obj_name in self.inventory:
            index = self.obj_to_index(i, obj_name)
            obs[index] = 1
            i += 1
        return obs

    def discrete_action_to_text(self, action: int) -> str:
        """Convert the discrete action to a text action"""
        if action < 26:
            if action >= 13:
                # Evolved object
                return "go to " + self.id_to_types[action - 13]
            if self.types_to_categories[self.id_to_types[action]] in {
                "small_herbivorous",
                "big_herbivorous",
            }:
                return "go to baby " + self.id_to_types[action]
            elif self.types_to_categories[self.id_to_types[action]] == "plant":
                return "go to " + self.id_to_types[action] + " seed"
            raise ValueError(
                f"The object with category {self.types_to_categories[self.id_to_types[action]]} is not supported"
            )
        elif action == 26:
            return "grasp"
        elif action in {27, 28}:
            return "release " + self.inventory[action - 27]
        else:
            return "release all"

    def _update_action_mask(self, obs: np.ndarray) -> None:
        """Return the possible actions from given state"""
        # Check go to action
        # We don't need to take care of the category
        # We need to flip the numpy array with column then
        flatten_seen_obs = obs[0].sum(-1).flatten("F")
        self.action_mask[:26] = flatten_seen_obs >= 1
        # Check grasp action always possible
        self.action_mask[26] = True
        # Check release action
        if np.all(obs[2] == 0):
            self.action_mask[27:] = False
        else:
            self.action_mask[27] = True
            if np.all(obs[3] == 0):
                self.action_mask[28:] = False
            else:
                self.action_mask[28:] = True

    def _reset(
        self, options: Optional[Dict[str, Any]]
    ) -> Tuple[np.ndarray, Dict[str, Any]]:
        # Old playground reset
        self.lp = None
        self.sr = None
        self.sr_delayed = None
        self.temperature = None
        # Set goal as rule
        # Change the rule if new one is presented
        if options is not None and "rule" in options:
            self.rule = options["rule"]

        self.goal_str = self.rule

        self.goals = self.goal_str.split(" then ")
        self.goals = [goal.capitalize() for goal in self.goals]
        self.goals_reached = [False for _ in self.goals]
        # Reset playground engine and start a new episode
        self.playground.reset()
        self.playground.reset_with_goal(self.goal_str)
        o, _, _, _, _ = self.playground.step(np.array([0, 0, 0]))  # Init step
        self.current_step = 0

        self.update_obj_info()
        obs = self.dict_to_feature(self.obj_dict)
        # Compute the str description
        obs_desc, _ = self.generate_description()
        text_obs = self.observation_to_text(obs_desc)
        self._update_action_mask(obs)
        info = {
            "goal": self.goal_str,
            "action_mask": self.action_mask,
            "text_obs": text_obs,
        }
        self._last_text_obs = obs_desc

        return obs.flatten(), info

    def get_diff(
        self, last_observation: str, observation: str, action: str
    ) -> Tuple[str, float]:
        """Compute the difference between two observations and return reward"""
        # Split text description
        last_obs_obj, last_obs_stand, last_obs_hold = self._split_description(
            last_observation
        )
        obs_obj, obs_stand, obs_hold = self._split_description(observation)
        action_type, action_obj = self._split_action(action)
        rewards = {
            "nothing": 1.8,
            "standing": 1.4,
            "holding1": 0.8,
            "holding2": 5.5,
            "transformP": 5.0,
            "transformSH": 3.8,
            "transformBH": 11.0,
        }
        if (
            Counter(last_obs_obj) == Counter(obs_obj)
            and Counter(last_obs_stand) == Counter(obs_stand)
            and Counter(last_obs_hold) == Counter(obs_hold)
        ):
            return "Nothing has changed.", rewards["nothing"]
        elif action_type == "go to":
            if action_obj == "nothing":
                return "You are standing on nothing.", rewards["standing"]
            return f"You are standing on the {action_obj}.", rewards["standing"]
        elif action_type == "grasp":
            counter_diff = Counter(obs_hold) - Counter(last_obs_hold)
            assert (
                len(counter_diff) == 1
            ), "There should be only one object grasped at a time"
            if last_obs_hold[0] == "empty":
                return (
                    f"You are holding the {list(counter_diff.keys())[0]}.",
                    rewards["holding1"],
                )
            if len(last_obs_hold) == 1:
                return (
                    f"You are holding the {last_obs_hold[0]} and the {list(counter_diff.keys())[0]}.",
                    rewards["holding2"],
                )
            raise ValueError("Inventory cannot contain more than 2 objects")
        elif action_type == "release":
            new_obj = Counter(obs_obj) - Counter(last_obs_obj)
            assert len(new_obj) == 1, "There should be only one new object emerging"
            old_obj = Counter(last_obs_obj) - Counter(obs_obj)
            new_object_type = list(new_obj.keys())[0]
            new_object_category = self.types_to_categories[new_object_type]
            if new_object_category == "plant":
                reward = rewards["transformP"]
            elif new_object_category == "small_herbivorous":
                reward = rewards["transformSH"]
            elif new_object_category == "big_herbivorous":
                reward = rewards["transformBH"]
            else:
                raise ValueError(
                    "The category " + new_object_category + " is not supported"
                )
            if action_obj == "all":
                return (
                    f"The {last_obs_hold[0]}, {last_obs_hold[1]} and {list(old_obj)[0]} transform into the {list(new_obj)[0]}.",
                    reward,
                )
            return (
                f"The {action_obj} and {list(old_obj)[0]} transform into the {list(new_obj)[0]}.",
                reward,
            )
        raise ValueError(
            f"The difference between the two observations: \n{last_observation} \n and: \n{observation} \nis not recognized"
        )

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        action_str = self.discrete_action_to_text(action)
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
        obs = self.dict_to_feature(self.obj_dict)
        self._update_action_mask(obs)
        # Compute the str description
        obs_desc, _ = self.generate_description()
        text_obs, reward = self.get_diff(self._last_text_obs, obs_desc, action_str)
        text_action = self.action_to_text(action_str)
        info = {
            "goal": self.goal_str,
            "action_mask": self.action_mask,
            "text_obs": text_obs,
            "obs_desc": obs_desc,
            "text_action": text_action,
        }
        self._last_text_obs = obs_desc
        # Reset the size of obj help to find the obj grown in the current step
        self.playground.unwrapped.reset_size()

        return obs.flatten(), reward, done, False, info

    def render(self):
        raise NotImplementedError("Rendering is not supported for the environment")

    def get_action_mask(self) -> np.ndarray:
        return self.action_mask
