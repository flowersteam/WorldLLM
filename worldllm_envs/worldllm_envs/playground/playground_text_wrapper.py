"""Main file for the Playground environment. Can be launched to generate a dataset"""

import argparse
import json
import os
import random
import re
from collections import Counter, OrderedDict
from typing import Any, Dict, List, Optional, Set, Tuple

import gymnasium as gym
import numpy as np

from worldllm_envs.base import (
    BaseAgent,
    BaseRuleEnv,
    BaseWrapper,
    RandomAgent,
    Trajectory,
)
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

    def __init__(self, action_space: gym.Space):
        super().__init__(action_space)
        self.perfect_agent_sh = PerfectAgent(
            action_space, curriculum_goals=["Grow any small_herbivorous"]
        )
        self.perfect_agent_shbh = PerfectAgent(
            action_space,
            curriculum_goals=[
                "Grow any small_herbivorous then grow any big_herbivorous"
            ],
        )
        self.random_agent = RandomAgent(action_space)

    def __call__(self, obs: str, **kwargs):
        """Act as random agent"""
        return self.random_agent(obs, **kwargs), False

    def reset(self, info: Dict[str, Any]):
        self.perfect_agent_sh.reset(info)
        self.perfect_agent_shbh.reset(info)
        self.random_agent.reset(info)

    def generate_trajectories(
        self,
        env: BaseWrapper,
        nb_trajectories: int,
        reset_info: Dict[str, Any],
        n_steps: Optional[int] = None,
    ):
        """Generate (n_traj-1)//3 Small Herbivores, (n_traj)//3 Big Herbivores and (n_tra+1)j//3 Random trajectories"""
        trajectories = []
        lst_transitions = []
        set_discovered_transition = set()
        for incr, agent in enumerate(
            [self.perfect_agent_sh, self.perfect_agent_shbh, self.random_agent]
        ):
            new_trajectories, new_discovered_transitions, new_transitions = (
                agent.generate_trajectories(
                    env,
                    (nb_trajectories + incr) // 3,
                    {"pipeline_progression": 0},
                    0,
                )
            )
            trajectories.extend(new_trajectories)
            set_discovered_transition.update(new_discovered_transitions)
            lst_transitions.extend(new_transitions)
        return trajectories, set_discovered_transition, lst_transitions


class PerfectAgent(BaseAgent):
    """Heuristic agent for the Playground environment, able to solve all goals"""

    def __init__(
        self, action_space: gym.Space, curriculum_goals: Optional[List[str]] = None
    ):
        super().__init__(action_space)
        self.obj_dict: Dict[str, Any]
        self.goal: str
        self.dtree: PlaygroundDecisionTree
        self.lst_actions: List[str]
        self.index: int
        self.is_done: bool
        self.curriculum_goals = curriculum_goals

    def split_goal(self, goal: str) -> Tuple[List[str], List[str]]:
        """Split the goal into objects"""
        lst_goal_type = []
        lst_goal_obj = []
        lst_goal = goal.split(" ")
        if lst_goal[0].lower() == "grow":
            lst_goal_type.append("grow")
            lst_goal_obj.append(lst_goal[2])
        else:
            raise ValueError(f"Unrecognized {lst_goal[0]} as a goal")
        if len(lst_goal) >= 7 and lst_goal[3] == "then":
            if lst_goal[4].lower() == "grow":
                lst_goal_type.append("grow")
                lst_goal_obj.append(lst_goal[6])
            else:
                raise ValueError(f"Unrecognized {lst_goal[4]} as a goal")

        return lst_goal_type, lst_goal_obj

    def __call__(self, obs: str, **kwargs) -> Tuple[str, bool]:
        """Take action according to plan"""
        if getattr(self, "is_done", False) or not hasattr(self, "obj_dict"):
            raise ValueError("You need to call reset first")
        action = self.lst_actions[self.index]
        self.index += 1
        if self.index == len(self.lst_actions):
            self.is_done = True
        return action, self.is_done

    def reset(self, info: Dict[str, Any]):
        """Compute plan to reach goal"""
        if not getattr(self, "is_done", False) and hasattr(self, "obj_dict"):
            raise ValueError("You need to finish the plan before resetting")
        self.obj_dict = info["obj_dict"]
        if self.curriculum_goals is None:
            self.goal = info["goal"]  # Solve everything
        else:
            self.goal = self.curriculum_goals[
                int(info["pipeline_progression"] * len(self.curriculum_goals) - 1e-4)
            ]

        # Split goal
        lst_goal_type, lst_goal_obj = self.split_goal(self.goal)

        # Define plan
        self.dtree = PlaygroundDecisionTree(self.obj_dict, lst_goal_type, lst_goal_obj)
        self.lst_actions = self.dtree.get_plan()
        self.index = 0
        self.is_done = False


class PlaygroundDecisionTree:
    """Decision tree to find the plan to reach a goal in the Playground environment"""

    def __init__(
        self,
        obj_dict: Dict[str, Dict[str, Any]],
        lst_goal_type: List[str],
        lst_goal_obj: List[str],
    ) -> None:
        self.obj_dict = obj_dict
        self.lst_goal_type = lst_goal_type
        self.lst_goal_obj = lst_goal_obj
        self.lst_action: List[str] = []

        # Clean obj_dict
        self.category: Dict[str, Dict[str, str]] = {}
        for k, v in self.obj_dict.items():
            if v["category"] not in self.category:
                self.category[v["category"]] = OrderedDict({k: ""})
            else:
                self.category[v["category"]][k] = ""
        self.obj_cat = {k: v["category"] for k, v in self.obj_dict.items()}
        for goal_type, goal_obj in zip(self.lst_goal_type, self.lst_goal_obj):
            # Compute plan
            if goal_type == "grow":
                if goal_obj in {
                    "small_herbivorous",
                    "big_herbivorous",
                    "small_carnivorous",
                    "big_carnivorous",
                    "plant",
                }:
                    goal_cat = goal_obj
                    goal_obj = None
                else:
                    for obj in self.obj_dict.keys():
                        if goal_obj in obj:
                            goal_obj = obj
                            break
                    goal_cat = self.obj_cat[goal_obj]
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
                    raise ValueError(
                        f"Could not find a plan for {goal_cat} and {goal_obj}"
                    )
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
        # Define all_transitions
        self.all_transition_to_prompt = {
            "standing": self.transition_type_to_observation("standing", ["x"]),
            "holding1": self.transition_type_to_observation("holding1", ["y"]),
            "holding2": self.transition_type_to_observation("holding2", ["y", "z"]),
            "transformP": self.transition_type_to_observation(
                "transformP", ["x", "y", "w"]
            ),
            "transformBH": self.transition_type_to_observation(
                "transformBH", ["x", "y", "z", "w"]
            ),
        }

        def statisitician_template(
            trajectory: Trajectory,
            discovered_transition: Set[str],
            rule: Optional[str] = None,
        ):
            """template given to the llm to compute the likelihood of a rule given a trajectory"""
            base_user_prompt = (
                "You are in an environment that contains multiple objects. It can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). "
                + "You can move an object, a plant or a herbivore and place it on another object to make them interact. "
            )
            if rule is not None:
                base_user_prompt += f"You know that: \n{rule}\n"
            base_user_prompt += "Your objective is to predict the next change in the environment given the state of the environment and the action taken. "
            base_user_prompt += "The last state was: \n"

            all_user_prompts = []
            all_assistant_prompts = []
            for incr in range(len(trajectory.lst_obs) - 1):
                user_prompt = base_user_prompt
                # Add initialisation observation and first action
                user_prompt += trajectory.lst_obs[incr] + " "
                user_prompt += f"\nThe action was: {trajectory.lst_act[incr]} "
                user_prompt += "\nThe change is:"
                # Compute the prompt for the assistant
                assistant_prompt = trajectory.lst_diff[incr]
                all_user_prompts.append(user_prompt)
                all_assistant_prompts.append(assistant_prompt)
            return all_user_prompts, all_assistant_prompts

        def _format_trajectory_for_theorist(trajectory: Trajectory) -> str:
            """Format trjaectory for theorist"""
            msg = f"In the current space: {trajectory.lst_obs[0]}. \nThe sequence of actions and observations is: "
            for i, diff in enumerate(trajectory.lst_diff):
                msg += f" a: {trajectory.lst_act[i]}"
                msg += f" o: {diff}"
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
                "You are in an environment that contains multiple objects. It can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). "
                + "You can move an object, a plant or a herbivore and place it on another object to make them interact. "
                + "Your previous experiences were: \n\n"
            )
            for trajectory in trajectories:
                msg += _format_trajectory_for_theorist(trajectory)
            if previous_rule is None:
                msg += "\nCan you find a set of easily understandable and concise rules to predict how the environment will change based on these trajectories? They should respect all the trajectories while still being as general as possible. Answer with just the rules."
            else:
                if worst_trajectories is not None:
                    msg += f"\nCan you find a set of easily understandable and concise rules to predict how the environment will change based on these trajectories? You can take inspiration from the previous rules:\n{previous_rule}\nYou also know that the previous set of rules failed the most on those trajectories:\n\n"
                    for trajectory in worst_trajectories:
                        msg += _format_trajectory_for_theorist(trajectory)
                    msg += "\nAnswer with just the rules."
                else:
                    msg += f"\nCan you find a set of easily understandable and concise rules to predict how the environment will change based on these trajectories? You can take inspiration from the previous rules:\n{previous_rule}\nAnswer with just the rules."
            return msg

        def experimenter_template(
            obs: str,
            possible_actions: List[str],
            rule: Optional[str],
            goal: str,
        ) -> str:
            """Template given to the experimenter to ask for a new action"""
            msg = (
                "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). "
                + "I can move an object, a plant or a herbivore and place it on another object to make them interact. "
            )
            msg += "Your objective is to take the best action given the past actions and observations. "
            if rule is not None:
                msg += f"You know that: \n{rule}\n"
            msg += "The goal is to " + goal + ". "
            msg += "The possible actions are: "
            for action in possible_actions:
                msg += f"\n{action} "
            msg += "\n\nIn the current space:\n"
            # Add initialisation observation and first action
            msg += obs
            msg += " \n\nWhat is the best action to take? Answer with just the action."
            return msg

        # WorldLLM parameters
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
        self.types_to_categories = {}
        sorted_categories = sorted(self.env_params["categories"].keys())
        for cat in sorted_categories:
            if cat == "living_thing":
                continue  # Skip living things there are just to make all work
            for k in self.env_params["categories"][cat]:
                self.types_to_categories[k] = cat

    def generate_rule(self, custom_rule: Optional[str] = None) -> str:
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

    # --- Convert environment logic to text ---

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

    def transition_type_to_observation(
        self, transition_type: str, objects: List[str]
    ) -> str:
        """Create observation from transition type and objects"""
        transition_type_to_obs = {
            "standing": "You are standing on the {0}.",
            "holding1": "You are holding the {0}.",
            "holding2": "You are holding the {0} and the {1}.",
            "transformBH": "The {3} appears from the transformation.",
            "transformP": "The {2} appears from the transformation.",
            "transformSH": "The {2} appears from the transformation.",
        }
        empty_object_alternative = {
            "standing": "You are standing on nothing.",
            "holding1": "You are holding nothing.",
        }
        if len(objects) > 0 or transition_type not in empty_object_alternative:
            return transition_type_to_obs[transition_type].format(*objects)
        return empty_object_alternative[transition_type]

    def observation_to_text(self, observation: str):
        "Format the observation into more of a sentence"
        obs_obj, obs_stand, obs_hold = self._split_description(observation)
        output = "You can see in the room the following objects: the "
        for i, obj in enumerate(obs_obj):
            output += obj
            if i == len(obs_obj) - 2:
                output += " and the "
            elif i != len(obs_obj) - 1:
                output += ", the "
            else:
                output += ". "
        output += self.transition_type_to_observation("standing", [obs_stand[0]]) + " "
        if len(obs_hold) == 2:
            output += self.transition_type_to_observation("holding2", obs_hold)
        elif len(obs_hold) == 1:
            if obs_hold[0] == "empty":
                output += self.transition_type_to_observation("holding1", [])
            else:
                output += self.transition_type_to_observation("holding1", obs_hold)
        return output, {}

    def get_diff(
        self, last_observation: str, observation: str, action: str
    ) -> Tuple[str, str]:
        """Compute the difference between two observations and return reward"""
        # Split text description
        last_obs_obj, last_obs_stand, last_obs_hold = self._split_description(
            last_observation
        )
        obs_obj, obs_stand, obs_hold = self._split_description(observation)
        action_type, action_obj = self._split_action(action)
        if action_type == "go to":
            if action_obj == "nothing":
                return (
                    self.transition_type_to_observation("standing", []),
                    "standing",
                )
            return (
                self.transition_type_to_observation("standing", [action_obj]),
                "standing",
            )
        elif action_type == "grasp":
            counter_diff = Counter(obs_hold) - Counter(last_obs_hold)
            assert (
                len(counter_diff) == 1
            ), "There should be only one object grasped at a time"
            if last_obs_hold[0] == "empty":
                return (
                    self.transition_type_to_observation(
                        "holding1", [list(counter_diff.keys())[0]]
                    ),
                    "holding1",
                )
            if len(last_obs_hold) == 1:
                return (
                    self.transition_type_to_observation(
                        "holding2", [last_obs_hold[0], list(counter_diff.keys())[0]]
                    ),
                    "holding2",
                )
            raise ValueError("Inventory cannot contain more than 2 objects")
        elif action_type == "release":
            new_obj = Counter(obs_obj) - Counter(last_obs_obj)
            assert len(new_obj) == 1, "There should be only one new object emerging"
            old_obj = Counter(last_obs_obj) - Counter(obs_obj)
            new_object_type = list(new_obj.keys())[0]
            new_object_category = self.types_to_categories[new_object_type]
            if new_object_category == "plant":
                transition_type = "transformP"
            elif new_object_category == "small_herbivorous":
                transition_type = "transformSH"
            elif new_object_category == "big_herbivorous":
                transition_type = "transformBH"
            else:
                raise ValueError(
                    "The category " + new_object_category + " is not supported"
                )
            if action_obj == "all":
                return (
                    self.transition_type_to_observation(
                        transition_type,
                        [
                            last_obs_hold[0],
                            last_obs_hold[1],
                            list(old_obj)[0],
                            list(new_obj)[0],
                        ],
                    ),
                    transition_type,
                )
            return (
                self.transition_type_to_observation(
                    transition_type, [action_obj, list(old_obj)[0], list(new_obj)[0]]
                ),
                transition_type,
            )
        raise ValueError(
            f"The difference between the two observations: \n{last_observation} \n and: \n{observation} \nis not recognized"
        )

    def _split_action(self, action: str) -> Tuple[str, str]:
        """Split the action into type and object"""
        action = action.lower()
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
        split_description = description.split(".")
        lst_objects = [
            elem.split(",")[0] for elem in split_description[0].split(" the ")[1:]
        ]
        if len(split_description) == 4:
            # There is a standing
            lst_standing = [split_description[1].split(" the ")[-1]]
            index_holding = 2
        else:
            # There is no standing
            lst_standing = ["nothing"]
            index_holding = 1
        if "nothing" in split_description[index_holding]:
            lst_holding = ["empty"]
        else:
            lst_holding = re.split(
                r" the | and| in ", split_description[index_holding]
            )[1::2]
        return lst_objects, lst_standing, lst_holding

    # --- Main gym functions---

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

        info["success"] = r
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
        desc = "You see the "
        desc += ", the ".join(
            rm_trailing_number(obj)
            for obj in self.obj_dict.keys()
            if not self.obj_dict[obj]["grasped"]
        )
        desc += ". "
        agent_on = [
            obj for obj in self.obj_dict.keys() if self.obj_dict[obj]["agent_on"]
        ]
        assert len(agent_on) <= 1, "In this environment, you can only stand on 1 object"
        if len(agent_on) == 1:
            desc += f"You are next to the {rm_trailing_number(agent_on[0])}. "
        obj_held = [
            (obj) for obj in self.obj_dict.keys() if self.obj_dict[obj]["grasped"]
        ]
        nb_held = 0
        for obj in self.obj_dict.keys():
            if self.obj_dict[obj]["grasped"]:
                nb_held += 1
        if len(obj_held) == 0:
            desc += "You have nothing in your inventory."
        else:
            desc += f'You have the {" and the ".join([rm_trailing_number(o_held) for o_held in obj_held])} in your inventory.'

        # Create a list of possible actions
        possible_actions = []
        # Check for Grasp
        if len(obj_held) < 2 and len(agent_on) > 0:
            possible_actions.append("Grasp")
        # Add Go to
        possible_actions.extend(
            [
                "Go to " + rm_trailing_number(obj)
                for obj in self.obj_dict.keys()
                if not self.obj_dict[obj]["grasped"]
            ]
        )
        # Check Release
        if len(obj_held) >= 1 and len(agent_on) >= 1:
            categories = {
                "water": [],
                "plant_seed": [],
                "plant": [],
                "baby_sh": [],
                "baby_bh": [],
            }
            agent_on_category = None
            for obj in agent_on + obj_held:
                if rm_trailing_number(obj) == "water":
                    category_to_add = "water"
                elif self.obj_dict[obj]["category"] == "plant":
                    if self.obj_dict[obj]["grown"]:
                        category_to_add = "plant"
                    else:
                        category_to_add = "plant_seed"
                elif (
                    self.obj_dict[obj]["category"] == "small_herbivorous"
                    and not self.obj_dict[obj]["grown"]
                ):
                    category_to_add = "baby_sh"
                elif (
                    self.obj_dict[obj]["category"] == "big_herbivorous"
                    and not self.obj_dict[obj]["grown"]
                ):
                    category_to_add = "baby_bh"
                else:
                    continue
                categories[category_to_add].append(obj)
                if obj == agent_on[0]:
                    agent_on_category = category_to_add

            if len(categories["water"]) >= 1 and len(categories["plant_seed"]) >= 1:
                if agent_on_category == "water":
                    for plant_seed in categories["plant_seed"]:
                        possible_actions.append(
                            f"Release {rm_trailing_number(plant_seed)}"
                        )
                elif agent_on_category == "plant_seed":
                    possible_actions.append("Release water")
            if len(categories["plant"]) >= 1 and len(categories["baby_sh"]) >= 1:
                if agent_on_category == "plant":
                    for baby_sh in categories["baby_sh"]:
                        possible_actions.append(
                            f"Release {rm_trailing_number(baby_sh)}"
                        )
                elif agent_on_category == "baby_sh":
                    for plant in categories["plant"]:
                        possible_actions.append(f"Release {rm_trailing_number(plant)}")
            if len(categories["plant"]) >= 2 and len(categories["baby_bh"]) >= 1:
                possible_actions.append("Release all")

        info = {
            "goal": self.goal_str,
            "possible_actions": possible_actions,
            "inventory": [rm_trailing_number(obj) for obj in obj_held],
        }

        return desc, info

    def get_all_possible_transitions(self) -> Tuple[List[str], List[str], List[bool]]:
        """Return all possible next observations from current observation"""
        # Split text description
        all_objects = []
        seen_objects = []
        all_seed_plant = []
        all_mature_plant = []
        all_baby_small_herbivorous = []
        all_baby_big_herbivorous = []
        grasped_objects = []
        standing_object = []
        for obj, obj_info in self.obj_dict.items():
            obj = rm_trailing_number(obj)
            if obj_info["grasped"]:
                grasped_objects.append(obj)
            elif obj_info["agent_on"]:
                standing_object.append(obj)
            else:
                seen_objects.append(obj)
            all_objects.append(obj)
            if obj_info["category"] == "plant":
                if obj_info["grown"]:
                    all_mature_plant.append(obj)
                else:
                    all_seed_plant.append(obj)
            elif obj_info["category"] == "small_herbivorous" and not obj_info["grown"]:
                all_baby_small_herbivorous.append(obj)
            elif obj_info["category"] == "big_herbivorous" and not obj_info["grown"]:
                all_baby_big_herbivorous.append(obj)

        all_transitions = []
        all_transitions_type = []
        possible_transitions_mask = []
        # Add all possible standing transitions
        all_transitions.append(self.transition_type_to_observation("standing", []))
        all_transitions_type.append("standing")
        possible_transitions_mask.append(False)
        for obj in all_objects:
            all_transitions.append(
                self.transition_type_to_observation("standing", [obj])
            )
            all_transitions_type.append("standing")
            possible_transitions_mask.append(obj in seen_objects + standing_object)
        # Add all holding1 transitions
        for obj in all_objects:
            all_transitions.append(
                self.transition_type_to_observation("holding1", [obj])
            )
            all_transitions_type.append("holding1")
            if obj in standing_object and len(grasped_objects) == 0:
                possible_transitions_mask.append(True)
            else:
                possible_transitions_mask.append(False)
        # Add all holding2 transitions
        for i, obj1 in enumerate(all_objects):
            for j in range(i + 1, len(all_objects)):
                all_transitions.append(
                    self.transition_type_to_observation(
                        "holding2", [obj1, all_objects[j]]
                    )
                )
                all_transitions.append(
                    self.transition_type_to_observation(
                        "holding2", [all_objects[j], obj1]
                    )
                )
                all_transitions_type.append("holding2")
                all_transitions_type.append("holding2")
                if grasped_objects == [obj1] and all_objects[j] in standing_object:
                    possible_transitions_mask.extend([True, False])
                elif grasped_objects == [all_objects[j]] and obj1 in standing_object:
                    possible_transitions_mask.extend([False, True])
                else:
                    possible_transitions_mask.extend([False, False])

        # Add all transformP transitions
        for seed in all_seed_plant:
            new_name = seed[:-5]
            all_transitions.append(
                self.transition_type_to_observation(
                    "transformP", ["water", seed, new_name]
                )
            )
            all_transitions.append(
                self.transition_type_to_observation(
                    "transformP", [seed, "water", new_name]
                )
            )
            all_transitions_type.append("transformP")
            all_transitions_type.append("transformP")
            if "water" in grasped_objects and seed in standing_object:
                possible_transitions_mask.extend([True, False])
            elif seed in grasped_objects and "water" in standing_object:
                possible_transitions_mask.extend([False, True])
            else:
                possible_transitions_mask.extend([False, False])

        # Add all transformSH transitions
        for baby in all_baby_small_herbivorous:
            new_name = baby[5:]
            for mature_plant in all_mature_plant:
                all_transitions.append(
                    self.transition_type_to_observation(
                        "transformSH", [baby, mature_plant, new_name]
                    )
                )
                all_transitions.append(
                    self.transition_type_to_observation(
                        "transformSH", [mature_plant, baby, new_name]
                    )
                )
                all_transitions_type.extend(["transformSH", "transformSH"])
                if baby in grasped_objects and mature_plant in standing_object:
                    possible_transitions_mask.extend([True, False])
                elif mature_plant in grasped_objects and baby in standing_object:
                    possible_transitions_mask.extend([False, True])
                else:
                    possible_transitions_mask.extend([False, False])
        # Add all transformBH transitions
        for baby in all_baby_big_herbivorous:
            new_name = baby[5:]
            for i, mature_plant1 in enumerate(all_mature_plant):
                for j, mature_plant2 in enumerate(all_mature_plant):
                    if i != j:
                        all_transitions.append(
                            self.transition_type_to_observation(
                                "transformBH",
                                [baby, mature_plant1, mature_plant2, new_name],
                            )
                        )
                        all_transitions.append(
                            self.transition_type_to_observation(
                                "transformBH",
                                [mature_plant1, baby, mature_plant2, new_name],
                            )
                        )
                        all_transitions.append(
                            self.transition_type_to_observation(
                                "transformBH",
                                [mature_plant1, mature_plant2, baby, new_name],
                            )
                        )
                        all_transitions_type.extend(
                            ["transformBH", "transformBH", "transformBH"]
                        )
                        if len(grasped_objects) == 2:
                            if (
                                baby == grasped_objects[0]
                                and mature_plant1 == grasped_objects[1]
                                and mature_plant2 in standing_object
                            ):
                                possible_transitions_mask.extend([True, False, False])
                            elif (
                                mature_plant1 == grasped_objects[0]
                                and baby == grasped_objects[1]
                                and mature_plant2 in standing_object
                            ):
                                possible_transitions_mask.extend([False, True, False])
                            elif (
                                baby in standing_object
                                and [mature_plant1, mature_plant2] == grasped_objects
                            ):
                                possible_transitions_mask.extend([False, False, True])
                            else:
                                possible_transitions_mask.extend([False, False, False])
                        else:
                            possible_transitions_mask.extend([False, False, False])
        unique_index = np.unique(all_transitions, return_index=True)[1]
        # Mask needs to be true where at least one of the possible transitions is true before removing duplicates
        possible_transitions_mask = np.array(possible_transitions_mask)
        for transition in [all_transitions[index] for index in unique_index]:
            indices = np.where(np.array(all_transitions) == transition)[0]
            possible_transitions_mask[indices] = np.any(
                possible_transitions_mask[indices]
            )
        return (
            [all_transitions[index] for index in unique_index],
            [all_transitions_type[index] for index in unique_index],
            [possible_transitions_mask[index] for index in unique_index],
        )

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


class PlayGroundDiscrete(PlayGroundText):
    """
    PlayGroundDiscrete is a discrete version of the PlayGroundText environment to be controled with a classic RL agent.
    The difference is only on the state and action space format.
    TThe transition id is returned as the reward. The real reward is computed after.

    Args:
        PlayGroundText (BaseRuleEnv): The base environment class.

    Raises:
        ValueError: If an unsupported object category is encountered.
        ValueError: If an unsupported action is encountered.
        ValueError: If the number of sequential goals is greater than 2.
        NotImplementedError: If rendering is attempted.
    """

    TRANSITION_TYPE_TO_ID = {
        "nothing": 0,
        "standing": 1,
        "holding1": 2,
        "holding2": 3,
        "transformP": 4,
        "transformSH": 5,
        "transformBH": 6,
    }

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
        self.obj_inventory = []

        # WE save last observation to compute the difference
        self._last_text_obs = None
        self.trajectory_obs_text = []
        self.trajectory_act_text = []
        self.trajectory_diff_text = []

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
        self.obj_inventory = []
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
                self.obj_inventory.append(obj_name)
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
            len(self.obj_inventory) <= 2
        ), "The inventory cannot contain more than 2 objects"
        i += 1
        for obj_name in self.obj_inventory:
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
            return "release " + rm_trailing_number(self.obj_inventory[action - 27])
        else:
            return "release all"

    def text_action_to_discrete(self, action: str) -> int:
        action, obj = self._split_action(action)
        if action == "go to":
            if obj == "nothing":
                return 13
            elif "seed" in obj:
                return self.types_to_id[obj[:-5]]
            elif "baby" in obj:
                return self.types_to_id[obj[5:]]
            return self.types_to_id[obj] + 13
        elif action == "grasp":
            return 26
        elif action == "release":
            if obj == "all":
                return 29
            return [rm_trailing_number(obj_i) for obj_i in self.obj_inventory].index(
                obj
            ) + 27
        raise ValueError("The action " + action + " is not recognized")

    def _update_action_mask(self, possible_actions: np.ndarray) -> None:
        """Return the possible actions from given state"""
        ## Convert the observation to feature
        indices = [self.text_action_to_discrete(action) for action in possible_actions]
        self.action_mask = np.zeros(30, dtype=bool)
        self.action_mask[indices] = True

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
        obs_desc, info_description = self.generate_description()
        text_obs = self.observation_to_text(obs_desc)
        self._update_action_mask(info_description["possible_actions"])
        self.trajectory_obs_text = [obs_desc]
        self.trajectory_diff_text = []
        self.trajectory_act_text = []
        info = {
            "goal": self.goal_str,
            "action_mask": self.action_mask,
            "text_obs": text_obs,
            "step": self.current_step,
            "trajectory_obs_text": self.trajectory_obs_text,
            "trajectory_act_text": self.trajectory_act_text,
            "trajectory_diff_text": self.trajectory_diff_text,
        }
        self._last_text_obs = obs_desc
        self.inventory = info_description["inventory"]

        return obs.flatten(), info

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
        # Compute the str description
        obs_desc, info_description = self.generate_description()
        text_obs, transition_type = self.get_diff(
            self._last_text_obs, obs_desc, action_str
        )
        self._update_action_mask(info_description["possible_actions"])
        text_action = self.action_to_text(action_str)
        self.trajectory_act_text.append(text_action)
        self.trajectory_diff_text.append(text_obs)
        self.trajectory_obs_text.append(obs_desc)
        info = {
            "goal": self.goal_str,
            "action_mask": self.action_mask,
            "text_obs": text_obs,
            "obs_desc": obs_desc,
            "text_action": text_action,
            "transition_type": transition_type,
            "step": self.current_step,
            "success": r,
            "trajectory_obs_text": self.trajectory_obs_text,
            "trajectory_act_text": self.trajectory_act_text,
            "trajectory_diff_text": self.trajectory_diff_text,
        }
        self._last_text_obs = obs_desc
        self.inventory = info_description["inventory"]
        # Reset the size of obj help to find the obj grown in the current step
        self.playground.unwrapped.reset_size()

        # For alp, the reward is the index on the transition type

        return (
            obs.flatten(),
            self.TRANSITION_TYPE_TO_ID[transition_type],
            done,
            False,
            info,
        )

    def render(self):
        raise NotImplementedError("Rendering is not supported for the environment")

    def get_action_mask(self) -> np.ndarray:
        return self.action_mask


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--filepath",
        type=str,
        default=os.path.join(os.path.dirname(__file__), "data/test_dataset.json"),
        help="Path to save the dataset",
    )

    # Argument for the environment
    parser.add_argument(
        "--max-steps",
        type=int,
        default=30,
        help="Maximum number of steps in the environment",
    )
    parser.add_argument(
        "--max-objects",
        type=int,
        default=8,
        help="Maximum number of objects in the environment",
    )
    parser.add_argument(
        "--nb-trajectories",
        type=int,
        default=9,
        help="Number of trajectories for the dataset. A third will be dedicated to collect small Herbivorous, a third to collect small and big Herbivorous and a third to act randomly",
    )
    parser.add_argument(
        "--rule",
        type=str,
        default="grow any small_herbivorous then grow any big_herbivorous",
        help="Rule for the environment",
    )
    args = parser.parse_args()

    env: BaseWrapper = gym.make(
        "worldllm_envs/PlaygroundText-v1",
        **{
            "max_steps": args.max_steps,
            "seed": args.seed,
            "playground_config": {"max_nb_objects": args.max_objects},
            "test_dataset_path": None,
        },
    )
    env.reset(options={"rule": args.rule})
    # Create the different agents
    perfect_agent_sh = PerfectAgent(
        env.action_space, curriculum_goals=["Grow any small_herbivorous"]
    )
    perfect_agent_shbh = PerfectAgent(
        env.action_space,
        curriculum_goals=["Grow any small_herbivorous then grow any big_herbivorous"],
    )
    random_agent = RandomAgent(env.action_space)
    # Collect all trajectories
    trajectories: List[Trajectory] = []
    for incr, agent in enumerate([perfect_agent_sh, perfect_agent_shbh, random_agent]):
        new_trajectories, new_discovered_transitions, lst_transition = (
            agent.generate_trajectories(
                env,
                (args.nb_trajectories + incr) // 3,
                {"pipeline_progression": incr / 3},
                0,
            )
        )
        trajectories.extend(new_trajectories)
    # Save the trajectories
    with open(args.filepath, "w") as f:
        json.dump([t.to_dict() for t in trajectories], f)

    print("Done.")
