import random
import re
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np

from worldllm_envs.base import BaseRuleEnv
from worldllm_envs.playground.descriptions import generate_all_descriptions
from worldllm_envs.playground.env_params import get_env_params
from worldllm_envs.playground.playgroundnavv1 import PlayGroundNavigationV1
from worldllm_envs.playground.reward_function import (
    get_reward_from_state,
    sample_descriptions_from_state,
)


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

        self.observation_space = gym.spaces.Discrete(2)
        self.action_space = gym.spaces.MultiDiscrete([3, 3, 3])
        self.tokens = ["closed", "opened"]
        self.stat_prompt = ""
        self.stat_template = statisitician_template
        self.th_prompt = ""
        self.th_template = theorist_template
        self.max_steps = 64
        self.train = True
        self.mask_growgrow = False
        self.grasp_only = False
        self.goal_sampler = None
        self.random_type = False

        super().__init__(**kwargs)

        # The playground environment
        self.playground = PlayGroundNavigationV1(**kwargs.get("playground_config", {}))

        # Dict containing all the playground environment parameters
        self.env_params = get_env_params()

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
    def generate_rule():
        print("WARNING: no other rule than the default one is available")
        return "None"

    def action_to_text(self, action):
        pass

    def observation_to_text(self):
        pass

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
                goal_type = random.choice(
                    [
                        r"^Grasp",
                        r"^Grow.*\b(lion|grizzly|shark|fox|bobcat|coyote|small_carnivorous|big_carnivorous)\b",
                        r"^Grow.*\b(elephant|giraffe|rhinoceros|pig|cow|sheep|small_herbivorous|big_herbivorous|animal)\b",
                        r"^Grow.*\b(carrot|potato|beet|berry|pea|thing|living_thing)\b",
                    ]
                )
                self.goal_str = random.choice(
                    [
                        goal
                        for goal in self.train_descriptions
                        if re.match(goal_type, goal) or not self.random_type
                    ]
                )
        else:
            # If we are in test mode, we want to test the model on unseen data
            # Sample goal uniformly for the type 'Grasp', 'Grow', 'Grow then grasp', 'Grow then grow'
            goal_type = random.choice(
                [
                    r"^Grasp",
                    r"^Grow.*\b(lion|grizzly|shark|fox|bobcat|coyote|small_carnivorous|big_carnivorous)\b",
                    r"^Grow.*\b(elephant|giraffe|rhinoceros|pig|cow|sheep|small_herbivorous|big_herbivorous|animal)\b",
                    r"^Grow.*\b(carrot|potato|beet|berry|pea|thing|living_thing)\b",
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

    # region Old playground utils and actions
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

            if obj.object_descr["categories"] == "plant" and not obj.grown_once:
                key = (
                    obj.object_descr["colors"]
                    + " "
                    + obj.object_descr["types"]
                    + " seed"
                )
            elif (
                "carnivorous" in obj.object_descr["categories"]
                or "herbivorous" in obj.object_descr["categories"]
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
                }
            else:  # If there are multiple objects with the same description
                self.obj_dict[key + str(i)] = {
                    "position": obj.position,
                    "grasped": obj.grasped,
                    "agent_on": agent_on,
                    "grown": obj.grown_once,
                }
                i += 1

    def generate_description(self):
        """
        Return a natural language description of the scene
        """
        desc = "You see: "
        desc += ", ".join(
            self.rm_trailing_number(obj)
            for obj in self.obj_dict.keys()
            if not self.obj_dict[obj]["grasped"]
        )
        agent_on = [
            self.rm_trailing_number(obj)
            for obj in self.obj_dict.keys()
            if self.obj_dict[obj]["agent_on"]
        ]
        desc += (
            f'\nYou are on: {", ".join(agent_on) if len(agent_on) > 0 else "nothing"}'
        )
        obj_held = [
            self.rm_trailing_number(obj)
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
                "Go to " + self.rm_trailing_number(obj)
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

    def rm_trailing_number(self, input_str):
        return re.sub(r"\d+$", "", input_str)

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


# endregion
