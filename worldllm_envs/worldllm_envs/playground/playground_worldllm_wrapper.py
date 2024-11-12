from typing import Any, Dict, List, Tuple

from worldllm_envs.base import BaseRuleEnv, TextWrapper


class PlaygroundWrapper(TextWrapper):
    """Text wrapper for the Playground environment"""

    def __init__(self, env: BaseRuleEnv):
        super().__init__(env)
        self.last_obs = None
        self.last_action = None

        self.act_trajectory: List[str]
        self.diff_trajectory: List[str]

    def action_to_text(self, action):
        act_text = self.env.unwrapped.action_to_text(action)
        self.last_action = action
        return act_text

    def observation_to_text(self, observation) -> Tuple[str, Dict[str, Any]]:
        if self.last_obs is None:
            text_obs = self.env.unwrapped.observation_to_text(observation)
            self.last_obs = observation
            return text_obs, {}
        text_obs, transition_type = self.env.unwrapped.get_diff(
            self.last_obs, observation, self.last_action
        )
        self.last_obs = observation
        return text_obs, {"transition_type": transition_type}

    def step(self, action):
        act_text = self.action_to_text(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs_diff, add_info = self.observation_to_text(observation)
        info.update(add_info)
        info["action_text"] = act_text
        self.obs_trajectory.append(observation)
        self.act_trajectory.append(act_text)
        self.diff_trajectory.append(obs_diff)
        info["trajectory_obs_text"] = self.obs_trajectory
        info["trajectory_act_text"] = self.act_trajectory
        info["trajectory_diff_text"] = self.diff_trajectory
        return (
            obs_diff,
            reward,
            terminated,
            truncated,
            info,
        )

    def reset(self, seed=None, options=None):
        self.last_obs = None
        self.last_action = None
        observation, info = self.env.reset(seed=seed, options=options)
        text_obs, add_info = self.observation_to_text(observation)
        info.update(add_info)
        self.obs_trajectory = [observation]
        self.act_trajectory = []
        self.diff_trajectory = []
        info["trajectory_obs_text"] = self.obs_trajectory
        info["trajectory_act_text"] = self.act_trajectory
        info["trajectory_diff_text"] = self.diff_trajectory
        return text_obs, info
