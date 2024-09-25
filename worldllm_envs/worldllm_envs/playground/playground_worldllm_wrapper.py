from typing import Any, Dict, Tuple

from worldllm_envs.base import BaseRuleEnv, TextWrapper


class PlaygroundWrapper(TextWrapper):
    """Text wrapper for the Playground environment"""

    def __init__(self, env: BaseRuleEnv):
        super().__init__(env)
        self.last_obs = None
        self.last_action = None

    def action_to_text(self, action):
        act_text = self.env.unwrapped.action_to_text(action)
        self.last_action = action
        return act_text

    def observation_to_text(self, observation) -> Tuple[str, Dict[str, Any]]:
        if self.last_obs is None:
            text_obs = self.env.unwrapped.observation_to_text(observation)
            self.last_obs = observation
            return text_obs, {}
        text_obs, add_info = self.env.unwrapped.get_diff_description(
            self.last_obs, observation, self.last_action
        )
        self.last_obs = observation
        return text_obs, add_info

    def step(self, action):
        act_text = self.action_to_text(action)
        observation, reward, terminated, truncated, info = self.env.step(action)
        obs_text, add_info = self.observation_to_text(observation)
        info.update(add_info)
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
        self.last_obs = None
        self.last_action = None
        return super().reset(seed=seed, options=options)
