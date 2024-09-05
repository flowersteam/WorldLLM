from worldllm_envs.base import BaseRuleEnv, TextWrapper


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

    def reset(self, seed=None, options=None):
        self.last_obs = None
        self.last_action = None
        return super().reset(seed=seed, options=options)
