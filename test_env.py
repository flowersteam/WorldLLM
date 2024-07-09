import gymnasium

from worldllm_envs.envs.base import BaseRuleEnv

env: BaseRuleEnv = gymnasium.make("worldllm_envs/Door-v0")

new_rule = env.generate_rule()
obs, info = env.reset(options={"rule": new_rule})
done = False
while not done:
    action = env.action_space.sample()
    print("Action: ", env.get_action_text(action))
    obs, _, done, _, info = env.step(action)
    print("Observation: ", obs)
