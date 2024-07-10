import gymnasium

from worldllm_envs.envs.base import BaseRuleEnv

env: BaseRuleEnv = gymnasium.make("worldllm_envs/Door-v0")

new_rule = env.generate_rule()
obs, info = env.reset(options={"rule": new_rule})
print("Observation: ", obs)

done = False
while not done:
    action = env.action_space.sample()
    print("Action: ", env.unwrapped.action_to_text(action))
    obs, _, done, _, info = env.step(action)
    print("Observation: ", obs)
