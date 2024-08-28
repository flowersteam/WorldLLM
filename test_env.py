import gymnasium

from worldllm_envs.base import BaseRuleEnv

seed = 5
env: BaseRuleEnv = gymnasium.make("worldllm_envs/Door-v0", seed=seed)
new_rule = env.generate_rule()
obs, info = env.reset(options={"rule": new_rule})
print("Rule:", new_rule)
print("Observation: ", obs)

done = False
while not done:
    action = env.action_space.sample()
    print("Action: ", env.unwrapped.action_to_text(action))
    obs, _, done, _, info = env.step(action)
    print("Observation: ", obs, ", action: ", info["action_text"])
