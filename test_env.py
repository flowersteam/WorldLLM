import gymnasium

from worldllm_envs.base import BaseRuleEnv

seed = 19
env: BaseRuleEnv = gymnasium.make(
    "worldllm_envs/PlaygroundText-v1", **{"max_steps": 20, "seed": seed}
)
new_rule = env.unwrapped.generate_rule()
obs, info = env.reset(options={"rule": new_rule})
print("Rule:", new_rule)
print("Goal: ", info["goal"])
print("Observation: ", obs)

done = False
while not done:
    # Record inputs from keyboard
    action = input("Enter action: ")
    print("Action: ", env.unwrapped.action_to_text(action))
    obs, _, done, _, info = env.step(action)
    print(info["obs_trajectory"][-1])
    print("Observation: ", obs, ", action: ", info["action_text"])

print(info["text_trajectory"])
