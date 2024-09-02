import gymnasium

from worldllm_envs.base import BaseRuleEnv

seed = 28
env: BaseRuleEnv = gymnasium.make(
    "worldllm_envs/PlaygroundText-v1", **{"max_steps": 20, "seed": seed}
)
new_rule = env.unwrapped.generate_rule()
obs, info = env.reset(options={"rule": new_rule})
print("Rule:", new_rule)
print("Goal: ", info["goal"])
print("Observation: ", obs)
recorded_actions = [
    "go to green carrot seed",
    "go to green water",
    "grasp",
    "go to baby green elephant",
    "grasp",
    "go to green carrot seed",
    "release baby green elephant",
    "release green water",
]
index = 0

done = False
while not done:
    # Record inputs from keyboard
    action = recorded_actions[index]
    obs, _, done, _, info = env.step(action)
    index += 1

print("\n".join(info["text_trajectory"]))
