import time

import gymnasium
from tqdm import tqdm

from worldllm_envs.base import BaseRuleEnv
from worldllm_envs.playground.playground_text_wrapper import PerfectAgent

seed = 15

env: BaseRuleEnv = gymnasium.make(
    "worldllm_envs/PlaygroundText-v1", **{"max_steps": 20, "seed": seed}
)
agent = PerfectAgent(env.action_space)
for _ in range(15):
    new_rule = env.unwrapped.generate_rule()
    obs, info = env.reset(options={"rule": new_rule})
    # Compute plan
    agent.reset(info)

    index = 0
    done = False
    while not done:
        # Record inputs from keyboard
        action = agent(obs)
        obs, _, done, _, info = env.step(action)
        index += 1
    # print(" ".join(info["text_trajectory"]))

print("Done.")
