import time

import gymnasium
from tqdm import tqdm

from worldllm_envs.base import BaseRuleEnv, RandomAgent

seed = None

env: BaseRuleEnv = gymnasium.make(
    "worldllm_envs/Door-v0",
    **{"seed": seed, "test_dataset_path": None},
)
new_rule = env.generate_rule("not_blue_key")
env.reset(options={"rule": new_rule})


agent = RandomAgent(env.action_space)
n_episodes = 10
start_time = time.perf_counter()
n_steps = 0
for _ in tqdm(range(n_episodes)):
    obs, info = env.reset()
    # Compute plan
    agent.reset(info)
    done = False
    while not done:
        # Record inputs from keyboard
        # Print possible actions
        action, agent_done = agent(obs, **info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated or agent_done
        n_steps += 1
        print(f"Action: {action}, Observation: {obs}")
    print(info["trajectory_diff_text"])

print(f"Time: {time.perf_counter() - start_time}")
print("Done.")
