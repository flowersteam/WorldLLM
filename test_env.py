import time

import gymnasium
from tqdm import tqdm

from utils.utils_env import RandomAgent, Trajectory
from worldllm_envs.base import BaseRuleEnv

seed = None

env: BaseRuleEnv = gymnasium.make(
    "worldllm_envs/Playground-v1",
    **{"max_steps": 30, "seed": seed, "playground_config": {"max_nb_objects": 8}},
)


agent = RandomAgent(env.action_space)
mean_reward = 0
n_episodes = 1
start_time = time.perf_counter()
for _ in tqdm(range(n_episodes)):
    new_rule = "grow any small_herbivorous then grow any big_herbivorous"
    obs, info = env.reset(options={"rule": new_rule})
    # Compute plan
    agent.reset(info)
    index = 0
    done = False
    sum_reward = 0
    while not done:
        # Record inputs from keyboard
        # Print possible actions
        action = agent(obs, **info)
        obs, reward, done, _, info = env.step(action)
        index += 1
        sum_reward += reward
    mean_reward += sum_reward
print("Mean reward:", mean_reward / n_episodes)
print(f"Time: {time.perf_counter() - start_time}")
print("Done.")
