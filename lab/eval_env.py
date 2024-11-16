import time

import gymnasium
from tqdm import tqdm

from worldllm_envs.base import BaseRuleEnv
from worldllm_envs.playground.playground_text_wrapper import RandomAgent

seed = None

env: BaseRuleEnv = gymnasium.make(
    "worldllm_envs/PlaygroundText-v1",
    **{"max_steps": 30, "seed": seed, "playground_config": {"max_nb_objects": 8}},
)


agent = RandomAgent(env.action_space)
n_episodes = 10
start_time = time.perf_counter()
n_steps = 0
for _ in tqdm(range(n_episodes)):
    new_rule = "grow any small_herbivorous then grow any big_herbivorous"
    obs, info = env.reset(options={"rule": new_rule})
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
    print(info["trajectory_obs_diff"])

print(f"Time: {time.perf_counter() - start_time}")
print("Done.")
