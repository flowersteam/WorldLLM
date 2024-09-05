import time

import gymnasium
from tqdm import tqdm

from worldllm_envs.base import BaseRuleEnv
from worldllm_envs.playground.playground_text_wrapper import PerfectAgent

success_rate = 0
all_reset_time = 0.0
all_plan_time = 0.0
all_exceute_time = 0.0
for seed in tqdm(range(10000)):
    time_start = time.perf_counter()
    env: BaseRuleEnv = gymnasium.make(
        "worldllm_envs/PlaygroundText-v1", **{"max_steps": 20, "seed": seed}
    )
    new_rule = env.unwrapped.generate_rule()
    obs, info = env.reset(options={"rule": new_rule})
    reset_time = time.perf_counter() - time_start
    # Compute plan
    agent = PerfectAgent(env.action_space, info["obj_dict"], info["goal"])
    plan_time = time.perf_counter() - reset_time - time_start

    index = 0
    done = False
    while not done:
        # Record inputs from keyboard
        action = agent(obs)
        obs, _, done, _, info = env.step(action)
        index += 1
    execute_time = time.perf_counter() - plan_time - reset_time - time_start
    # print("\n".join(info["text_trajectory"]))
    assert agent.is_done == True
    success_rate += 1
    all_reset_time += reset_time
    all_plan_time += plan_time
    all_exceute_time += execute_time
print("Reset time:", (all_reset_time) / 10000)
print("Plan time:", (all_plan_time) / 10000)
print("Execute time:", (all_exceute_time) / 10000)
print(success_rate / 10000)
