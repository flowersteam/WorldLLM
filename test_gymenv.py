import time

import gymnasium
import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from worldllm_envs.base import BaseRuleEnv, RandomAgent, Trajectory

writer = SummaryWriter(log_dir="./logs_ppo_sb3/RandomAgent2")
seed = None

env: BaseRuleEnv = gymnasium.make(
    "worldllm_envs/Playground-v1",
    **{"max_steps": 30, "seed": seed, "playground_config": {"max_nb_objects": 8}},
)

count_transition_type = {}


agent = RandomAgent(env.action_space)
n_episodes = 1667
start_time = time.perf_counter()
n_steps = 0
all_transitions_episodes = {}
timestamps_sh = []
for _ in tqdm(range(n_episodes)):
    new_rule = "grow any small_herbivorous then grow any big_herbivorous"
    obs, info = env.reset(options={"rule": new_rule})
    # Compute plan
    agent.reset(info)
    done = False
    sum_reward = 0
    count_transition_type_episode = {}
    while not done:
        # Record inputs from keyboard
        # Print possible actions
        action, agent_done = agent(obs, **info)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated or agent_done
        n_steps += 1
        sum_reward += reward

        # Add logs
        count_transition_type[info["transition_type"]] = (
            count_transition_type.get(info["transition_type"], 0) + 1
        )
        count_transition_type_episode[info["transition_type"]] = (
            count_transition_type_episode.get(info["transition_type"], 0) + 1
        )
        if info["transition_type"] == "transformSH":
            timestamps_sh.append(info["step"])
        for transition_type, count in count_transition_type.items():
            writer.add_scalar(f"transitions/{transition_type}", count, n_steps)
        writer.add_scalar(f"rewards/{info['transition_type']}", reward, n_steps)
    for transi_type in count_transition_type:
        if transi_type not in all_transitions_episodes:
            all_transitions_episodes[transi_type] = []
        all_transitions_episodes[transi_type].append(
            count_transition_type_episode.get(transi_type, 0)
        )
    for transi_type, value in all_transitions_episodes.items():
        writer.add_scalar(
            f"transitions_episodes/{transi_type}", np.mean(value[-100:]), n_steps
        )

    writer.add_scalar("timestamp/transformSH", np.mean(timestamps_sh[-5:]), n_steps)

    writer.add_scalar("rollout/ep_rew_mean", sum_reward, n_steps)

print(f"Time: {time.perf_counter() - start_time}")
writer.flush()
print("Done.")
