import argparse
from functools import partial
from itertools import product
from typing import Any, Dict

import gymnasium
import numpy as np
from sb3_contrib import MaskablePPO

# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from worldllm_envs.base import BaseRuleEnv


class TransitionCounterCallback(BaseCallback):
    def __init__(self, verbose=0, n_envs=1):
        super(TransitionCounterCallback, self).__init__(verbose)
        self.transition_counts = {}
        self._lst_transitions_episodes = [dict() for _ in range(n_envs)]
        self.all_transitions_episodes = {}
        self.all_timestamps_sh = []

    def _on_step(self) -> bool:
        # Access the current observation and info
        lst_infos = self.locals.get("infos")
        lst_rewards = self.locals.get("rewards")
        for incr, (info, reward) in enumerate(zip(lst_infos, lst_rewards)):
            if info["step"] == 1:
                # Reset the transition count for the episode
                for transi_type in self.transition_counts:
                    if transi_type not in self.all_transitions_episodes:
                        self.all_transitions_episodes[transi_type] = []
                    self.all_transitions_episodes[transi_type].append(
                        self._lst_transitions_episodes[incr].get(transi_type, 0),
                    )
                self._lst_transitions_episodes[incr] = dict()

            # Assuming you have a way to get the current transition type
            self.transition_counts[info["transition_type"]] = (
                self.transition_counts.get(info["transition_type"], 0) + 1
            )
            self._lst_transitions_episodes[incr][info["transition_type"]] = (
                self._lst_transitions_episodes[incr].get(info["transition_type"], 0) + 1
            )
            if info["transition_type"] == "transformSH":
                self.all_timestamps_sh.append(info["step"])
            self.logger.record(f"rewards/{info['transition_type']}", reward)

        # Log the transition counts to TensorBoard
        for transition_type, count in self.transition_counts.items():
            self.logger.record(f"transitions/{transition_type}", count)
        self.logger.record(
            "timestamp/transformSH", np.mean(self.all_timestamps_sh[-5:])
        )
        for transi_type, value in self.all_transitions_episodes.items():
            self.logger.record(
                f"transitions_episodes/{transi_type}",
                np.mean(value[-100:]),
            )

        return True


def mask_fn(env):
    return env.unwrapped.action_mask


def make_env(countbased_dict: Dict[str, int]):
    env: BaseRuleEnv = gymnasium.make(
        "worldllm_envs/Playground-v1",
        **{
            "max_steps": 30,
            "playground_config": {"max_nb_objects": 8},
        },
    )
    new_rule = "grow any small_herbivorous then grow any big_herbivorous"
    env.reset(options={"rule": new_rule})
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    env.unwrapped.set_shared_countbased(countbased_dict)
    return env


# hyperparameters = {
#     "gamma": [0.95, 0.99],
#     "learning_rate": [1e-3, 5e-4, 1e-4],
#     "vf_coef": [0.1, 0.25, 0.5],
#     "n_steps": [128, 256, 512, 1024],
#     "n_epochs": [2, 5, 10],
# }

if __name__ == "__main__":
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument("--gamma", type=float)
    argsparser.add_argument("--lr", type=float)
    argsparser.add_argument("--vf_coef", type=float)
    argsparser.add_argument("--n_steps", type=int)
    argsparser.add_argument("--n_epochs", type=int)
    argsparser.add_argument("--gae", type=float)
    config = vars(argsparser.parse_args())

    n_envs = 9

    # Load first environment
    env: BaseRuleEnv = gymnasium.make(
        "worldllm_envs/Playground-v1",
        **{"max_steps": 30, "playground_config": {"max_nb_objects": 8}},
    )
    new_rule = "grow any small_herbivorous then grow any big_herbivorous"
    env.reset(options={"rule": new_rule})
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    countbased = env.unwrapped.get_shared_countbased()

    envs = make_vec_env(
        partial(make_env, countbased_dict=countbased),
        n_envs=n_envs,
    )
    # Train PPO
    model = MaskablePPO(
        "MlpPolicy",
        envs,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        ent_coef=0.01,
        vf_coef=config["vf_coef"],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        gae_lambda=config["gae"],
        device="cuda",
        verbose=1,
        tensorboard_log="./logs_ppo_sb3",
    )
    callback = TransitionCounterCallback(model.verbose, n_envs)
    model.learn(
        50_000,
        tb_log_name="PPO_Test_vecEnv",
        progress_bar=True,
        log_interval=1,
        callback=callback,
    )
    model.save("ppo_mask")
    # Load model
    model = MaskablePPO.load("ppo_mask")

    for _ in range(3):
        print("New episode")
        obs, _ = env.reset()
        done = False
        while not done:
            # Retrieve current action mask
            action_masks = get_action_masks(env)
            action, _ = model.predict(obs, action_masks=action_masks)
            obs, reward, terminated, truncated, info = env.step(action.item())
            done = terminated or truncated
            print(info["text_obs"])
