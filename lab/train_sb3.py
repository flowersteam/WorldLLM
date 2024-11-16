import argparse
import random
from functools import partial
from typing import Any, ClassVar, Dict, Optional, Tuple, Type, TypeVar, Union

import gymnasium
import numpy as np
import torch as th
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import MaskableRolloutBuffer
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv

from worldllm_envs.base import BaseRuleEnv


def correct_reward(prev_rewards: np.ndarray):
    """Compute the rewards given the previous rewards. To gain time the previous rewards are indices for the new rewards"""
    lst_rewards = np.array(
        [
            0.1,
            0.09,
            0.21,
            0.03,
            0.77,
            0.87,
            1.74,
        ]
    )
    new_rewards = lst_rewards[prev_rewards.round().astype(int)]
    return new_rewards


class CustomMaskableRolloutBuffer(MaskableRolloutBuffer):
    """
    Modification of the rollout buffer to include true rewards tracking
    """

    true_rewards: np.ndarray

    def reset(self) -> None:
        self.true_rewards = np.zeros((self.buffer_size, self.n_envs), dtype=np.float32)
        super().reset()

    def add(
        self,
        *args,
        action_masks: Optional[np.ndarray] = None,
        true_rewards: Optional[np.ndarray] = None,
        **kwargs,
    ) -> None:
        """
        :param action_masks: Masks applied to constrain the choice of possible actions.
        """
        self.true_rewards[self.pos] = np.array(true_rewards)
        if action_masks is not None:
            self.action_masks[self.pos] = action_masks.reshape(
                (self.n_envs, self.mask_dims)
            )

        super().add(*args, **kwargs)


class CustomMaskablePPO(MaskablePPO):
    """Modification of the PPO algorithm to modify rewards between end of the collecting of the rollout
    and the training"""

    def collect_rollouts(
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: RolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> bool:
        """
        Collect experiences using the current policy and fill a ``RolloutBuffer``.
        The term rollout here refers to the model-free notion and should not
        be used with the concept of rollout used in model-based RL or planning.

        This method is largely identical to the implementation found in the parent class.

        :param env: The training environment
        :param callback: Callback that will be called at each step
            (and at the beginning and end of the rollout)
        :param rollout_buffer: Buffer to fill with rollouts
        :param n_steps: Number of experiences to collect per environment
        :param use_masking: Whether or not to use invalid action masks during training
        :return: True if function returned with at least `n_rollout_steps`
            collected, False if callback terminated rollout prematurely.
        """

        assert isinstance(
            rollout_buffer, CustomMaskableRolloutBuffer
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()

        if use_masking and not is_masking_supported(env):
            raise ValueError(
                "Environment does not support action masking. Consider using ActionMasker wrapper"
            )

        callback.on_rollout_start()

        while n_steps < n_rollout_steps:
            with th.no_grad():
                # Convert to pytorch tensor or to TensorDict
                obs_tensor = obs_as_tensor(self._last_obs, self.device)

                # This is the only change related to invalid action masking
                if use_masking:
                    action_masks = get_action_masks(env)

                actions, values, log_probs = self.policy(
                    obs_tensor, action_masks=action_masks
                )

            actions = actions.cpu().numpy()
            new_obs, rewards, dones, infos = env.step(actions)
            # Copy rewards to get reward before adding the value function
            true_rewards = np.copy(rewards)

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                return False

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with th.no_grad():
                        terminal_value = self.policy.predict_values(terminal_obs)[0]
                    rewards[idx] += self.gamma * terminal_value

            rollout_buffer.add(
                self._last_obs,
                actions,
                rewards,
                self._last_episode_starts,
                values,
                log_probs,
                action_masks=action_masks,
                true_rewards=true_rewards,
            )
            self._last_obs = new_obs  # type: ignore[assignment]
            self._last_episode_starts = dones

        with th.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        # We need to adjust the reward
        correct_rewards = correct_reward(self.rollout_buffer.true_rewards)
        # We need to keep the terminated value in case. The true rewards are there for that
        self.rollout_buffer.rewards = (
            self.rollout_buffer.rewards - self.rollout_buffer.true_rewards
        ) + correct_rewards
        # End
        rollout_buffer.compute_returns_and_advantage(last_values=values, dones=dones)

        callback.update_locals(locals())
        callback.on_rollout_end()

        return True

    def learn(  # type: ignore[override]
        self,
        total_timesteps: int,
        callback=None,
        log_interval: int = 1,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
        use_masking: bool = True,
        progress_bar: bool = False,
    ):
        iteration = 0

        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar,
        )

        callback.on_training_start(locals(), globals())

        assert self.env is not None

        while self.num_timesteps < total_timesteps:
            continue_training = self.collect_rollouts(
                self.env, callback, self.rollout_buffer, self.n_steps, use_masking
            )

            if not continue_training:
                break

            iteration += 1
            self._update_current_progress_remaining(self.num_timesteps, total_timesteps)

            # Display training infos
            if log_interval is not None and iteration % log_interval == 0:
                self._dump_logs(iteration)

            self.train()

        callback.on_training_end()

        return self


class TransitionCounterCallback(BaseCallback):
    def __init__(self, hyperparameters, verbose=0, n_envs=1):
        super(TransitionCounterCallback, self).__init__(verbose)
        self.transition_counts = {}
        self._lst_transitions_episodes = [dict() for _ in range(n_envs)]
        self.all_transitions_episodes = {}
        self.all_timestamps_sh = []
        self.all_timestamps_p = []
        self.hyperparameters = hyperparameters

    def _on_rollout_end(self) -> None:
        """Save mean of the true reward"""
        self.logger.record(
            "rollout/Corrected_mean_reward", np.mean(self.locals.get("correct_rewards"))
        )

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
            # We add only the first time the transition appears
            if info["transition_type"] == "transformSH":
                self.all_timestamps_sh.append(info["step"])
            if (
                info["transition_type"] == "transformP"
                and self._lst_transitions_episodes[incr][info["transition_type"]] == 1
            ):
                self.all_timestamps_p.append(info["step"])
            self.logger.record(f"rewards/{info['transition_type']}", reward)

        # Log the transition counts to TensorBoard
        for transition_type, count in self.transition_counts.items():
            self.logger.record(f"transitions/{transition_type}", count)
        self.logger.record(
            "timestamp/transformSH", np.mean(self.all_timestamps_sh[-5:])
        )
        self.logger.record("timestamp/transformP", np.mean(self.all_timestamps_p[-5:]))
        for transi_type, value in self.all_transitions_episodes.items():
            self.logger.record(
                f"transitions_episodes/{transi_type}",
                np.mean(value[-100:]),
            )

        # Log config values
        for hyperparam in self.hyperparameters:
            self.logger.record(
                "hyperparameters/" + hyperparam, self.hyperparameters[hyperparam]
            )

        return True


def mask_fn(env):
    return env.unwrapped.action_mask


def make_env():
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
    return env


if __name__ == "__main__":
    argsparser = argparse.ArgumentParser()
    argsparser.add_argument("--gamma", type=float, required=True)
    argsparser.add_argument("--lr", type=float, required=True)
    argsparser.add_argument("--vf_coef", type=float, required=True)
    argsparser.add_argument("--n_steps", type=int, required=True)
    argsparser.add_argument("--n_epochs", type=int, required=True)
    argsparser.add_argument("--gae", type=float, required=True)
    argsparser.add_argument("--seed", type=int, required=True)
    argsparser.add_argument("--nn_size", type=int, required=True)
    config = vars(argsparser.parse_args())

    # Set the seed
    np.random.seed(config["seed"])
    th.manual_seed(config["seed"])
    th.cuda.manual_seed_all(config["seed"])
    random.seed(config["seed"])
    n_envs = 9

    envs = make_vec_env(
        partial(make_env),
        n_envs=n_envs,
    )
    tensorboard_log_name = f"./PPO_{config['seed']}"
    # Train PPO
    model = CustomMaskablePPO(
        "MlpPolicy",
        envs,
        rollout_buffer_class=CustomMaskableRolloutBuffer,
        gamma=config["gamma"],
        learning_rate=config["lr"],
        ent_coef=0.01,
        vf_coef=config["vf_coef"],
        n_steps=config["n_steps"],
        n_epochs=config["n_epochs"],
        gae_lambda=config["gae"],
        device="cuda",
        verbose=1,
        tensorboard_log="./logs_ppo_sb3_test",
        policy_kwargs=dict(
            net_arch=dict(
                pi=[config["nn_size"], config["nn_size"]],
                vf=[config["nn_size"], config["nn_size"]],
            )
        ),
        seed=config["seed"],
    )
    print(model.policy)
    callback = TransitionCounterCallback(config, model.verbose, n_envs)
    model.learn(
        1_000_000,
        tb_log_name=tensorboard_log_name,
        progress_bar=True,
        log_interval=1,
        callback=callback,
    )
    model.save("ppo_mask")
    # Load model
    model = CustomMaskablePPO.load("ppo_mask")

    env = make_env()
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
