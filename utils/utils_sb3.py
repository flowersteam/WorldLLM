import random
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import gymnasium
import numpy as np
import torch
from gymnasium import spaces
from sb3_contrib import MaskablePPO
from sb3_contrib.common.maskable.buffers import (
    MaskableRolloutBuffer,
    MaskableRolloutBufferSamples,
)
from sb3_contrib.common.maskable.utils import get_action_masks, is_masking_supported
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.buffers import DictRolloutBuffer, RolloutBuffer
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.utils import obs_as_tensor
from stable_baselines3.common.vec_env import VecEnv
from tqdm import tqdm

from utils.utils_env import BaseAgent, Trajectory


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

    def collect_rollouts(  # type: ignore[override]
        self,
        env: VecEnv,
        callback: BaseCallback,
        rollout_buffer: CustomMaskableRolloutBuffer,
        n_rollout_steps: int,
        use_masking: bool = True,
    ) -> Tuple[List[List[Trajectory]], Set[str], torch.Tensor, np.ndarray]:
        """Modified collect rollouts to modify the rewards before training"""

        assert isinstance(
            rollout_buffer, CustomMaskableRolloutBuffer
        ), "RolloutBuffer doesn't support action masking"
        assert self._last_obs is not None, "No previous observation was provided"
        # Switch to eval mode (this affects batch norm / dropout)
        self.policy.set_training_mode(False)
        n_steps = 0
        action_masks = None
        rollout_buffer.reset()
        # Reset env since we want full trajectory for easy scoring
        self._last_obs = env.reset()  # type: ignore[assignment]
        self._last_episode_starts = np.ones(env.num_envs, dtype=bool)

        trajectories: List[List[Trajectory]] = [[] for _ in range(env.num_envs)]
        set_discovered_transitions = set()

        if use_masking and not is_masking_supported(env):
            raise ValueError(
                "Environment does not support action masking. Consider using ActionMasker wrapper"
            )

        callback.on_rollout_start()
        pbar = tqdm(total=n_rollout_steps, desc="Collecting Rollouts", leave=False)
        while n_steps < n_rollout_steps:
            with torch.no_grad():
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
            # Update seen transitions
            for idx, info in enumerate(infos):
                set_discovered_transitions.add(info["transition_type"])

            self.num_timesteps += env.num_envs

            # Give access to local variables
            callback.update_locals(locals())
            if not callback.on_step():
                raise ValueError("Callback failed")

            self._update_info_buffer(infos, dones)
            n_steps += 1

            if isinstance(self.action_space, spaces.Discrete):
                # Reshape in case of discrete action
                actions = actions.reshape(-1, 1)

            # Handle timeout by bootstraping with value function
            # see GitHub issue #633
            for idx, done in enumerate(dones):
                if done:
                    trajectories[idx].append(Trajectory(infos[idx]["trajectory_text"]))
                if (
                    done
                    and infos[idx].get("terminal_observation") is not None
                    and infos[idx].get("TimeLimit.truncated", False)
                ):
                    terminal_obs = self.policy.obs_to_tensor(
                        infos[idx]["terminal_observation"]
                    )[0]
                    with torch.no_grad():
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

            pbar.update(1)

        with torch.no_grad():
            # Compute value for the last timestep
            # Masking is not needed here, the choice of action doesn't matter.
            # We only want the value of the current observation.
            values = self.policy.predict_values(obs_as_tensor(new_obs, self.device))  # type: ignore[arg-type]

        # Add trajectories that are partially filled
        for idx, done in enumerate(dones):
            if not done:  # We don't want to appen twice the same trajectory
                trajectories[idx].append(Trajectory(infos[idx]["trajectory_text"]))
        # Return the trajectories done and last information
        return trajectories, set_discovered_transitions, values, dones

    def get_rewards(self, rollout_buffer: CustomMaskableRolloutBuffer) -> np.ndarray:
        return rollout_buffer.true_rewards

    def set_rewards(
        self,
        new_rewards: np.ndarray,
        rollout_buffer: CustomMaskableRolloutBuffer,
        last_values: torch.Tensor,
        last_dones: np.ndarray,
        callback: "TransitionCounterCallback",
    ):
        # We need to keep the terminated value in case. The true rewards are there for that
        rollout_buffer.rewards = (
            rollout_buffer.rewards - rollout_buffer.true_rewards
        ) + new_rewards
        # End
        rollout_buffer.compute_returns_and_advantage(
            last_values=last_values, dones=last_dones
        )

        callback.update_locals(locals())
        callback.on_rollout_end()

    def dump_logs(self, iteration: int) -> None:
        self._dump_logs(iteration)

    def setup_learn(
        self,
        total_timesteps: int,
        callback=None,
        tb_log_name: str = "PPO",
        reset_num_timesteps: bool = True,
    ):
        total_timesteps, callback = self._setup_learn(
            total_timesteps,
            callback,
            reset_num_timesteps,
            tb_log_name,
            progress_bar=False,
        )
        callback.on_training_start(locals(), globals())

        assert self.env is not None
        return total_timesteps, callback


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
        TRANSITION_TYPE_TO_ID = {
            "nothing": 0,
            "standing": 1,
            "holding1": 2,
            "holding2": 3,
            "transformP": 4,
            "transformSH": 5,
            "transformBH": 6,
        }  # From playground text wrapper
        old_rewards = self.locals["rollout_buffer"].true_rewards
        new_rewards = self.locals["rollout_buffer"].rewards
        # Take the mean of the transition type
        for transition_type, transition_id in TRANSITION_TYPE_TO_ID.items():
            self.logger.record(
                f"rewards_debug/{transition_type}",
                new_rewards[np.where(old_rewards == transition_id)].mean(),
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


class SB3Agent:
    def __init__(self, model: CustomMaskablePPO, callback: TransitionCounterCallback):
        self.model = model
        self.callback = callback
        self.iteration = 0

        self._last_values: torch.Tensor
        self._last_dones: np.ndarray

    def __call__(self, obs, **info):
        return self.model.predict(obs, action_mask=info["action_mask"])

    def generate_trajectories(self, n_steps: int) -> Tuple[List[Trajectory], Set[str]]:
        """Generate trajectories using the agent

        Args:
            n_steps (int): The number of steps per env to generate.
        """
        (
            trajectories_text,
            set_discovered_transitions,
            self._last_values,
            self._last_dones,
        ) = self.model.collect_rollouts(
            self.model.env,
            self.callback,
            self.model.rollout_buffer,
            int(n_steps / self.model.n_envs),
            use_masking=True,
        )
        # Flatten the list of trajectories
        prompt_trajectories = [
            traj for sublist in trajectories_text for traj in sublist
        ]
        return prompt_trajectories, set_discovered_transitions

    def train_step(self, new_rewards: np.ndarray):
        # Modify reward before training
        self.model.set_rewards(
            new_rewards,
            self.model.rollout_buffer,
            self._last_values,
            self._last_dones,
            self.callback,
        )

        self.iteration += 1
        self.model.dump_logs(self.iteration)
        self.model.train()


def create_agent(config: Dict[str, Any], build_env: Callable, seed: int) -> SB3Agent:
    """Create SB3 agent and setup for learning"""
    # Set seed
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    random.seed(seed)

    # Define actions for masking
    def mask_fn(env):
        return env.unwrapped.action_mask

    def make_env():
        env = ActionMasker(build_env(), mask_fn)
        return env

    envs = make_vec_env(make_env, n_envs=config["n_envs"], seed=seed)
    hyperparameters = config["hyperparameters"]
    model = CustomMaskablePPO(
        "MlpPolicy",
        envs,
        rollout_buffer_class=CustomMaskableRolloutBuffer,
        gamma=hyperparameters["gamma"],
        learning_rate=hyperparameters["lr"],
        ent_coef=0.01,
        vf_coef=hyperparameters["vf_coef"],
        n_steps=hyperparameters["n_steps"],
        n_epochs=hyperparameters["n_epochs"],
        gae_lambda=hyperparameters["gae"],
        device="cuda",
        verbose=1,
        tensorboard_log=config["tb_folder"],
        policy_kwargs=dict(
            net_arch=dict(
                pi=[hyperparameters["nn_size"], hyperparameters["nn_size"]],
                vf=[hyperparameters["nn_size"], hyperparameters["nn_size"]],
            )
        ),
        seed=seed,
    )
    # Prepare learning before the pipeline
    _, callback = model.setup_learn(
        total_timesteps=10_000_000,  # Shouldn't be important
        tb_log_name=config["tb_name"],
        callback=TransitionCounterCallback(
            hyperparameters, model.verbose, config["n_envs"]
        ),
    )
    return SB3Agent(model, callback)