from functools import partial
from typing import Any, Dict

import gymnasium
from sb3_contrib import MaskablePPO

# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.callbacks import BaseCallback
from stable_baselines3.common.env_util import make_vec_env
from stable_baselines3.common.logger import TensorBoardOutputFormat
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from worldllm_envs.base import BaseRuleEnv


class TransitionCounterCallback(BaseCallback):
    def __init__(self, verbose=0):
        super(TransitionCounterCallback, self).__init__(verbose)
        self.transition_counts = {}

    def _on_step(self) -> bool:
        # Access the current observation and info
        lst_infos = self.locals.get("infos")
        lst_rewards = self.locals.get("rewards")
        for info, reward in zip(lst_infos, lst_rewards):
            # Assuming you have a way to get the current transition type
            self.transition_counts[info["transition_type"]] = (
                self.transition_counts.get(info["transition_type"], 0) + 1
            )
            # Log the transition counts to TensorBoard
            for transition_type, count in self.transition_counts.items():
                self.logger.record(f"transitions/{transition_type}", count)
            self.logger.record(f"rewards/{info['transition_type']}", reward)

        return True


def mask_fn(env):
    return env.unwrapped.action_mask


seed = None
# Load first environment
env: BaseRuleEnv = gymnasium.make(
    "worldllm_envs/Playground-v1",
    **{"max_steps": 30, "seed": seed, "playground_config": {"max_nb_objects": 8}},
)
new_rule = "grow any small_herbivorous then grow any big_herbivorous"
env.reset(options={"rule": new_rule})
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

countbased = env.unwrapped.get_shared_countbased()


def make_env(countbased_dict: Dict[str, int]):
    env: BaseRuleEnv = gymnasium.make(
        "worldllm_envs/Playground-v1",
        **{"max_steps": 30, "seed": seed, "playground_config": {"max_nb_objects": 8}},
    )
    new_rule = "grow any small_herbivorous then grow any big_herbivorous"
    env.reset(options={"rule": new_rule})
    env = ActionMasker(env, mask_fn)  # Wrap to enable masking
    env.unwrapped.set_shared_countbased(countbased_dict)
    return env


envs = make_vec_env(
    partial(make_env, countbased_dict=countbased),
    n_envs=9,
)
# Train PPO
model = MaskablePPO(
    "MlpPolicy",
    envs,
    gamma=0.99,
    learning_rate=0.0005,
    ent_coef=0.01,
    vf_coef=0.1,
    n_steps=256,
    n_epochs=10,
    gae_lambda=0.9,
    device="cuda",
    verbose=1,
    tensorboard_log="./logs_ppo_sb3",
)
callback = TransitionCounterCallback(model.verbose)
model.learn(50_000, tb_log_name="PPO_Test_vecEnv", progress_bar=True, callback=callback)
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
