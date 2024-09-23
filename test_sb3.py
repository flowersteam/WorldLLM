import gymnasium
from sb3_contrib import MaskablePPO

# This is a drop-in replacement for EvalCallback
from sb3_contrib.common.maskable.callbacks import MaskableEvalCallback
from sb3_contrib.common.maskable.evaluation import evaluate_policy
from sb3_contrib.common.maskable.utils import get_action_masks
from sb3_contrib.common.wrappers import ActionMasker
from stable_baselines3.common.preprocessing import get_flattened_obs_dim

from worldllm_envs.base import BaseRuleEnv


def mask_fn(env):
    return env.unwrapped.action_mask


seed = None
env: BaseRuleEnv = gymnasium.make(
    "worldllm_envs/Playground-v1",
    **{"max_steps": 30, "seed": seed, "playground_config": {"max_nb_objects": 8}},
)
new_rule = "grow any small_herbivorous then grow any big_herbivorous"
env.reset(options={"rule": new_rule})
env = ActionMasker(env, mask_fn)  # Wrap to enable masking

# # Train PPO
model = MaskablePPO(
    "MlpPolicy",
    env,
    gamma=0.99,
    learning_rate=0.001,
    ent_coef=0.1,
    vf_coef=0.1,
    n_steps=256,
    n_epochs=4,
    gae_lambda=0.9,
    device="cuda",
    verbose=1,
    tensorboard_log="./logs_ppo_sb3",
)
print(model.policy)
model.learn(200_000, tb_log_name="PPO_Test", progress_bar=True)
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
