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
    return env.action_mask


seed = None
env: BaseRuleEnv = gymnasium.make(
    "worldllm_envs/Playground-v1",
    **{"max_steps": 30, "seed": seed, "playground_config": {"max_nb_objects": 8}},
)
new_rule = "grow any small_herbivorous then grow any big_herbivorous"
env.reset(options={"rule": new_rule})
env = ActionMasker(env, mask_fn)  # Wrap to enable masking
model = MaskablePPO("MlpPolicy", env, gamma=0.4, seed=32, verbose=1)
model.learn(1_000_000)

evaluate_policy(model, env, n_eval_episodes=20)

model.save("ppo_mask")
del model  # remove to demonstrate saving and loading

model = MaskablePPO.load("ppo_mask")

obs, _ = env.reset()
while True:
    # Retrieve current action mask
    action_masks = get_action_masks(env)
    action, _states = model.predict(obs, action_masks=action_masks)
    obs, reward, terminated, truncated, info = env.step(action)
