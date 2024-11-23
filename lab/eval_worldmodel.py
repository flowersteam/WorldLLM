import json
import os
import sys

import gymnasium
from tqdm import tqdm

from worldllm_envs.base import BaseRuleEnv, RandomAgent, Trajectory
from worldllm_envs.playground.playground_text_wrapper import PerfectAgent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils_llm import (
    Statistician,
    build_stat_prompt_info,
    compute_likelihood,
    load_transformers,
)

SEED = 15
NB_EPISODES = 3
BATCH_SIZE = 20

# Load the environment
env: BaseRuleEnv = gymnasium.make(
    "worldllm_envs/PlaygroundText-v1",
    **{"seed": SEED, "test_dataset_path": None, "max_steps": 30},
)
new_rule = env.generate_rule("Grow any small_herbivorous then grow any big_herbivorous")
env.reset(options={"rule": new_rule})

# Load the LLM
config = {
    "name": "microsoft/Phi-3-mini-4k-instruct",
    "model_params": {},
    "tokenizer_params": {},
    "is_quantized": True,
    "generation_kwargs": {"cache_implementation": None, "cache_config": None},
    "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
}
world_model = load_transformers(config)
stat_prompt_info = build_stat_prompt_info(
    world_model,
    env.get_message_info(),
    "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
    BATCH_SIZE,
)
statistician = Statistician(
    world_model[0],
    world_model[1],
    stat_prompt_info,
)
rules_to_test = [
    # Norule
    None,
    # Rule ALP
    '1. If no state change, say "Nothing has changed."\n2. If you stand on an object, say "You are standing on the [object]."\n3. If you end up on a new object, say "You are standing on the [new object]."\n4. Continue if no active actions or progress is made, repeat "Nothing has changed."\n5. If an object transforms, say "The [object] and the water transform into the [new object]."',
    '1. If no state change, say "Nothing has changed."\n2. If you stand on an object, say "You are standing on the [object]."\n3. If you would have ended up on a new object, say "You are standing on the [object]."\n4. Continue if no active actions or progress is made, repeat "Nothing has changed."\n5. If an object transforms, say "The [object] and the water transform into the [new object]."',
    '1. If no state change, say "Nothing has changed."\n2. If you stand on an object, say "You are standing on the [object]."\n3. If you would have ended up on a new object, say "You are standing on the [object]."\n4. Continue if no active actions or progress is made, repeat "Nothing has changed."\n5. If an object transforms when you are on it, say "The [object] and the water transform into the [new object]."',
]
algorithm_used = [None, "ALP", "ALP", "ALP"]
# Create the different agents

perfect_agent_sh = PerfectAgent(
    env.action_space, curriculum_goals=["Grow any small_herbivorous"]
)
perfect_agent_shbh = PerfectAgent(
    env.action_space,
    curriculum_goals=["Grow any small_herbivorous then grow any big_herbivorous"],
)
random_agent = RandomAgent(env.action_space)
all_scores = []
all_index = []
all_transitions_type = []
pbar = tqdm(range(3 * NB_EPISODES), desc="Generating trajectories")
for agent in [perfect_agent_sh, perfect_agent_shbh, random_agent]:
    for _ in range(NB_EPISODES):
        all_scores_per_episode = []
        all_index_per_episode = []
        all_transitions_type_per_episode = []
        obs, info = env.reset()
        reset_info = {"pipeline_progression": 0}
        info.update(reset_info)
        # Compute plan
        agent.reset(info)
        done = False
        while not done:
            # Record inputs from keyboard
            # Print possible actions
            action, agent_done = agent(obs, **info)
            all_transitions, transitions_type = (
                env.unwrapped.get_all_possible_transitions()
            )
            # Get the score of the LLM for all transitions
            lst_possible_trajectories = [
                Trajectory(
                    lst_obs=[
                        info["trajectory_obs_text"][-1],
                        "",
                    ],  # We don't use the last observation
                    lst_act=[env.action_to_text(action)],
                    lst_diff=[transition],
                )
                for transition in all_transitions
            ]
            (_, logp), _ = compute_likelihood(
                statistician,
                rules_to_test,
                lst_possible_trajectories,
                return_all_logp=True,
            )
            all_scores_per_episode.append(logp.tolist())
            all_transitions_type_per_episode.append(transitions_type)
            obs, reward, terminated, truncated, info = env.step(action)
            true_diff = info["trajectory_diff_text"][-1]
            index_true = all_transitions.index(true_diff)
            all_index_per_episode.append(index_true)
            done = terminated or truncated or agent_done
        pbar.update(1)
        all_scores.append(all_scores_per_episode)
        all_index.append(all_index_per_episode)
        all_transitions_type.append(all_transitions_type_per_episode)

# Save the scores to json
with open("./outputs/scores.json", "w", encoding="utf-8") as f:
    json.dump(
        {
            "scores": all_scores,
            "rules": rules_to_test,
            "algorithm_used": algorithm_used,
            "true_obs_index": all_index,
            "transition_type": all_transitions_type,
        },
        f,
    )

print("Done.")
