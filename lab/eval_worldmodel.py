import argparse
import gc
import json
import os
import sys
from typing import Any, Dict, List, Optional

import gymnasium
import torch
from tqdm import tqdm

from worldllm_envs.base import BaseWrapper, RandomAgent, Trajectory
from worldllm_envs.playground.playground_text_wrapper import PerfectAgent

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))


from utils.utils_llm import (
    Statistician,
    build_stat_prompt_info,
    compute_likelihood,
    load_transformers,
)


def flatten_list(l):
    """Flatten list of list"""
    return [item for sublist in l for item in sublist]


def concatenate_third_axis(score1, score2):
    """Concatenate two 4D lists along the third axis"""
    # Ensure both lists have the same number of outer lists
    if len(score1) != len(score2):
        raise ValueError("The outer lists must have the same length")

    concatenated_list = []
    for inner_list1, inner_list2 in zip(score1, score2):
        # Ensure both inner lists have the same number of sublists
        if len(inner_list1) != len(inner_list2):
            raise ValueError("The inner lists must have the same length")

        concatenated_inner_list = []
        for sublist1, sublist2 in zip(inner_list1, inner_list2):
            # Concatenate along axis 2
            concatenated_sublist = sublist1 + sublist2
            concatenated_inner_list.append(concatenated_sublist)

        concatenated_list.append(concatenated_inner_list)

    return concatenated_list


CONFIG_LLM = {
    "use_unsloth": False,
    "model_params": {},
    "tokenizer_params": {},
    "is_quantized": True,
    "max_seq_len": 4096,
    "generation_kwargs": {"cache_implementation": None, "cache_config": None},
    "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
}

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-name",
        "-o",
        type=str,
        default="scores_modified.json",
        help="Name of the output file",
    )
    parser.add_argument(
        "--nb-episodes",
        "-n",
        type=int,
        default=3,
        help="Number of episodes for each agent. Need to be the same as the one used for the testset",
    )
    parser.add_argument(
        "--batch-size", "-b", type=int, default=30, help="batch size of the model"
    )
    parser.add_argument(
        "--seed",
        "-s",
        type=int,
        default=42,
        help="Seed for the environment. Need to be the same as the one used for the testset",
    )
    parser.add_argument(
        "--finetuned_model_paths",
        action="append",
        help="Paths to the finetuned models",
    )
    parser.add_argument(
        "--finetuned_model_names",
        action="append",
        help="Saved names of the finetuned models",
    )
    parser.add_argument(
        "--base_model_path",
        type=str,
    )
    args = parser.parse_args()

    # Add configs to test
    configs: List[Dict[str, Any]] = []
    all_rules_to_test: List[List[Optional[str]]] = []
    all_algorithm_used: List[List[Optional[str]]] = []
    # region Add not finetuned LLM
    configs.append(CONFIG_LLM | {"name": args.base_model_path})
    rules_to_test = []
    if len(rules_to_test) > 0:
        all_rules_to_test.append(rules_to_test)
        algorithm_used: List[Optional[str]] = []
        for key in []:
            algorithm_used.extend([key] * 8)
        all_algorithm_used.append(algorithm_used)
        assert len(rules_to_test) == len(algorithm_used)
    # endregion

    # region Add finetuned LLM
    if (
        args.finetuned_model_paths is not None
        and args.args.finetuned_model_paths is not None
    ):
        for model_path, algo_name in zip(
            args.finetuned_model_paths, args.finetuned_model_names
        ):
            configs.append(CONFIG_LLM | {"name": model_path})
            all_rules_to_test.append([None])
            all_algorithm_used.append([algo_name])
    # endregion
    # # Main loop to evaluate the different world models
    all_scores = []
    all_index = []
    all_transitions_type = []
    all_possible_mask = []
    print(f"{len(rules_to_test)} world models to evaluate.")
    for incr, (config, rules_to_test, algorithm_used) in enumerate(
        zip(configs, all_rules_to_test, all_algorithm_used)
    ):
        print("Evaluating model", config["name"])
        print(f"Number of rules to test: {len(rules_to_test)}")
        # Load the environment
        env: BaseWrapper = gymnasium.make(
            "worldllm_envs/PlaygroundText-v1",
            **{"seed": args.seed, "test_dataset_path": None, "max_steps": 30},
        )
        new_rule = env.unwrapped.generate_rule(
            "Grow any small_herbivorous then grow any big_herbivorous"
        )
        env.reset(options={"rule": new_rule})
        # Load the LLM
        world_model = load_transformers(config)
        stat_prompt_info = build_stat_prompt_info(
            world_model,
            env.unwrapped.get_message_info(),
            "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
            args.batch_size,
        )
        statistician = Statistician(
            world_model[0],
            world_model[1],
            stat_prompt_info,
        )
        # Add all the abstract trajectories
        statistician.prompt_info.discovered_transitions = (
            env.unwrapped.get_all_transition_to_prompt()
        )
        # Create the different agents
        perfect_agent_sh = PerfectAgent(
            env.action_space, curriculum_goals=["Grow any small_herbivorous"]
        )
        perfect_agent_shbh = PerfectAgent(
            env.action_space,
            curriculum_goals=[
                "Grow any small_herbivorous then grow any big_herbivorous"
            ],
        )
        random_agent = RandomAgent(env.action_space)
        nb_episodes = args.nb_episodes
        pbar = tqdm(range(3 * nb_episodes), desc="Generating trajectories")
        episode_counter = 0
        all_score_config = []
        for agent in [perfect_agent_sh, perfect_agent_shbh, random_agent]:
            for _ in range(nb_episodes):
                all_scores_per_episode = []
                all_index_per_episode = []
                all_transitions_type_per_episode = []
                all_possible_mask_per_episode = []
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
                    all_transitions, transitions_type, possible_transition_mask = (
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
                    all_possible_mask_per_episode.append(possible_transition_mask)
                    obs, reward, terminated, truncated, info = env.step(action)
                    true_diff = info["trajectory_diff_text"][-1]
                    index_true = all_transitions.index(true_diff)
                    all_index_per_episode.append(index_true)
                    done = terminated or truncated or agent_done
                pbar.update(1)
                all_score_config.append(all_scores_per_episode)
                if incr == 0:
                    all_index.append(all_index_per_episode)
                    all_transitions_type.append(all_transitions_type_per_episode)
                    all_possible_mask.append(all_possible_mask_per_episode)
                else:
                    # Check if the same transitions
                    assert all_index_per_episode == all_index[episode_counter]
                    assert (
                        all_transitions_type_per_episode
                        == all_transitions_type[episode_counter]
                    )
                    assert (
                        all_possible_mask_per_episode
                        == all_possible_mask[episode_counter]
                    )
                episode_counter += 1
        # Fusion score
        if incr == 0:
            all_scores = all_score_config
        else:
            all_scores = concatenate_third_axis(all_scores, all_score_config)

        # Empty gpu cache
        del statistician
        del stat_prompt_info
        del world_model
        gc.collect()
        torch.cuda.empty_cache()

    # Save the scores to json
    with open(args.output_name, "w", encoding="utf-8") as f:
        json.dump(
            {
                "scores": all_scores,
                "rules": flatten_list(all_rules_to_test),
                "algorithm_used": flatten_list(all_algorithm_used),
                "true_obs_index": all_index,
                "transition_type": all_transitions_type,
                "possible_transition_mask": [
                    [[int(z) for z in x] for x in y] for y in all_possible_mask
                ],
            },
            f,
        )

    print("Done.")
