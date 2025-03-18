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
    configs.append(
        {
            "name": args.base_model_path,
            "use_unsloth": False,
            "model_params": {},
            "tokenizer_params": {},
            "is_quantized": True,
            "max_seq_len": 4096,
            "generation_kwargs": {"cache_implementation": None, "cache_config": None},
            "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
        }
    )
    rules_to_test = [
        # LL
        "1. If you are on water, the state changes to: You are standing on the water.\n2. If you pick up the object in water, the state changes to: In your inventory, there are the water and the water.\n3. If you go to any other seed, the state changes to: You are standing on the seed.\n4. If you give water to any seed, it transforms into a plant.",
        "1. The action was: You give the pea seed.\nThe change was: In your inventory, there is the pea (from the pea seed).\n2. The action was: You go to the berry seed.\nThe change was: You are standing on the berry seed.\n3. The action was: You give the pea seed.\nThe change was: The objects transform into the berry (from the pea seed).\n4. The action was: You go to the water.\nThe change was: You are standing on the water.\n5. The action was: You go to the beet seed.\nThe change was: You are standing on the beet seed.\n6. The action was: You give the water.\nThe change was: The objects transform into the beet.\n\nThe rules to predict the change based on your actions are as follows:\n\n- Picking up an",
        "1. Standing on the water -> Pick up the water (object becomes water).\n\n2. Go to the water -> Pick up the water (object becomes water).\n\n3. Go to the pea seed -> Pick up the pea seed (object becomes pea).\n\n4. Go to the pea seed -> Pick up the pea seed (object becomes pea).\n0 Action: You go to the water.\nChange: You are standing on the water.]\n0 Action: You give the water.\nChange: The objects transform into the pea.]\n0 Action: You pick up the object.\nChange: In your inventory, there are the water and the pea.]\n0 Action: You go to the pea seed.\nChange: You are standing on the pea seed.]\n0 Action: Giving the water turns the objects into the pea.\nChange: The objects transform into the potato.]",
        "1. Hold one object - when picking up an item, you can only hold one item at a0000\n\nAction: You go to the <Object>.\nChange: You are standing on the <Object>.\n\n2. Keep both peas - regardless of holder, both peas can be transformed into different objects in an inventory by picking them up separately.\nAction: You pick up the pea.\nChange: In your inventory, there is the pea.\nAction: You go to the pea seed.\nChange: You are standing on the pea seed.\nAction: You give the water.\nChange: The objects transform into the pea.\nAction: You pick up the object.\nChange: In your inventory, there are the pea and the potato.\nAction: You go to the potato seed.\nChange: You are standing on the potato seed.\nAction:",
        "1. In your inventory, you have nothing.\n2. You pick up the object.\n3. In your inventory, there is the object.\n4. You go to the water.\n5. You are standing on the water.\n6. You pick up the object.\n7. In your inventory, there are the object and the object.\n8. You go to the potato seed.\n9. You are standing on the potato seed.\n10. You give the object.\n11. The objects transform into the potato.\n12. You go to the water.\n13. You are standing on the water.\n14. You pick up the object.\n15. In your inventory, there are the water and the object.\n16. You go to the berry seed.\n17. You are standing on the berry seed.\n18. You",
        '1. When you want to learn about an object, the action will be "you go to the [object]." The change will be "you are standing on the [object]."\n\n2. When you want to observe an object, the action will be "you pick up the object." The change will be "In your inventory, there are [water] and the [object]."\n\n3. When you give water to a seed, the action will be "you give the water." The change will be "the objects transform into the [object]."\n\n4. When you want to go back to the water, the action will be "you give the water." This will transform the objects in your inventory as before.',
        '1. Stand on an object. Picking up an object keeps you standing on that same object. The inventory increases by one on this object.\n2. Action: "The action was: You pick up the object. The change was: In your inventory, there are the water and the beet." \nFrom this, we can infer that picking up the beet transforms it into the beet, and picking up the water keeps you standing on it.\n3. Action: "The action was: You go to the water. The change was: You are standing on the beet." \nThis indicates that standing on the beet does not transform it, but being on the same object after picking the beet is possible.\n4. Action: "The action was: You give the beet. The change was: The objects transform into the beet."\nThis shows that giving the beet keeps it in its transformed state',
        "1. Go to the water.\nChange: You are standing on the water.\n2. Pick up the object.\nChange: In your inventory, there is the water.\n3. Go to the water.\nChange: You are standing on the water.\n4. Pick up the object.\nChange: In your inventory, there are the water and the water.\n5. Go to the berry seed.\nChange: You are standing on the berry seed.\n6. Give the water.\nChange: The objects transform into the berry.\n7. Go to the water.\nChange: You are standing on the water.\n8. Pick up the object.\nChange: In your inventory, there are the water and the water.\n9. Go to the pea seed.\nChange: You are standing on the pea seed.\n10. Give the water.\nChange: The objects transform into",
        # ALP
        "1. If you move to a baby pig, output: You are standing on the baby pig.\n2. If you move to a baby giraffe, output: You are standing on the baby giraffe.\n3. If you move to a water, output: You are standing on the water.\n4. If you go to a baby pig, output: You are standing on the baby pig.\n5. If you go to a baby giraffe, output: You are standing on the baby giraffe.\n6. If you move to a baby pig, output: You are standing on the baby pig.\n7. If you move to a baby giraffe, output: You are standing on the baby giraffe.\n8. If you pick up the object, output: In your inventory, there are the baby pig and the berry seed.",
        '1. When you stand on the baby elephant, the action changes to "you are standing on the baby elephant."\n2. When you go to the water, the action changes to "you are standing on the water."\n3. When you go to the water, the action changes to "you are standing on the water."\n4. When you go to the water, the action changes to "you are standing on the water."\n5. When you go to the pea seed, the action changes to "you are standing on the pea seed."\n6. When you pick up the pea seed, the action changes to "in your inventory, there is the pea seed."\n7. When you go to the baby elephant, the action changes to "you are standing on the baby elephant."\n8. When you pick up the pea seed, the action changes to "in your inventory, there',
        "1. You start by standing on the object.\n\n2. When you stand on a carrot seed, the change is: You are standing on the carrot seed.\n\n3. When you go to the carrot seed and stand on it, the change is: You are standing on the carrot seed.\n\n4. When you go to the carrot seed, the change is: You are standing on the carrot seed.\n\n5. When you go to the water, the change is: You are standing on the water.\n\n6. When you go to the baby cow, the change is: You are standing on the baby cow.\n\n7. When you go to the water, the change is: You are standing on the water.\n\n8. When you go to the baby cow, the change is: You are standing on the baby cow.\n\n9. When you go to the baby cow, the change is",
        "1. Moving from an object keeps you on the same object.\n2. Going to an object changes your location to that object.\n3. If you are standing on an object and go to another object of the same kind, you are still standing on the last object.\n4. If you are standing on any object and go back to the same type of object, you are still standing on that object.\n\nAction: You give the water.\nThe change was: The objects transform into the water.\nAction: You are standing on the beet.\nThe change was: You are standing on the beet.\nAction: You go to the beet.\nThe change was: You are standing on the beet.\nAction: You go to the pea seed.\nThe change was: You are standing on the pea seed.\nAction: You go to the baby sheep.\nThe change was: You are standing on the",
        "1. If you are standing on a baby tree, go to the baby giraffe.\n2. If you are standing on a baby giraffe, stay standing.\nThe most recent set of actions that failed the test were:\nFirst state: You see the baby cow, the water, the carrot seed, the baby rhinoceros, the water, the potato seed, the water, the carrot seed. In your inventory, there is nothing.. \nThe sequence:\n- You go to the water. Change: You are standing on the water.\n- You go to the potato seed. Change: You are standing on the potato seed.\n- You go to the potato seed. Change: You are standing on the potato seed.\n- You go to the water. Change: You are standing on the water.\n- You go to the water. Change: You are standing on the water.\n-",
        "The action was: You pick up the insignia.\nChange was: In your inventory, the Baby cow.\nThe action was: You go to the potato seed.\nChange was: You are standing on the potato seed.\nThe action was: You go to the water.\nChange was: You are standing on the water.\nThe action was: You go to the water.\nChange was: You are standing on the water.\nThe action was: You go to the baby cow.\nThe change was: You are standing on the baby cow.\nThe action was: You go to the baby cow.\nThe change was: You are standing on the baby cow.\nThe action was: You go to the baby cow.\nThe change was: You are standing on the baby cow.\nThe action was: You go to the baby cow.\nThe change was: You are standing on the baby cow.\nThe action was:",
        '1. Walking on a flat, dry surface does not change your state.\n2. Going to a baby animal (e.g., baby pig, baby sheep, baby elephant) changes your state to standing on that animal.\n3. Walking on the same animal changes your state.\n4. Water always changes your state to the state you are standing on the water.\n5. You only stand on the area if it\'s an animal and not water.\n\n The first action was, "You go to the baby pig."\nThe first change was, "You are standing on the baby pig."\nThe first action was, "You go to the baby pig."\nThe first change was, "You are standing on the baby pig."\nThe first action was, "You go to the baby pig."\nThe first change was, "You are standing on the baby pig."\nThe first action',
        "1. If you see a baby pig, The change was: You are standing on the baby pig.\n2. Go to the beet seed. The change was: You are standing on the beet seed.\n3. Go to the potato seed. The change was: You are standing on the potato seed.\n4. Pick up the object. The change was: In your inventory, there is the potato seed.\n5. Go to the baby giraffe. The change was: You are standing on the baby giraffe.\n6. Go to the baby pig. The change was: You are standing on the baby pig.\n7. Go to the potato seed. The change was: You are standing on the potato seed.\n8. Go to the beet seed. The change was: You are standing on the beet seed.\n9. Go to the potato seed. The change",
        # ALPEXP
        "1. You are on an object or water in your inventory.\n2. If you move onto an object, you remain on that object.\n3. If you give an object to a baby animal, it transforms into that animal.\n4. You cannot have any animals (babies or adults) in your inventory.\n\nFor the last scenario (you see the baby pig, the water, the berry seed, the baby rhinoceros, the water, the pea seed, the water, the beet seed; in your inventory, there is nothing), here were the corresponding actions and changes according to these simple IF rules:\n\nActions and Changes:\n\n1. You go to the beet seed.  \n   Change: You are standing on the beet seed.\n\n2. You pick up the object.\n   Change: In your inventory, there is the beet seed.\n",
        "1. You can only pick up one item at a time.\n2. You cannot pick up an item that transforms into the same item.\n3. If you are standing on an object that transforms into the same item, you need to go to a different object to avoid confusion.\n4. The first state is: You see the baby sheep, the water, the beet seed, the baby elephant, the water, the berry seed, the water, the beet seed. In your inventory, there is nothing.\n5. The sequence of actions and states is: The action was: You go to the water. The change was: You are standing on the water.\n6. The action was: You pick up the object. The change was: In your inventory, there is the water.\n7. The action was: You go to the beet seed. The change was: You are standing on the beet seed",
        "1. Walking on the water -> You are standing on the water.\n2. Picking up an object -> In your inventory, there is the object.\n3. Giving nothing -> Nothing happens.\n4. Picking up an object -> In your inventory, there is the object.\n5. Giving water -> The objects transform into the pea.\n6. Going to the water -> You are standing on the water.\n7. Giving the object -> The objects transform into the water.\n8. Going to the berry seed -> You are standing on the berry seed.\n9. Going to the beet seed -> You are standing on the beet seed.\n10. Giving nothing -> Nothing happens.\n11. Picking up the object -> In your inventory, there are the beet and the beet.\n12. Giving the object -> The objects transform into the",
        "1. Only one object can be taken into the inventory at a0800Z:You are standing on the water.\nThe action was: You pick up the object.\nThe change was: In your inventory, there are the water and the water.\nThe action was: You go to the carrot seed.\nThe change was: You are standing on the carrot seed.\nThe action was: You give the water.\nThe change was: The objects transform into the carrot.\nThe action was: You pick up the object.\nThe change was: In your inventory, there are the water and the carrot.\nThe action was: You go to the baby giraffe.\nThe change was: You are standing on the baby giraffe.\nThe action was: You go to the water.\nThe change was: You are standing on the water.\nThe action was: You go to the water.",
        "The first state was: You see the baby pig, the water, the berry seed, the baby giraffe, the water, the beet seed, the water, the beet seed. In your inventory, there is nothing. \nAvailable action is: You go to the baby pig. The change was: You are standing on the baby pig.\nThe action was: You go to the water.\nThe change was: You are standing on the water.\nThe action was: You pick up the object. The change was: In your inventory, there is the water.\nThe action was: You go to the water.\nThe change was: You are standing on the water.\nThe action was: You go to the water.\nThe change was: You are standing on the water.\nThe action was: You go to the beet seed.\nThe change was: You are standing on the beet seed.\n",
        "- You go to [something]: You are standing on [something].\n- You pick up [object]: In your inventory, there are [object].\nThe first state was: You see the baby sheep, the water, the pea seed, the baby rhinoceros, the water, the beet seed, the water, the berry seed. In your inventory, there is nothing.\nThe sequence of actions and states was:\n1. The action was: You go to the pea seed.\nChange: You are standing on the pea seed.\n2. The action was: You go to the pea seed.\nChange: You are standing on the pea seed.\n3. The action was: You give the water.\nChange: The objects transform into the pea.\n4. The action was: You pick up the object.\nChange: In your inventory, there are the pea and the pea",
        "Start by walking to any object to stand on it. Beating the previous slate states: OBJECT will always transform into the next one as you move.\n\nSequence: The action was: You go to the water.\nChange was: You are standing on the water.\nThe action was: You go to the beet seed.\nChange was: You are standing on the beet seed.\nThe action was: You go to the pea seed.\nChange was: You are standing on the pea seed.\n\nThe action was: You pick up the object.\nChange was: In your inventory, there are the water and the pea seed.\nThe action was: You go to the pea seed.\nChange was: You are standing on the pea seed.\n\nThe action was: You give the water.\nChange was: The objects transform into the pea.\nThe action was: You pick up the object",
        "1. If you see a water, potion, or plant with a single animal or seed in your inventory, and the action is to pick up the object, the state remains unchanged.\n2. If you go to any of the aforementget_scene_state=((baby pig, potion, beet seed, baby elephant, water, carrot seed, water, pea seed))\nthe action was: You go to the pea seed.\nThe change was: You are standing on the pea seed.\nThe action was: You go to the beet seed.\nThe change was: You are standing on the beet seed.\nThe action was: You go to the water.\nThe change was: You are standing on the water.\nThe action was: You give the pea seed.\nThe change was: The objects transform into the pea.\nThe action was: You pick up",
    ]
    all_rules_to_test.append(rules_to_test)
    algorithm_used: List[Optional[str]] = []
    for key in ["LL", "ALP", "ALPEXP"]:
        algorithm_used.extend([key] * 8)
    all_algorithm_used.append(algorithm_used)
    assert len(rules_to_test) == len(algorithm_used)
    # endregion

    # region Add finetuned LLM
    if (
        args.finetuned_model_paths is not None
        and args.finetuned_model_names is not None
    ):
        for model_path, algo_name in zip(
            args.finetuned_model_paths, args.finetuned_model_names
        ):
            configs.append(
                {
                    "name": args.finetuned_model_paths,
                    "use_unsloth": False,
                    "model_params": {},
                    "tokenizer_params": {},
                    "is_quantized": True,
                    "max_seq_len": 4096,
                    "generation_kwargs": {"cache_implementation": None, "cache_config": None},
                    "chat_template": "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}",
                }
            )
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
