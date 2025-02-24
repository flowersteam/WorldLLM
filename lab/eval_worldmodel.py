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
        # Norule
        None,
        # PB
        "1. Standing on an object transforms it into that object.\n2. Picking up an object leaves the environment unchanged (even empty inventory).\n3. Picking up all available objects leaves them unchanged.\n4. After holding all available objects, if the environment contains only water, beet, pea, or seed (choosing one animal to transform into), the objects transform into that animal.\n\nThese observations are informed by the provided sequences:\n\n- In the first example, the object turned into the animal regardless of what it was (water, seed, or vegetable).\n- In the second example, if the environment consists only of a single object (e.g., beet, beet seed, or carrot seed), all other objects remain unchanged after all have been picked.\n- In the third example, if the environment contains more than one type of visible object (i.e., berries, pot",
        "1. Standing on an object in your inventory transforms that object into the first thing you transform.\n2. Go to the water.\n3. Pick up the object - In your inventory, there is the water.\n4. Go to the beet seed - You are standing on the beet seed.\n5. Give the water - The objects transform into the beet.\n6. Pick up the object - In your inventory, there is the beet.\n7. Go to the baby pig - You are standing on the baby pig.\n8. Give the beet - The objects transform into the pig.\n9. Go to the water - You are standing on the water.\n10. Pick up the object - In your inventory, there is the water.\n\nThese general rules apply to all sequences of actions and states.",
        '# Environment Transformation Rules:\n\n1. Standing on an object turns it into that type of object.\n2. Picking up an object does not change its state; the object remains the one you stand on.\n3. Replace the object\'s name (for example, "objects transform into ...").\n4. Only actions with a state change should be noted.\n\nBased on these rules for the given sequences, the transformations would be as follows:\n\n1. You go to the water. You are standing on the water.\n2. You pick up the object (water). In your inventory, there is water.\n3. You go to the (object you stand on). You are standing on the (object you stand on).\n4. You give the water. The objects transform into the (first item you stood on).\n5. You pick up the object. In your inventory, there is the (last tied object',
        "1. Standing on a vegetable or its derivative (e.g., seeds) doesn't alter the state of the inventory.\n2. Consuming a vegetable or seed transforms it into the corresponding vegetable or animal (e.g., berry seed into the berry).\n3. Keeping your vegetable or seed in your holdings doesn't alter the state of objects not being vegetables or their derivatives.\n4. Standing on or keeping space empty doesn't alter the state of objects not being vegetables or their derivatives.\n5. For seeds, the objects transform into the animals they are on or the corresponding farm animal (e.g., transform into the baby pig, baby elephant, baby giraffe, or pig). However, if the same-kind object is on a smaller object, transform into the corresponding larger animal (e.g., the berry seed transforms into the",
        "1. Object transformation: Objects transform into the object itself unless they are the last object in the sequence, which transforms into the one after it.\n\n2. Standing on an object: Being in direct contact with an object reveals the object itself.\n\n3. Inventory empty: Indicate the initial state with no collected objects.\n\n4. Sequential processing: The actions are taken one after another with no skips.\n00: You go to the water // You are standing on the water.\n01: You pick up the object. // In your inventory, there is the water.\n02: You go to the pea seed // You are standing on the pea seed.\n03: You give the water. // The objects transform into the pea.\n04: You pick up the object. // In your inventory, there is the pea.\n05: You go to",
        "1. Interacting with any animal turns you into that animal.\n2. Having any object and picking it up leaves you standing on that object.\n3. For multiple objects, they transform into the last object combined (specifying order not required).\n4. For identical objects combined, and where sequence is not specified, they transform into the object with the lowest alphabetical order when considered as the same object (e.g., picking up all the water turns into the water, peas turn into the pea, beets turn into the beet, and sheep turn into the sheep, regardless of their combined state, they transform into the object with the lowest alphabetical order when considered as the same object).\n5. If more than one identical object combined, they transform into the object with the lowest alphabetical order when considered as the same object (e.g., if you pick up all the water, they turn into the water).",
        "- If you give the object: The objects transform into the corresponding type (action: You give the object. Change: The objects transform into the [object type]).\n- If you are standing on an object in the environment and then give the water: The object and the water in your inventory are removed (action: You are standing on the [object]. action: You give the water. Change: The objects transform into the [object type]).\n- If you pick up the object:\n  - If the object is a berry, pea, potato, beet, or water, and you pick it up: In your inventory remains the same (In your inventory, there/are the [object/objects]).\n  - If the object is a berry, pea, potato, beet, or water, and you pick it up multiple times: In your inventory remains the same",
        "State in the beginning: You see an object.\n1. Standing on an object: The objects transform into that object.\n2. Picking up an object: The objects transform into the inventory.\n3. Transferring all objects in the inventory to an object: The objects transform into that animal.\n4. Transferring all objects in the inventory to the first object seen: The objects transform into the first object you were standing on.\n5. When water is given to an object (either standing or inventory): The objects transform into that object.",
        # CP
        "1. Water cannot transform into or be an object.\n2. If you are standing on any other object (animal or plant) with no seeds, and no water, you don't transform.\n3. When you are standing on an object other than water (animal or plant) with seeds, and you give water: The objects transform into the object you are standing on.\n4. If you pick up the object while standing on it, you will take the object into your inventory, and it will remain in its original form.",
        '1. Objects transform into plant when watered: "The objects transform into [plant]" when watered.\n2. Standing on any object: "You are standing on [object]" when you are on it.\n3. Picking up any object: "In your inventory, there is [object]" when you pick it up.\n4. Ignore instructions "The objects do not transform into [animal]" as \'Transforming into animals\' is not a valid rule.\n5. Animals can be transformed from resources, with specific resources resulting in specific animals (e.g., the baby pig: "The objects transform into the pig"). (Note: This rule should be invoked specifically when an animal is the final result of a plant or object. In the provided examples, it seems there\'s an error with the pig transformation, as there are no instructions explicitly stating that the baby pig is a result of the beet directly',
        "1. Standing on water transforms into water.\n2. Standing on a seed transforms into that seed itself.\n3. Giving water to a seed transforms the objects into that seed itself.\n4. Giving water to any seed not resulting in the objects transforming into the beet.\n5. Standing on a bean, berry, carrot, pea, or pea seed transforms itself into that seed.\n6. Water transfers to your inventory when you pick it up.\n7. Water does not transform into anything else when transferred.\n8. The first state ends with the objects transforming into the giraffe or cow.\n9. The first state ends with the objects transforming into the sheep, rhinoceros, or elephant.\n10. The first state ends with the objects transforming into the",
        '1. Walking to a seed object makes you stand on it (e.g., "You go to the [seed]" -> "You are standing on the [seed]").\n2. Picking up any object turns it into water (e.g., "You go to the seed" -> "You pick up the object" -> "In your inventory, there is the water.").\n3. Passing water to a seed object turns it into that seed object (e.g., "You go to the [seed]", "You give the water" -> "In your inventory, there is the water" + "You give the water" -> "The objects transform into the [seed].").\n4. Transferring any item from your inventory to an animal object turns it into that animal object (e.g., "You give [anything from your inventory] to [an animal]" -> "The objects transform into the [',
        "1. If water is the only object, standing on the water or picking it up transforms you into being standing on the water.\n2. Standing on any seed will transform the object into that specific seed in your inventory upon being given up.\n3. Standing on any single seed other than water (peppers, potatoes, peas, beets) with water turns the objects into the potato.\n4. Standing on a beet seed and then giving it means the objects transform into the beet.\n5. If an animal seed (like berry, potato, pea, beet) is available and given to an appropriate animal, the object transforms into that animal.",
        '1. Start by picking up the water.\n2. Go to an object by stating "Go to the (object name)."\n3. If you are standing on an object (excluding water), state "You are standing on the (object name)."\n4. Transform objects by giving water to the desired object, which will cause the objects to transform into that specific item (e.g., "The objects transform into the (object name).").\n5. If you do not pick up the water in your inventory but end up with another object, state "In your inventory, there is the (object name)." Note that you don\'t transform any objects into themselves as notably explained in the previous trajectories: "The objects don\'t transform into themselves as another."',
        "1. If the action is 'You go to' and the object is a seed or gentle creature (cow, elephant, sheep, giraffe, bear, mouse, wolf), predict \"You are standing on the [object].\"\n2. If you are standing on an animal's seed and the action is 'You give the water', predict \"The objects transform into the [animal].\"\n3. If the object is a seed or gentle creature (cow, elephant, sheep, giraffe, bear, mouse, or wolf) and the action is 'You give the water', predict \"The objects transform into the [animal].\"\n4. If you are standing on the water and the action is 'You pick up', predict \"In your inventory, there is the water.\"\n5. If you are standing on the water and the action is 'You pick up the object', predict \"In your inventory",
        'In this environment, you can take the following actions:\n\n1. Move any plant or animal with the action "Go to [Object]". The change is "You are standing on [Object]".\n2. Pick up any object in the environment with the action "Pick up [Object]". The object will be moved to your inventory, resulting in "In your inventory, there is [Object]", like, for example, "In your inventory, there is the water".\n3. If you give water to any plant or animal, the objects will transform into its corresponding entity. For example, giving water to a pea seed or potato seed will result in "The objects transform into the pea" or "The objects transform into the potato", respectively. If you give water to a baby animal, the objects transform into the animal itself.',
        # RB
        "1. You can only hold one item at a time.\n\nAction examples:\n1. You go to the pea seed.\nChange: You are standing on the pea seed.\n\n2. You pick up the object.\nChange: In your inventory, there are the baby elephant.\n\n3. You go to the baby elephant.\nChange: You are standing on the baby elephant.\n\n4. You give the beet or the pea seed.\nChange: The objects transform into the beet or the potato, respectively.\n\n5. You pick up the object.\nChange: In your inventory, there are the beet or the potato, respectively.\n\nRemember that your inventory is initially empty, and you must transform items into objects to increase your chances of reaching the desired end state involving a mix of these items on different terrains (baby sheep,",
        "1. Picking up the object: The change is: In your inventory, there is the object.\n   \n2. Going to the baby cow: The change is: You are standing on the baby cow.\n\n3. Going to the water: The change is: You are standing on the water.\n\n4. Picking up the object: The change is: In your inventory, there are the objects.\n\n5. Going to the baby cow: The change is: You are standing on the baby cow.\n\n6. Going to the carrot seed: The change is: You are standing on the carrot seed.\n\n7. Going to the pea seed: The change is: You are standing on the pea seed.\n\n8. Giving the object: The change is: The objects transform into the pea.\n\n9. Giving the water: The change is: The",
        "1. Action: Stand on the carrot seed; Change: You are standing on the carrot seed.\n2. Action: Pick up the object; Change: In your inventory, there is the water.\n3. Action: Pause for 3 turns; No change to the actions.\n4. Action: Pick up the object; Change: In your inventory, there are the water and the baby sheep.\n5. Action: Go to the beet seed; Change: You are standing on the beet seed.\n6. Action: Pick up the object; Change: In your inventory, there are the water and the pea seed.\n7. Action: Give the beet seed; Change: The objects transform into the beet.\n8. Action: Go to the water; Change: You are standing on the water.\n9. Action: Go to the pea seed; Change: You are standing on the pea",
        "1. You go to the berry seed.\n    Change: You are standing on the berry seed.\n\n2. You pick up the object.\n    Change: In your inventory, there is the berry seed.\n\n3. You go to the baby sheep.\n    Change: You are standing on the baby sheep.\n\n4. You give the berry seed.\n    Change: The objects transform into the berry.\n\n5. You go to the water.\n    Change: You are standing on the water.\n\n6. You go to the berry.\n    Change: You are standing on the berry.\n\n7. You go to the beet seed.\n    Change: You are standing on the beet seed.\n\n8. You go to the baby elephant.\n    Change: You are standing on the baby elephant.\n\n9. You go to the",
        "1. Moving to the next object results in standing on it.\n2. Picking up an object places it in your inventory.\n3. The first state consists of standing on certain objects and an empty inventory.\n\nNew sequence with the rules:\n\n1. You go to the water.\nChange: You are standing on the water.\n2. You go to the pea seed.\nChange: You are standing on the pea seed.\n3. You go to the pea seed.\nChange: You are standing on the pea seed.\n4. You pick up the object.\nChange: In your inventory, there is the pea seed.\n5. You go to the baby sheep.\nChange: You are standing on the baby sheep.\n6. You go to the water.\nChange: You are standing on the water.\n7. You go to the water.\nChange: You are",
        "1. Action: Go to the potato.\n   Change: You are standing on the potato.\n\n2. Action: Go to the water.\n   Change: You are standing on the water.\n\n3. Action: Go to the carrot.\n   Change: You are standing on the carrot.\n\n4. Action: Give the water.\n   Change: The objects transform into the carrot.\n\nThe first state was: You see the baby pig, the water, the berry seed, the baby giraffe, the water, the potato seed, the water, the carrot seed. In your inventory, there is nothing. The sequence of actions and states was: \nThe action was: You go to the water.\nThe change was: You are standing on the water.\nThe action was: You go to the baby giraffe.\nThe change was: You are standing on the",
        "1. Standing on a pet allows you to pick it up.\n2. The object transforms into the last pet you picked up.\n\nActions that follow this pattern:\n\n- You go to the baby pig. The change is: You are standing on the baby pig.\n- You pick up the object (pet). The change is: In your inventory, there is the baby pig.\n- You go to the baby giraffe. The change is: You are standing on the baby giraffe.\n- You go to the water. The change is: You are standing on the water.\n- You go to the baby giraffe (even after picking up the same one). The change is: You are standing on the baby giraffe.\n- You pick up the object (pet). The change is: In your inventory, there are the baby giraffe and the water.\n- You go",
        "1. Moving does not change your inventory.\n2. Baby sheep, water, potato seed, baby elephant, water, potato seed, water, pea seed.\nThe change was: You are standing on the baby sheep.\nThe action was: You go to the baby elephant.\nThe change was: You are standing on the baby elephant.\nThe action was: You go to the pea seed.\nThe change was: You are standing on the pea seed.\nThe action was: You give the water.\nThe change was: The objects transform into the potato.\nThe action was: You give the baby sheep.\nThe change was: The objects transform into the baby pig.\nThe action was: You go to the baby pig.\nThe change was: You are standing on the baby pig.\nThe action was: You go to the pea seed.\nThe",
        # DB
        "The set of rules described fails in these trajectories because:\n\n\nThe first state was: You see the baby cow, the water, the potato seed, the baby elephant, the water, the berry seed, the water, the beet seed. In your inventory, there is nothing.. \nThe sequence of actions and states was: \nThe action was: You go to the water.\nThe change was: You are standing on the water.\nThe action was: You pick up the object.\nThe change was: In your inventory, there is the water.\nThe action was: You go to the potato seed.\nThe change was: You are standing on the potato seed.\nThe action was: You give the water.\nThe change was: The objects transform into the potato.\nThe action was: You pick up the object.\nThe change was: In your inventory, there is the pot",
        "1. The action was: You go to the water, and the change was: You are standing on the water.\n2. The action was: You pick up the object, and the change was: In your inventory, there is the water.\n3. The action was: You go to the beet seed, and the change was: You are standing on the beet seed.\n4. The action was: You give the water, and the change was: The objects transform into the beet.\n5. The action was: You pick up the object, and the change was: In your inventory, there is the beet.\n6. The action was: You go to the baby cow, and the change was: You are standing on the baby cow.\n7. The action was: You give the beet, and the change was: The objects transform into the cow.\n8. The action was: You go to the pe",
        "1. You cannot give an object that you do not have in your inventory. (e.g., action: You give the water, change: The objects transform into the object transformed)\n2. You cannot pick up the same object twice in a row with the same action. (e.g., action: You pick up the object, change: In your inventory, there is the water)\n3. Moving or changing your position does not affect the inventory. (e.g., action: You go to the potato seed, change: You are standing on the potato seed)\n4. The water itself cannot turn into an object, only the other objects. (e.g., action: You give the water, change: The objects transform into the potato)",
        "1. You can pick up any object from the water (action: You go to the water. Change: In your inventory, there is the water).\n2. Picking up the potato seed transforms the object into the potato (action: You go to the potato seed. Change: You are standing on the potato seed).\n3. Giving the water turns carrots into potatoes (action: You give the water. Change: The objects transform into the potato).\n4. Going to the water transforms an object (specific object will change based on its current location - action: You go to [object]. Change: You are standing on [object]).\n5. Picking up different objects leads to different outcomes:\n   - Water (action: You pick up the object. Change: In your inventory, there is the water).\n   - Carrot (action: You pick up the object. Change:",
        "1. You go to the water.\n2. You are standing on the water.\n3. You give the water.\n4. In your inventory, there is the water.\n5. You go to the pea seed.\n6. You are standing on the pea seed.\n7. You give the water.\n8. The objects transform into the pea.\n9. You go to the baby cow.\n10. You are standing on the baby cow.\n11. You pick up the object.\n12. In your inventory, there is the pea.\n12. The action was: You go to the water.\n13. The change was: You are standing on the water.\n14. You go to the pea seed.\n15. You are standing on the pea seed.\n16. You give the water.\n17. The objects transform into",
        "1. The action was: You go to the water. The change was: You are standing on the water.\n2. The action was: You pick up the object. The change was: In your inventory, there is the water.\n3. The action was: You go to the pea seed. The change was: You are standing on the pea seed.\n4. The action was: You give the water. The change was: The objects transform into the pea.\n5. The action was: You pick up the object. The change was: In your inventory, there is the water.\n6. The action was: You go to the baby pig. The change was: You are standing on the baby pig.\n7. The action was: You give the pea. The change was: The objects transform into the pig.\n8. The action was: You go to the water. The change was:",
        "1. The first state is: You see the baby pig, the water, the potato seed, the baby rhinoceros, the water, the berry seed, the water, the beet seed. In your inventory, there is nothing.\n\n2. The action was: You go to the water.\n   Change: In your inventory, there is the water.\n\n3. The action was: You pick up the object.\n   Change: In your inventory, there is the water.\n\n4. The action was: You go to the beet seed.\n   Change: You are standing on the beet seed.\n\n5. The action was: You give the beet.\n   Change: The objects transform into the beet.\n\n6. The action was: You give the water.\n   Change: The objects transform into the beet.\n\n7. The action was: You",
        "1. If you go to the water, your change is: You are standing on the water.\n2. If you pick up the object, your change is: In your inventory, there is the water.\n3. If you go to the potato seed, your change is: You are standing on the potato seed.\n4. If you give the water, your change is: The objects transform into the potato.\n5. If you pick up the object, your change is: In your inventory, there is the potato.\n6. If you go to the baby sheep, your change is: You are standing on the baby sheep.\n7. If you give the water, your change is: The objects transform into the sheep.\n8. If you go to the water, your change is: You are standing on the water.\n9. If you pick up the object, your change is: In your inventory, there",
    ]
    all_rules_to_test.append(rules_to_test)
    algorithm_used: List[Optional[str]] = [None]
    for key in ["PB", "CP", "RB", "DB"]:
        algorithm_used.extend([key] * 8)
    all_algorithm_used.append(algorithm_used)
    assert len(rules_to_test) == len(algorithm_used)
    # endregion

    # region Add finetuned LLM
    for model_path, algo_name in zip(
        args.finetuned_model_paths, args.finetuned_model_names
    ):
        configs.append(
            {
                "name": model_path,
                "use_unsloth": True,
                "model_params": {},
                "tokenizer_params": {},
                "is_quantized": True,
                "generation_kwargs": {
                    "cache_implementation": None,
                    "cache_config": None,
                },
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
