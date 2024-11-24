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
    # Rule PB
    "a. Interacting with a beet and water leads to the goal transforming into the beet.\nb. Consistently interacting with water and any object results in the object transforming into itself.\nc. You go to the water. o: You are standing on the water.\nd. You give the object. o: You are holding the water.\ne. You go to the baby pig. o: You are standing on the baby pig.\nf. You give all the objects you hold. o: The beet, the carrot and the baby pig transform into the pig.\ng. You go to the potato seed.\nh. You give the water. o: The water and the potato seed transform into the potato.\ni. You go to the beet seed.\nj. You give the water. o: The water and the beet seed do not transform into anything but remain",
    "a. Interacting with a beet and water leads to the goal transforming into the beet.\nb. Consistently interacting with water and any object results in the object transforming into itself.\nc. You go to the water. o: You are standing on the water.\nd. You give the object. o: You are holding the water.\ne. You go to the baby pig. o: You are standing on the baby pig.\nf. You give all the objects you hold. o: The pea, the beet and the baby pig transform into the pig.\ng. You go to the potato seed.\nh. You give the water. o: The water and the potato seed transform into the potato.\ni. You go to the beet seed.\nj. You give the water. o: The water and the beet seed do not transform into anything but remain",
    "a. Interacting with a carrot and water leads to the goal transforming into the carrot.\nb. Consistently interacting with water and an object results in the object transforming into itself.\nc. You go to the water. o: You are standing on the water.\nd. You give the object. o: You are holding the water.\ne. You go to the baby cow. o: You are standing on the baby cow.\nf. You give all the objects you hold. o: The carrot, the carrot and the baby cow transform into the cow.\ng. You go to the potato seed. o: You are standing on the potato seed.\nh. You give the water. o: The water and the potato seed transform into the potato.\ni. You go to the pea seed.\nj. You give the water. o: The water and the pea seed",
    # Rule RB
    "The rules as derived from the sequence of actions and observations are:\n\n1. Action on an object gives you whatever is currently in the potato. e.g.: A: You go to the potato seed. o: You are standing on the potato seed. A: You pick up the object. o: You are holding the potato and the potato.\n2. If you give your entire inventory to the stated location, the last object transforms into whatever you give it. e.g.: A: You give the potato and the water. o: The water and the potato transform into the potato.\n3. If you go to the same action twice: A: You go to the potato seed. o: Nothing has changed. A: You go to the potato seed. o: Nothing has changed.\n4. Inventory (0/2): empty.\n\nGiven the new set of rules and the most",
    "1. Action on an object gives you whatever is currently in the potato. e.g.: A: You go to the potato seed. o: You are standing on the potato seed. A: You pick up the object. o: You are holding the potato and the potato.\n2. If you give your entire inventory to the stated location, the last object transforms into whatever you give it. e.g.: A: You give the potato and the water. o: The water and the potato transform into the potato.\n3. If you go to the same action twice: A: You go to the potato seed. o: Nothing has changed. A: You go to the potato seed. o: Nothing has changed.\n4. Inventory (0/2): empty.",
    "In the current space: You see: baby pig, water, pea seed, baby elephant, water, berry seed, water, carrot seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations are:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the pea seed. o: You are standing on the pea seed. a: You go to the pea seed. o: Nothing has changed. a: You give the water. o: The water and the pea seed transform into the pea. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You go to the baby pig. o: You are standing on the baby pig. a: You go to",
    # Rule CP
    '1. Stand on Object: When you "go to" any object, you should be standing on it (Example: "go to the potato" → "You are standing on the potato").\n2. Harvest Transformation: When you "pick up" the water and another object, they will merge into a new object (Example: "pick up the water and give it to the beet seed" → "The water and the beet seed transform into the beet").\n3. Transformation of Life: When you "feed" any object to its respective entity (Example: "give the potato to the baby cow" → "The potato and the baby cow transform into the cow").',
    "Inventory (0/2): empty. \nThe sequence of actions and observations are:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The water and the potato seed transform into the potato. a: You pick up the object. o: You are holding the potato. a: You go to the baby pig. o: You are standing on the baby pig. a: You give the potato. o: The potato and the baby pig transform into the pig. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the carrot seed",
    '1. Be Ready Position: When you "go to" any object, you should be standing on it (e.g., "go to the potato" → "You are standing on the potato").\n2. Harvest Environment: When you "pick up" the water and another object, they will merge into a new object (e.g., "pick up the water and give it to the beet seed" → "The water and the beet seed transform into the beet").\n3. Foster Life: When you "feed" any food to an animal or plant, they transform into the adult form of their species (e.g., "give the potato to the baby ostrich" → "The potato and the baby ostrich transform into the ostrich").',
    # Rule DB
    '1. You are standing on an associated object: "You go to the associated object: berry seed. o: You are standing on the berry seed. a: You give the water. o: The water and the berry seed transform into the berry."\n2. All objects in a set are consumed together to transform into the associated object: "Nothing has changed. You go to the water. o: You are standing on the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The cow, the pea, and the baby elephant transform into the elephant."\n3. Water alone does not transform or reset into an associated object.\n4. No change occurs if an object is not associated with any other objects: "Nothing has changed. You go to the beet seed. o: You are standing on the beet seed',
    '1. You are standing on the associated object: "You go to the associated object: pea seed. o: You are standing on the pea seed. a: You give the water. o: The water and the pea seed transform into the pea."\n2. All objects in a set are consumed together to transform into the associated object: "Nothing has changed. You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The cow, the pea, and the baby elephant transform into the elephant."\n3. Water alone does not transform or reset into an associated object: "Nothing has changed. You go to the berry seed. o: You are standing on the berry seed. a: You go to the water. o: You are standing on the water. a: You give the berry. o: The berry and the water',
    '1. You are standing on an associated object: "You go to the associated object: berry seed. o: You are standing on the berry seed. a: You give the water. o: The water and the berry seed transform into the berry."\n2. All objects in a set are consumed together to transform into the associated object: "Nothing has changed. You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The cow, the pea, and the baby elephant transform into the elephant."\n3. Water alone does not transform or reset into an associated object.\n4. No change occurs if an object is not associated with any other objects: "Nothing has changed. You go to the beet seed. o: You are standing on the beet seed. a: You go to the water. o: You are standing on the water',
    # rule LL
    "a. Ignoring objects or certain objects removes them from the inventory.\na: You focus on the baby rhinoceros. o: You are standing on the baby rhinoceros.\na: You collect the potato. o: The water and the potato seed transform into the potato.\na: Ignoring objects or certain objects removes them from the inventory.\na: You focus on the pea. o: Nothing has changed.\na: You focus on the water. o: Nothing has changed.\na: You give the water and the pea seed transform into the pea.\na: You go to the potato seed. o: You are standing on the potato seed.\na: You give the water. o: Nothing has changed.\na: You give the water and the potato seed transform into the potato.\n\nThus, the completed set of rules is:\n1. Pick",
    '1. Picking up an object has no effect unless it is the last action before an object transformation.\n2. Water transforms into the object you pick it up with or into the object you last touched it with.\n3. Moving to a location with an object does nothing unless you pick it up.\n4. An empty inventory does nothing.\n5. Actions with no state change are represented as "Nothing has changed."\n6. Actions result in the same thing if the object is a baby, regardless of the sequence.',
    '1. Picking up an object has no effect unless it is the last action before an object transformation.\n2. Water turns into the object you pick it up with or the object you last touched it with.\n3. Moving to a location with an object does nothing unless you pick it up.\n4. An empty inventory does nothing.\n5. Actions with no state change are represented as "Nothing has changed."\n6. Actions result in the same thing if the object is a baby, regardless of the sequence.',
    # Rule ALP
    '1. If no state change, say "Nothing has changed."\n2. If you stand on an object, say "You are standing on the [object]."\n3. If you end up on a new object, say "You are standing on the [new object]."\n4. Continue if no active actions or progress is made, repeat "Nothing has changed."\n5. If an object transforms, say "The [object] and the water transform into the [new object]."',
    '1. If no state change, say "Nothing has changed."\n2. If you stand on an object, say "You are standing on the [object]."\n3. If you would have ended up on a new object, say "You are standing on the [object]."\n4. Continue if no active actions or progress is made, repeat "Nothing has changed."\n5. If an object transforms, say "The [object] and the water transform into the [new object]."',
    '1. If no state change, say "Nothing has changed."\n2. If you stand on an object, say "You are standing on the [object]."\n3. If you would have ended up on a new object, say "You are standing on the [object]."\n4. Continue if no active actions or progress is made, repeat "Nothing has changed."\n5. If an object transforms when you are on it, say "The [object] and the water transform into the [new object]."',
    # Rule ALPLL
    '1. Any action towards water results in "Nothing happens."\n2. Any interaction with an object results in holding the object and the object itself. Example: You go to the pea. o: You are standing on the pea. a: You pick up the object. o: You are holding the water and the water.\n3. Any object, water, and another object transform into that object itself. Example: You go to the pea seed. o: You are standing on the pea seed. a: You give the water. o: The water and the pea seed transform into the pea.\n4. Any two objects result in holding both objects.\n5. Any water and the rhinoceros transform into the water.\n6. Any water and the pea transform into the pea.\n7. Any pea and the baby cow or pig transform into the pea.\n8. Any',
    '1. Any action towards water results in "Nothing happens". \n2. Any interaction makes you hold the object and the object itself. Example: You go to the pea. o: You are standing on the pea. a: You pick up the object. o: You are holding the water and the water. \n3. Any object, water, and another object transform into that object itself. Example: You go to the pea seed. o: You are standing on the pea seed. a: You give the water. o: The water and the pea seed transform into the pea.\n4. Any two objects result in holding both objects.\n5. Any water and the rhinoceros transform into the water.\n6. Any water and the pea transform into the pea.\n7. Any pea and the baby cow or pig transform into the pea with the cow or pig.\n8. Any',
    '1. Any action towards water results in "Nothing happens".\n2. Any pickup results in holding the object, and the object itself. Example: You go to the pea. o: You are standing on the pea. a: You pick up the pea. o: You are holding the pea and the water. \n3. Any object, water, and another object transforms into that object. Example: You go to the pea seed. o: You are standing on the pea seed. a: You give the water. o: The water and the pea seed transform into the pea.  \n4. Any action, object, and another object transforms into that object itself. \n5. Any object and the baby rhinoceros transform into the object itself. \n6. Any two objects result in holding both objects. If cow is picked, and no all objects are given, the cow and the object that',
]
algorithm_used = [
    None,
    "PB",
    "PB",
    "PB",
    "RB",
    "RB",
    "RB",
    "CP",
    "CP",
    "CP",
    "DB",
    "DB",
    "DB",
    "LL",
    "LL",
    "LL",
    "ALP",
    "ALP",
    "ALP",
    "ALPLL",
    "ALPLL",
    "ALPLL",
]
assert len(rules_to_test) == len(algorithm_used)
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
