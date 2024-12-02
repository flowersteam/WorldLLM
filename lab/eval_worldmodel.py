import json
import os
import sys
from typing import List, Optional

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
    # PB
    "Objects transform into themselves.\n1. Water transforms into itself.\n2. Placing an object or its seed on a large herbivore results in the large herbivore transforming into its corresponding animal.\nA: You go to the pea. o: You are standing on the pea. a: You give the water. o: The water and the pea seed transform into the pea. a: You pick up the object. o: You are holding the pea. a: You go to the baby pig. o: You are standing on the baby pig. a: You give the pea. o: The pea and the baby pig transform into the pig. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the baby elephant. o: You are standing",
    "The simple rules for the given scenarios are:\n\n1. The water transmutes any plant seed into the same seed.\n2. Water and a plant seed swap identities if the target seed is plant-based (potato, carrot, beet, pea).\n3. Whe disappears after transformation or swap.\n4. The water and a plant seed transform into the respective plant.\n5. The water and a pea seed transform into the pea.\n6. The water and a carrot seed transform into the carrot.\n7. The water and a beet seed transform into the beet.\n8. You give all the objects you have to an animal, you transform all those objects into that animal (sheep, cow, elephant, giraffe, rhinoceros).\n\nAll these rules are simplified to consist of only one-action-one-result sequences to allow for straightforward prediction of the outcome based",
    "1. Animals and seeds transform into their respective seed versions.\n2. If you pick up an object or seed, both transform into the same object or seed.\n3. The water and the corresponding seed or animal transform into the seed or animal.\n4. The water and the beet seed transform into the beet.\n5. The water and the pea seed transform into the pea.\n6. The water and the potato seed transform into the potato.\n7. When you go to the water, o: You are standing on the water.\n8. When you go to a seed, o: You are standing on that seed.\n9. When you have a creature (animal or seed), a: You give the object or seed. o: The water, object, animal, or seed all transform into",
    'For this complex transformation scenario, I\'ll outline a set of rules that seem clearer and more manageable, taking inspiration from the previous ones:\n\n1. Any plant or seed transforms into the specific type of the plant or the seed.\n2. Water directly transforms or has an impact on objects, transforming them into the object or transforming the object into water.\n3. Large animals transform using only water.\n\nGiven these revised rules:\n\n- When you go to a seed, water, or plant that\'s already there, you pick up "You go to ..." o: You are standing on the seed. a: You give the water. o: The water and the seed transform into the seed.\n- If you pick up an object, do nothing, and it\'s still water, you hold "The object remains the same." o: You are holding the water.\n- When you perform actions specifically directed',
    '1. Animals transform into adult versions based on their social ranking, e.g., "You give water to the baby <Object>" becomes "The water and the <Object> transform into the adult <Object>."\n2. Plants associated with objects become the plant itself when given to it, e.g., "You give water to the carrot seed." turns into "The water and the carrot seed transform into the carrot."\n3. If an object is picked up and held with another object, the original object remains tied up with it, e.g., "You go to the water." becomes "You are standing on the water." and "You give water" becomes "The water and the object continue to exist as tied up.".',
    '1. Water interaction does not alter the state of objects.\n2. Any interaction with a pea results in a pea (action "o: You go to the pea seed" followed by "You are standing on the pea seed. action: You give the water. o: The water and the pea seed transform into the pea.").\n3. Placing both water and a carrot or pea seed on a base object transforms them into that object (e.g., "o: You go to the water. action: You pick up the object. o: You are holding the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You give the water. o: The water and the carrot seed transform into the carrot."; similarly with pea).\n4. Placing both water and a pea or carrot on a base object that can transform',
    '1. Introduction: Navigate to or stand on a target object/seed for transformation. For example: "You go to the beet seed. o: You are standing on the beet seed."\n2. Absence of action or mere observation results in the object/seed and water mutation. For example: "You give the water. o: The water and the object transform into the object."\n3. Holding an object alone will combine the present objects. For example: "You go to the baby elephant. a: You pick up the pea, the beet, and the water. o: The pea, the beet, the water, and the baby elephant transform into the elephant."',
    "1. Animals transform into their adult form: baby sheep into the sheep, baby cow into the cow, baby giraffe into the giraffe, baby elephant into the elephant, baby pig into the pig, baby rhinoceros into the rhinoceros.\n2. A pea seed and water can transform into the pea.\n3. A carrot seed and water can transform into the carrot.\n4. A potato seed and water can transform into the potato.\n5. A beet seed and water can transform into the beet.\n6. Water transforms into the corresponding object/plant: The water and the corresponding object/plant (seed) transform into the corresponding object/plant after interacting with the seed. You are no longer limited to these interactions, but whenever the initial rules are not clear and too complex, you can directly go to",
    # RB
    "In the current space: You see: baby pig, water, carrot seed, baby giraffe, water, potato seed, water, pea seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the baby pig. o: You are standing on the baby pig. a: You pick up the object. o: You are holding the water. a: You go to the water. o: You are standing on the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You go to the potato seed. o: You are standing on the potato seed. a: You go to the pea seed. o: You are standing on the pea seed. a: You give the object. o: The water and the pea seed transform into the pea.",
    "Here are the updated rules that should be used successfully on all the mentioned trajectories:\nThe sequence of actions and observations is:  a: You go to the beet. o: You are standing on the beet. a: You go to the beet. o: You are standing on the beet. a: You go to the peak. o: You are standing on the peak. a: You go to the tree. o: You are standing on the tree. a: You go to the water. o: You are standing on the water. a: You go to the cow. o: You are standing on the cow. a: You go to the tree. o: You are standing on the tree. a: You pick up the object. o: You are holding the cow and the tree (inventory capacity). a: You give the cow. o: The cow and the tree (inventory capacity) transform into the cow.\nThe sequence",
    "The balls:\n\nIn the current space: baby sheep, water, berry seed, baby rhinoceros, water, berry seed, water, potato seed\nYou are on: nothing\nInventory (0/2): empty. \n\nThe sequence of actions and observations is:  a: You take the water. o: You are standing on the water. a: You go to the berry seed. o: You are standing on the berry seed. a: You go to the water. o: You are standing on the water. a: You go to the water. o: You are standing on the water. a: You go to the berry seed. o: You are standing on the berry seed. a: You give the water. o: The water and the berry seed turn into the berry. a: You go to the baby rhinoceros. o: You are standing on the baby rhin",
    "1. To find derived fruit from a potato, you go to the potato as you are standing on the potato.\n\nIn the current space: You see: baby sheep, water, berry seed, baby rhinoceros, water, carrot seed, water, berry seed\nYou are on: carrot seed\nInventory (0/2): empty. \nThe sequence of actions and observations is:\n a: You go to the water. o: You are standing on the water.\n a: You go to the water. o: You are standing on the water.\n a: You pick up the object. o: You are holding the berry.\n a: You give the water. o: The water and the berry seed transform into the berry.\n a: You go to the water. o: You are standing on the water.\n a: You pick up the water. o: You are holding",
    "The rules you were given were:\n\nYou also know that the previous set of rules failed the most on those trajectories:\n\nYou are on: nothing\nInventory (0/2): empty. \n\nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You go to the pea. o: You are standing on the pea. a: You go to the water. o: You are standing on the water. a: You give the object. o: The pea and the water transform into the pea (now your Inventory has the pea). a: You say turn left. o: You are standing on the detention cell. a: You say turn right. o: You are standing on the pea. a: You say turn right. o: You are standing on the pea. a: You say turn left. o: You",
    "In the current world you see: baby sheep, water, pea seed, baby giraffe, water, beet seed, potato seed, water, pea seed. You are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations are:  a: You go to the pea seed. o: You are standing on the pea seed. a: You go to the baby giraffe. o: You are standing on the baby giraffe. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the baby giraffe. a: You go to the pea seed. o: You are standing on the pea seed. a: You give the water. o: The water and the pea seed transform into the pea. a: You give the pea seed. o: The pea",
    "I apologize, but I cannot generate the question. \n\nIn the current space: You see: baby sheep, water, berry seed, baby elephant, water, berry seed, water, berry seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You go to the berry seed. o: You are standing on the berry seed. a: You go to the water. o: You are standing on the water. a: You give the object. o: The water and the berry seed transform into the berry. a: You go to the berry seed. o: You are standing on the berry seed. a: You go to the water. o: You are standing on the water. a: You go to the berry seed. o",
    "In the current space: You see: baby pig, water, pea seed, baby giraffe, water, pea seed, water, carrot seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You give the object. o: You are holding the carrot seed and the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You pick up the object. o: You are holding the water and the carrot seed. a: You give the water. o: The water and the carrot seed transform into the water. a: You give the pea seed. o: The pea seed and the water transform into the pea. a: You give the baby pig. o: The pea and",
    # CP
    '1. You can move to an object or seed: "You go to [object or seed]" leads to "You are standing on [object or seed]".\n2. If you want to pick up an object or seed: "You pick up the [object or seed]" leads to "You are holding the [object or seed]".\n3. When given water to a pea, the dialogue would be, "You give the water to the pea", which results in "The water and the pea seed transform into the pea".\n4. Water will transform into the object or seed if:\n   - Given to a potato seed: "The water and the potato seed transform into the potato".\n   - Given to a baby cow: "The water and the cow transform into the cow".\n   - Given to a baby giraffe: "The water and the giraffe transform into the giraffe".\n  ',
    '1. Empty inventory (0/2).\n2. Find the nearest water source, always ends with "You are standing on the water."\n3. Collect water: "You pick up the water." leads to "You are holding the water."\n4. After water, move to the specified object: "You go to [object]." becomes "You are standing on the [object]."\n5. Transform all non-water items with water into the object itself: "You give the water [object]." turns into "The water and the [object] transform into the [object]." if the object is not water itself',
    '1. Always start at a water source.\n2. Use the action "go to the water" to move towards an unoccupied water source.\n3. Once at the water source, perform the sequence "you are standing on the water," "you pick up the object," and "you are holding the object you picked."\n4. When on an object (be it a seed or animal), perform the action "you give the object" to apply water to it.\n5. If you encounter a seed, follow the sequence "the water and the _ seed transform into the _." (e.g., the water and the carrot seed transform into the carrot.)\n6. If you encounter an animal or an object that transforms into an animal (like a baby), perform the sequence "The water and the _ and the baby transform into the _." (e.g., The water and the carrot and the baby sheep transform into the sheep.)',
    "1. Water is used in its place for one at a time.\n2. If standing on a plant seed and has water, the water and seed will transform into the corresponding plant.\n\nMatching the rules to the scenarios provided yields the following observations:\n\nIn the current space: You see: baby sheep, water, carrot seed, baby giraffe, water, carrot seed, water, pea seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations are:\na: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You give the water. o: The water and the carrot seed transform into the carrot.\na: You go to the pe",
    "1. Trees remain unchanged.\na. Go to a tree.\n   o: The tree remains unchanged.\n\n2. All life forms (animals and plants) transform based on the object:\n   a. Go to water.\n   b. Pick up the object.\n   o: The object and the water transform into the same object.\n   a. Go to the pea seed.\n   b. Give the water to the pea seed.\n   o: The water & the pea seed transform into the pea.\n   a. Go to the berry seed.\n   b. Give the water to the berry seed.\n   o: The water & the berry seed transform into the berry.\n   a. Go to the baby sheep.\n   b. Give the water to the baby sheep.\n   o: The water & the baby sheep transform into the sheep.\n   a",
    "1. Interaction with water will transform objects of similar kinds.\n2. Non-transformable items (animals or seeds) remain unchanged when only one is involved.\n3. Interactions with non-water do not result in any transformation.\n4. Objects that are trades of 1:1 will transform into the type of the only object present.\n5. When two objects of the same kind are in water, they transform together into another kind.\n\nThe puzzle begins in an empty environment. Then out of nowhere, I encountered the following:\n\n- Round 1: a little baby pig?o: I am standing on the little baby pig. You have a baby pig. A: Give the water. o: The little baby pig and the water transform into the pig. A: Pick up the object. o: You are holding the pig.\n- Round 2: a little",
    '1. Obtain Water: Locate and pick up water, transforming into water.\n   Actions: You go to the water → You are standing on the water.\n\n2. Stand On Object: Approach and stand on any object, transforming into that object.\n   Actions: You stand on [object] → "[You are standing on the] [object]."\n\n3. Pick up Object: Pick up any object, transforming into that object.\n   Actions: You pick up the object → "[You are holding the] [object]."\n\n4. Interaction Transformation: If water is given to any non-water object (other than the pea, beet, or pea seed), they transform into that object. \n   Actions: You give the water to the [object] → "The water and the [object] transform into the [object]."',
    "In the current space: You see: baby pig, water, pea seed, baby elephant, water, beet seed, water, berry seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the beet seed. o: You are standing on the beet seed. a: You give the water. o: The water and the beet seed transform into the beet. a: You pick up the object. o: You are holding the beet. a: You go to the baby pig. o: You are standing on the baby pig. a: You give the beet. o: The beet and the baby pig transform into the pig. a",
    # DB
    "1. Go to a seed/object/seed. o: You are standing on the seed/object/seed.\n2. Pick up the object, you are standing on that object/seed. a: You pick up the object. o: You are holding the object/seed.\n3. Go to the water, you are standing on it. a: You go to the water. o: You are standing on the water.\n4. Give the water to an object/seed, the water and the object/seed transform into the object/seed. a: You give the water. o: The water and the object/seed transform into the object/seed.\n5. You can transform an additional object/seed after meeting the water. a: You go to a seed/object/seed. o: You are standing on the seed/object/seed.",
    "in the current space: You see: baby sheep, beet seed, water, beet seed, baby giraffe, water, carrot seed\nYou are on: everywhere\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the beet seed. o: You are standing on the beet seed. a: You give the water. o: The water and the beet seed transform into the beet. a: You pick up the object. o: You are holding the beet. a: You go to the baby giraffe. o: You are standing on the baby giraffe. a: You give the beet. o: The beet and the baby giraffe transform into the giraffe.",
    "Go to the water, pick up the object.\nYou also know that the previous set of rules failed on those trajectories:\n\nIn the current space: You see: baby pig, water, berry seed, baby elephant, water, beet seed, water, beet seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the beet seed. o: You are standing on the beet seed. a: You give the water. o: The water and the beet seed transform into the beet. a: You pick up the object. o: You are holding the beet. a: You go to the baby pig. o: You are standing on the baby",
    "a: Go to a given object. In the current space: You see: baby sheep, water, carrot seed, baby elephant, water, berry seed, water, potato seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You give the water. o: The water and the carrot seed transform into the carrot. a: You give all the objects you hold. o: The carrot, the baby sheep and the baby elephant transform into the respective creatures. a: Go to a given object. o: The remaining objects are standing on the potato seed and the berry",
    "1. Picking up the object and the water transforms them both into new objects.\n2. Consuming one object on water results in the object and the water transforming into a larger object, oft0: In the current space: You see: baby cow, beef, berry seed, baby giraffe, bottle of water, baby potato, baby rhinoceros, pea seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the beef. o: You are standing on the beef. a: You give the water. o: The water and the beef transform into the beef. a: You pick up the object. o: You are holding the",
    "a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The water and the potato seed transform into the potato. a: You pick up the object. o: You are holding the potato. a: You go to the beet seed. o: You are standing on the beet seed. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the potato and the beet. a: You go to the baby cow. o: You are standing on the baby cow. a: You give the potato. o: The potato and the baby cow transform into the cow. a: You go to the car",
    "Identify the object/plant/seed/beverage: When you stand on an object/plant/seed/beverage, move to it. o: You are standing on the [object/plant/seed/beverage]. \nGo to the object/plant/seed/beverage. o: You are standing on the [object/plant/seed/beaver]. \nGive the water: Give water to whatever you are standing on. o: The water and the [object/plant/seed/beaver] transform into the [beaver]. \nPick up the object: When you have water or are standing on an object/plant/seed/beaver, pick it up. o: You are holding the [object/plant/seed/beaver]. \nGo to the object/plant/seed. o: You are standing on the [object/group/plant/beaver].",
    "Inventory (0/2): empty pairs ||\nIn the current space: baby pig, water, potato seed, baby elephant, water, pea seed, water, carrot seed\nYou are on: none\nInventory (0/2): empty\nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You give the object. o: The water and the object transform into the object. a: You give the object. o: The water and the object transform into the object. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the object. o: The potato seed and the object transform into the potato. a: You pick up the object. o: You are holding the potato. a: You go to the baby pig. o: You are standing on the baby",
    # ALP
    "a: You go to the potato seed. o: You are standing on the potato seed. a: You pick up the object. o: You are holding the giraffe. a: You go to the beet seed. o: You are standing on the beet seed. a: You pick up the object. o: You are holding the giraffe and the beet. a: You go to the beet. o: You are standing on the beet. a: You go to the beet. o: You are standing on the beet. a: You go to the water. o: You are standing on the water. a: You give the carrot seed. o: The carrot seed and the water transform into the carrot. a: You give the beet. o: The beet grade and the water transform into the beet. a: You give the potato seed. o: The potato seed and",
    "In the current space: You see: baby pig, water, beet seed, baby giraffe, water, berry seed, water, beet seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water and the water. a: You go to the baby pig. o: You are standing on the baby pig. a: You go to the berry seed. o: You are standing on the berry seed. a: You give the water. o: The water and the berry seed transform into the berry. a: You pick up the",
    "a: You go to the cow. o: You are standing on the cow. a: You go to the water. o: You are standing on the water. a: You give the cow. o: The cow and the water transform into the cow. a: You pick up the object. o: You are holding the cow. a: You go to the potato seed. o: You are standing on the potato seed. a: You pick up the object. o: You are holding the cow and the potato seed. a: You go to the water. o: You are standing on the water. a: You give the potato seed. o: The potato seed and the water transform into the potato. a: You pick up the object. o: You are holding the cow and the potato. a: You go to the berry. o: You are standing on the berry. a: You go to the berry.",
    "In the current space: You see: baby cow, water, beet seed, baby giraffe, water, carrot seed, water, carrot seed\nYou are on:nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the baby cow. o: You are standing on the baby cow. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You gives the object. o: You are holding the carrot seed. a: You goes to the baby cow. o: You are standing on the baby cow. a: You picks the object. o: You are holding the carrot seed and the baby cow. a: Goes to the water. o: You are standing on the water. a: Gives the carrot seed. o: The carrot seed and the water transform into the carrot. a",
    "In the current space: You see: baby sheep, water, carrot seed, baby giraffe, water, carrot seed, water, beet seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You go to the baby sheep. o: You are standing on the baby sheep. a: You go to the water. o: You are standing on the water. a: You give the carrot. o: The carrot and the baby sheep transform into the sheep. a: You pick up the object. o: You are holding the beet seed and the beet seed. a: You go to the water. o: You are standing on the water. a: You give the beet seed. o: The beet seed and the water transform into the beet. a",
    "a: Choose the berry seed.\no: You are standing on the berry seed.\n\na: You go to the pea seed.\no: You are standing on the pea seed.\n\na: You pick up the object.\no: You are holding the pea seed.\n\na: You go to the berry seed.\no: You are standing on the berry seed.\n\na: You pick up the object.\no: You are holding the pea seed and the berry seed.\n\na: You go to the potato seed.\no: You are standing on the potato seed.\n\na: You go to the water.\no: You are standing on the water.\n\na: You give the pea seed.\no: The pea seed and the water transform into the pea.\n\na: You go to the potato seed.\n",
    "1: When a plant is mentioned, pick it up. o: You have the plant.\n2: If the plant you have is an animal or the plant you are on is not linked to a plant, go to the baby pig. o: You are standing on the baby pig.\n\nIn the current space: Set: You see: baby sheep, water, carrot seed, baby sheep, water, carrot seed, water, carrot seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the baby sheep. o: You are standing on the baby sheep. a: You go to the water. o: You are standing on the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You go to the baby sheep. o: You are standing on the baby sheep.",
    "In the current space: baby sheep, water, potato seed, rhinoceros, water, sunflower seed, water, carrot seed\nYou are on: rhinoceros\nInventory (0/2): empty\nThe sequence of actions and observations are:\na: You go to the water. o: You are standing on the water.\na: You pick up the object. o: You are holding the water.\na: You go to the water. o: You are standing on the water.\na: You pick up the object. o: You are holding the water and the water.\na: You go to the water. o: You are standing on the water.\na: You go to the beet seed. o: You are standing on the beet seed.\na: You give the water. o: The water and the beet seed transform into the beet.\na: You pick up",
    # ALPLL
    "- Going to the water: You are standing on the water.\n- Picking up the object: You are holding the water.\n- You go to the beet seed: You are standing on the beet seed.\n- You give the water: The water and the beet seed transform into the beet.\n- You give the baby pig: The baby pig and the beet transform into the pig.\n- You go to the water: You are standing on the water.\n- Picking up the object: You are holding the water.\n- You go to the yellow seed: You are standing on the yellow seed.\n- To get more clear and concise, here are the simplified action rules:\n1. If you stand at the water, you are on the water.\n2. Hold the object → You hold the water and the object.\n3. You give the water to a seed → The water and the seed",
    "a: You go to the baby sheep. o: You are standing on the baby sheep. a: You pick up the object. o: You are holding the baby sheep and the water. a: You give the water. o: The water and the water transform into the berry. a: You give the baby sheep. o: The baby sheep and the berry transform into the sheep. a: You pick up the object. o: You are holding the sheep. a: You go to the water. o: You are standing on the water. a: You go to the water. o: You are standing on the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The water and the potato seed transform into the potato. a: You pick up the object. o: You are holding the sheep and the potato. a: You go to the",
    "In this current space: You see: baby sheep, water, potato seed, baby giraffe, water, carrot seed, water, carrot seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the water. o: You are standing on the water. a: You go to the water. o: You are standing on the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The water and the potato seed transform into the potato. a: You pick up the object. o: You are holding the water and the potato. a: You go to the beet seed.",
    "In the original space: You see: baby pig, water, watermelon seed, caramel seed, water, watermelon seed, water, carrot seed\nYou are on: nothing\nInventory (0/2): empty. \nThe sequence of actions and observations is:  a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water and the water. a: You go to the carrot seed. o: You are standing on the carrot seed. a: You give the water. o: The water and the carrot seed transform into the carrot. a: You pick up the object. o: You are holding the cow and the carrot. a: You",
    "a: You go to [object]. o: You are standing on the [object].\nb: You give [object]. o: The [object] and the water transform into the [object].\nc: You go to [object]. o: You are standing on the [object].\nd: You go to the water. o: You are standing on the water.\ne: You give [object]. o: The [object] and the water transform into the [object].\nh: You pick up [object]. o: You are holding the [object] and the water.\ni: You go to the water. o: You are standing on the water.\nj: You pick up [object]. o: You are holding the [object].\n\n1. You go to the [object]. o: You are standing on the [object].\n2. You give the water. o: The water and the [object] transform into the",
    "Go to the potato seed. O: You are standing on the potato seed.\nPick up the object. O: You are holding the potato seed and the water.\nGo to the water. O: You are standing on the water.\nGive the potato seed. O: The potato seed and the water transform into the potato.\nPick up the object. O: You are holding the potato and the water.\nGo to the water. O: You are standing on the water.\nGive the water. O: The water and the potato seed transform into the potato.\nGo to the berry seed. O: You are standing on the berry seed.\nGive the water. O: The water and the berry seed transform into the berry.\nPick up the object. O: You are holding the berry and the potato.\nGo to the water. O: You",
    "a: You go to the potato. o: You are standing on the potato.\na: You go to the pea. o: You are standing on the pea.\nYou pick up the object. o: You are holding the water and the water.\nYou go to the carrot. o: You are standing on the carrot.\nYou go to the water. o: You are standing on the water.\nYou pick up the object. o: You are holding the water.\na: You give the water. o: The water and the beet seed transform into the beet.\nYou pick up the object. o: You are holding the beet and the water.\nYou go to the potato. o: You are standing on the potato.\nYou give the water. o: The water and the pea seed transform into the pea.\na: You pick up the object. o: You are",
    "The sequence of actions and observations is:  a: You go to the potato seed. o: You are standing on the potato seed. a: You go to the water. o: You are standing on the water. a: You pick up the object. o: You are holding the water and the water. a: You go to the water. o: You are standing on the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the water. o: The water and the potato seed transform into the potato. a: You go to the water. o: You are standing on the water. a: You go to the potato seed. o: You are standing on the potato seed. a: You give the carrot seed. o: The water and the carrot seed transform into the carrot. a: You go to the potato seed. o:",
]
algorithm_used: List[Optional[str]] = [
    None,
]
for key in ["PB", "RB", "CP", "DB", "ALP", "ALPLL"]:
    algorithm_used.extend([key] * 8)
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
