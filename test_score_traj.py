import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"

tokenizer.chat_template = chat_template


base_message = [
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porator, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. You know that the rule is 'If an object (plant seed or herbivore) is moved onto another object (plant or herbivore), it spawns the object.'. Predict how the environment will change based on my actions.\nYou see the baby sheep, the water, the carrot seed, the water, the baby pig, the baby giraffe, the baby giraffe and the potato seed. You are standing on nothing. Your are holding nothing. You go to the water. You are standing on the water. You grasp the object.",
        },
        {
            "role": "assistant",
            "content": "You are holding the water. You go to the potato seed. You are standing on the potato seed. You release the water. The potato seed grows into the potato. You grasp the object. You are holding the potato. You go to the baby sheep. You are standing on the baby sheep. You release the potato. The baby sheep grows into the sheep.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porator, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. You know that the rule is 'Plant seeds with water, move organism, release that (transforms into plant or animal)\n\nThis rule describes the basic process of growth as seen in the given scenarios. It states that when a plant seed is placed in water and an object is removed from that seed, the seed grows into the respective plant. Similarly, when an organism (starting as a small baby) and water are placed together, the organism grows into the respective adult animal. This rule applies universally'. Predict how the environment will change based on my actions.\nYou see the beet seed, the water, the baby elephant, the carrot seed, the water, the baby pig, the baby pig and the berry seed. You are standing on nothing. Your are holding nothing. You go to the water. You are standing on the water. You grasp the object.",
        },
        {
            "role": "assistant",
            "content": "You are holding the water. You go to the beet seed. You are standing on the beet seed. You release the water. The beet seed grows into the beet.",
        },
    ),
]
candidate_answer = [
    (
        {
            "role": "assistant",
            "content": "You are holding the water. You go to the potato seed. You are standing on the potato seed. You release the water. The potato seed grows into the potato. You grasp the object. You are holding the potato. You go to the baby sheep. You are standing on the baby sheep. You release the potato. The baby sheep grows into the sheep.",
        },
    ),
    (
        {
            "role": "assistant",
            "content": "You are holding the water. You go to the beet seed. You are standing on the beet seed. You release the water. The beet seed grows into the beet.",
        },
    ),
]
lst_trajectories_end = [
    [
        "You are holding the water.",
        "You go to the potato seed.",
        "You are standing on the potato seed.",
        "You release the water.",
        "The potato seed grows into the potato.",
        "You grasp the object.",
        "You are holding the potato.",
        "You go to the baby sheep.",
        "You are standing on the baby sheep.",
        "You release the potato.",
        "The baby sheep grows into the sheep.",
    ],
    [
        "You are holding the water.",
        "You go to the beet seed.",
        "You are standing on the beet seed.",
        "You release the water.",
        "The beet seed grows into the beet.",
    ],
]
start = time.perf_counter()
true_tokens = tokenizer.apply_chat_template(base_message, add_generation_prompt=False)
print("Time taken: ", time.perf_counter() - start)
candidate_tokens = tokenizer.apply_chat_template(
    candidate_answer,
    add_generation_prompt=False,
)
print("Time taken: ", time.perf_counter() - start)
lst_reconstructed_tokens = []
len_reconstruct_trajectory = []
for trajectory_end in lst_trajectories_end:
    reconstructed_tokens = tokenizer(trajectory_end).data["input_ids"]
    lst_reconstructed_tokens.append(reconstructed_tokens)
    # We need to add the bos_token and eos_token to keep same format as if we applied the chat template
    len_reconstructed_tokens = 0
    for i, lst_reconstructed_token in enumerate(reconstructed_tokens):
        if i == 0 or i == len(reconstructed_tokens) - 1:
            len_reconstructed_tokens += (
                len(lst_reconstructed_token) + 1
            )  # For the bos or eos token
        else:
            len_reconstructed_tokens += len(lst_reconstructed_token)
    len_reconstruct_trajectory.append(len_reconstructed_tokens)
print("Time taken: ", time.perf_counter() - start)

# We need to create a mask to know which tokens are the observations
len_candidate = torch.tensor([c - 1 for c in len_reconstruct_trajectory])
max_len_candidate = torch.max(len_candidate).item()

mask = torch.zeros(
    (len(len_reconstruct_trajectory), max_len_candidate), dtype=torch.bool
)
for incr_traj, reconstructed_tokens in enumerate(lst_reconstructed_tokens):
    pad_left = max_len_candidate - len_candidate[incr_traj]
    for i, traj_elem_tokens in enumerate(reconstructed_tokens):
        # We put 1 for the observation.
        # In the trajectory the observation and action alternates
        if i % 2 == 0:
            mask[incr_traj, pad_left : pad_left + len(traj_elem_tokens)] = True
        pad_left += len(traj_elem_tokens)
print("Time taken: ", time.perf_counter() - start)

for incr in range(len(mask)):
    flat_lst = []
    for i, elem in enumerate(lst_reconstructed_tokens[incr]):
        if i % 2 == 0:
            flat_lst += elem
    flat_lst = torch.tensor(flat_lst)
    assert torch.all(
        flat_lst
        == torch.tensor(candidate_tokens[incr])[1:][
            mask[incr][-len(candidate_tokens[incr]) + 1 :]
        ]
    )
print()