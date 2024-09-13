import time
from typing import List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="auto",
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
            "content": "I am in a space that can contain water, plant seeds(carrot, porator, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. You know that the rule is 'If an object (plant seed or herbivore) is moved onto another object (plant or herbivore), it spawns the object.'. In the current space:\nYou see the baby sheep, the water, the carrot seed, the water, the baby pig, the baby giraffe, the baby giraffe and the potato seed. You are standing on nothing. Your are holding nothing. \n\nNow please continue the following sequence: \na: You go to the water. \no: You are standing on the water. \na: You grasp the object. \no:",
        },
        {
            "role": "assistant",
            "content": "You are holding the water. \na: You go to the potato seed. \no: You are standing on the potato seed. \na: You release the water. \no: The potato seed grows into the potato. \na: You grasp the object. \no: You are holding the potato. \na: You go to the baby sheep. \no: You are standing on the baby sheep. \na: You release the potato. \no: The baby sheep grows into the sheep.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porator, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. You know that the rule is 'Plant seeds with water, move organism, release that (transforms into plant or animal)\n\nThis rule describes the basic process of growth as seen in the given scenarios. It states that when a plant seed is placed in water and an object is removed from that seed, the seed grows into the respective plant. Similarly, when an organism (starting as a small baby) and water are placed together, the organism grows into the respective adult animal. This rule applies universally'. Predict the next observation based on the previous actions and observations and use the same words:\nYou see the beet seed, the water, the baby elephant, the carrot seed, the water, the baby pig, the baby pig and the berry seed. You are standing on nothing. Your are holding nothing. You go to the water. You are standing on the water. You grasp the object.",
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
            "content": "You are holding the water. \na: You go to the potato seed. \no: You are standing on the potato seed. \na: You release the water. \no: The potato seed grows into the potato. \na: You grasp the object. \no: You are holding the potato. \na: You go to the baby sheep. \no: You are standing on the baby sheep. \na: You release the potato. \no: The baby sheep grows into the sheep.",
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
        "\na:",
        "You go to the potato seed.",
        "\no:",
        "You are standing on the potato seed.",
        "\na:",
        "You release the water.",
        "\no:",
        "The potato seed grows into the potato.",
        "\na:",
        "You grasp the object.",
        "\no:",
        "You are holding the potato.",
        "\na:",
        "You go to the baby sheep.",
        "\no:",
        "You are standing on the baby sheep.",
        "\na:",
        "You release the potato.",
        "\no:",
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
candidate_tokens = tokenizer.apply_chat_template(
    candidate_answer,
    add_generation_prompt=False,
)
print("Time taken: ", time.perf_counter() - start)
lst_reconstructed_tokens = []
len_reconstruct_trajectory = []
for incr, trajectory_end in enumerate(lst_trajectories_end):
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
    assert candidate_tokens[incr][1:-1] == [
        x for xs in reconstructed_tokens for x in xs
    ]
print("Time taken: ", time.perf_counter() - start)

# We need to create a mask to know which tokens are the observations
len_candidate = torch.tensor([c - 1 for c in len_reconstruct_trajectory])
max_len_candidate = torch.max(len_candidate).item()

all_indices_obs = []
for incr_traj, reconstructed_tokens in enumerate(lst_reconstructed_tokens):
    indices: List[List[int]] = []
    pad_left = max_len_candidate - len_candidate[incr_traj]
    for i, traj_elem_tokens in enumerate(reconstructed_tokens):
        if i % 4 == 0:
            indices.append(list(range(pad_left, pad_left + len(traj_elem_tokens))))
        pad_left += len(traj_elem_tokens)
    all_indices_obs.append(indices)


print("Time taken: ", time.perf_counter() - start)
# Check if mask is correct
for incr, index_row in enumerate(all_indices_obs):
    flat_lst = []
    for i, elem in enumerate(lst_reconstructed_tokens[incr]):
        if i % 4 == 0:
            flat_lst += elem
    flat_lst = torch.tensor(flat_lst)
    all_tokens = torch.tensor(
        [
            candidate_tokens[incr][1:][index - index_row[0][0]]
            for index_transition in index_row
            for index in index_transition
        ]
    )
    assert torch.all(flat_lst == all_tokens)
print("Time taken: ", time.perf_counter() - start)
with torch.no_grad():
    inputs = tokenizer.apply_chat_template(
        base_message,
        add_generation_prompt=False,
        return_tensors="pt",
        padding=True,
        return_dict=True,
    )
    inputs = inputs.to(model.device)
    results = model(
        inputs["input_ids"],
        attention_mask=inputs["attention_mask"],
    )
    inputs = inputs.to("cpu")
print("Time taken: ", time.perf_counter() - start)
# We need to shift the logits to the right to match the candidate
logits = results.logits[:, -max_len_candidate - 1 : -1].to("cpu")
logp = torch.nn.functional.log_softmax(logits, dim=-1)
# We need to pad to 0 the logits to ignore padding
traj_scoring = []
all_transition_scoring = []
for incr_traj, traj_obs_ind in enumerate(all_indices_obs):
    transition_scoring = []
    for incr_transi, obs_ind in enumerate(traj_obs_ind):
        values = logp[
            incr_traj,
            obs_ind,
            inputs["input_ids"][incr_traj, -max_len_candidate:][obs_ind],
        ]
        transition_scoring.append(values.sum(-1).item())

    all_transition_scoring.append(transition_scoring)
    traj_scoring.append(sum(transition_scoring))
print("Time taken: ", time.perf_counter() - start)
print(traj_scoring)
