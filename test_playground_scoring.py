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

test_content = "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. \n"
test_content += "Your objective is to predict the next observation in the sequence given the past actions and observations. The sequence will be under this form, with x,y and z 3 objects and a an action:\n\n In the current space:\nYou see x, y, and z. You are standing on the y. Your are holding nothing. \na: action. \no: You are standing on x. \na: action. \no: You are holding y. \na: action. \no: x and y transform into z. \na: action. \no: Nothing has changed."
test_content += "\n\nNow please complete the sequence:\n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \na: You go to the water. \no: "

base_message = [
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": test_content,
        },
        {
            "role": "assistant",
            "content": "You are standing on the water. \na: You pick up the object. \no: You are holding the water. \na: You go to the beet seed. \no: You are standing on the beet seed. \na: You give the water. \no: The water and beet seed transform into the beet. \na: You pick up the object. \no: You are holding the beet. \na: You go to the water. \no: You are standing on the water. \na: You pick up the object. \no: You are holding the beet and the water. \na: You go to the potato seed. \no: You are standing on the potato seed. \na: You give the water. \no: The water and potato seed transform into the potato. \na: You pick up the object. \no: You are holding the beet and the potato. \na: You go to the baby rhinoceros. \no: You are standing on the baby rhinoceros. \na: You give all the objects you hold. \no: The potato, beet and baby rhinoceros transform into the rhinoceros.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. \nYour objective is to predict the next observation in the sequence given the past actions and observations. The sequence will be under this form:\n\n In the current space:\nYou see the baby sheep, the water, the carrot seed, the baby rhinoceros, the beet seed, the pea seed, the water and the potato seed. You are standing on the baby rhinoceros. Your are holding nothing. \na: You go to the water. \no: You are standing on the water. \na: You pick up the object. \no: You are holding the water. \na: You go to the potato seed. \no: You are standing on the potato seed. \na: You give the water. \no: The water and potato seed transform into the potato. \na: You pick up the object. \no: You are holding the potato. \na: You go to the baby sheep. \no: You are standing on the baby sheep. \na: You give the potato. \no: The potato and baby sheep transform into the sheep. \n\nNow please complete the sequence:\n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \na: You go to the water. \no:",
        },
        {
            "role": "assistant",
            "content": "You are standing on the water. \na: You pick up the object. \no: You are holding the water. \na: You go to the beet seed. \no: You are standing on the beet seed. \na: You give the water. \no: The water and beet seed transform into the beet. \na: You pick up the object. \no: You are holding the beet. \na: You go to the water. \no: You are standing on the water. \na: You pick up the object. \no: You are holding the beet and the water. \na: You go to the potato seed. \no: You are standing on the potato seed. \na: You give the water. \no: The water and potato seed transform into the potato. \na: You pick up the object. \no: You are holding the beet and the potato. \na: You go to the baby rhinoceros. \no: You are standing on the baby rhinoceros. \na: You give all the objects you hold. \no: The potato, beet and baby rhinoceros transform into the rhinoceros.",
        },
    ),
]
candidate_answer = [
    (
        {
            "role": "assistant",
            "content": "You are standing on the water. \na: You pick up the object. \no: You are holding the water. \na: You go to the beet seed. \no: You are standing on the beet seed. \na: You give the water. \no: The water and beet seed transform into the beet. \na: You pick up the object. \no: You are holding the beet. \na: You go to the water. \no: You are standing on the water. \na: You pick up the object. \no: You are holding the beet and the water. \na: You go to the potato seed. \no: You are standing on the potato seed. \na: You give the water. \no: The water and potato seed transform into the potato. \na: You pick up the object. \no: You are holding the beet and the potato. \na: You go to the baby rhinoceros. \no: You are standing on the baby rhinoceros. \na: You give all the objects you hold. \no: The potato, beet and baby rhinoceros transform into the rhinoceros.",
        },
    ),
    (
        {
            "role": "assistant",
            "content": "You are standing on the water. \na: You pick up the object. \no: You are holding the water. \na: You go to the beet seed. \no: You are standing on the beet seed. \na: You give the water. \no: The water and beet seed transform into the beet. \na: You pick up the object. \no: You are holding the beet. \na: You go to the water. \no: You are standing on the water. \na: You pick up the object. \no: You are holding the beet and the water. \na: You go to the potato seed. \no: You are standing on the potato seed. \na: You give the water. \no: The water and potato seed transform into the potato. \na: You pick up the object. \no: You are holding the beet and the potato. \na: You go to the baby rhinoceros. \no: You are standing on the baby rhinoceros. \na: You give all the objects you hold. \no: The potato, beet and baby rhinoceros transform into the rhinoceros.",
        },
    ),
]
lst_trajectories_end = [
    [
        "You are standing on the water.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the water.",
        "\na:",
        "You go to the beet seed.",
        "\no:",
        "You are standing on the beet seed.",
        "\na:",
        "You give the water.",
        "\no:",
        "The water and beet seed transform into the beet.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the beet.",
        "\na:",
        "You go to the water.",
        "\no:",
        "You are standing on the water.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the beet and the water.",
        "\na:",
        "You go to the potato seed.",
        "\no:",
        "You are standing on the potato seed.",
        "\na:",
        "You give the water.",
        "\no:",
        "The water and potato seed transform into the potato.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the beet and the potato.",
        "\na:",
        "You go to the baby rhinoceros.",
        "\no:",
        "You are standing on the baby rhinoceros.",
        "\na:",
        "You give all the objects you hold.",
        "\no:",
        "The potato, beet and baby rhinoceros transform into the rhinoceros.",
    ],
    [
        "You are standing on the water.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the water.",
        "\na:",
        "You go to the beet seed.",
        "\no:",
        "You are standing on the beet seed.",
        "\na:",
        "You give the water.",
        "\no:",
        "The water and beet seed transform into the beet.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the beet.",
        "\na:",
        "You go to the water.",
        "\no:",
        "You are standing on the water.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the beet and the water.",
        "\na:",
        "You go to the potato seed.",
        "\no:",
        "You are standing on the potato seed.",
        "\na:",
        "You give the water.",
        "\no:",
        "The water and potato seed transform into the potato.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the beet and the potato.",
        "\na:",
        "You go to the baby rhinoceros.",
        "\no:",
        "You are standing on the baby rhinoceros.",
        "\na:",
        "You give all the objects you hold.",
        "\no:",
        "The potato, beet and baby rhinoceros transform into the rhinoceros.",
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
print(traj_scoring, all_transition_scoring)
