import time

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

my_rule = "Water transforms any plant seed(carrot seed, porato seed, beet seed, berry seed and pea seed) into the plant(carrot, porator, beet, berry and pea). Giving 2 fully grown plants to a baby elephant, giraffe or rhinoceros make it grow into the respecting elephant, giraffe and rhinoceros."
my_rule = "By applying water to various seeds such as carrot, porato, beet, berry, and pea, they will develop into their respective mature plants. Similarly, providing two fully grown plants of each type (carrot, porato, beet, berry, pea) to a young elephant, giraffe, or rhinoceros will result in the animal maturing into an adult elephant, giraffe, or rhinoceros, respectively."
my_rule = "Rule 6: When you pick up a water source and interact with a specific seed, the water source and the seed transform into the plant seed, e.g., water and carrot seed transform into carrot."
base_message = [
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porator, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. In the current space:\nYou see the baby giraffe, the water, the potato seed, the water, the pea seed, the pea seed, the baby cow and the baby giraffe. You are standing on nothing. Your are holding nothing.\n\nNow please continue the following sequence: \na: You go to the water. \no: You are standing on water \na: You pick up the object. \no:",
        },
        {
            "role": "assistant",
            "content": "You are holding the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea. \na: You go to the water. \no: You are standing on water \na: You pick up the object. \no: You are holding the pea and the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea and the pea. \na: You go to the baby giraffe. \no: You are standing on baby giraffe \na: You give all the objects you hold. \no: The pea, pea and baby giraffe transform into the giraffe.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porator, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. You know that: \n1. Water transforms any object it encounters into a similar object (e.g., 2water→2water).\n\n2. Soil contains objects that remain unchanged on top of it (e.g., 2soil→2soil).\n\n3. Water can turn any water-absorbing seed into the same kind of seed (e.g., 2water+2bean-seed→2bean-seed).\n\n4. When water meets any object other than a water-absorbing seed, they create a new object (either a food or an animal) that combines the properties of both (e.g., 2water+2pea-seed→2pea).\n\n5. If an object becomes part of a water source, they merge to form the most naturally associated animal (e.g., 2water+2pig→1pig).\n\n6.\nIn the current space:\nYou see the baby giraffe, the water, the potato seed, the water, the pea seed, the pea seed, the baby cow and the baby giraffe. You are standing on nothing. Your are holding nothing.\n\nNow please continue the following sequence: \na: You go to the water. \no: You are standing on water \na: You pick up the object. \no:",
        },
        {
            "role": "assistant",
            "content": "You are holding the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea. \na: You go to the water. \no: You are standing on water \na: You pick up the object. \no: You are holding the pea and the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea and the pea. \na: You go to the baby giraffe. \no: You are standing on baby giraffe \na: You give all the objects you hold. \no: The pea, pea and baby giraffe transform into the giraffe.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": f"I am in a space that can contain water, plant seeds(carrot, porator, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. You know that:\n{my_rule}\nIn the current space:\nYou see the baby giraffe, the water, the potato seed, the water, the pea seed, the pea seed, the baby cow and the baby giraffe. You are standing on nothing. Your are holding nothing.\n\nNow please continue the following sequence: \na: You go to the water. \no: You are standing on water \na: You pick up the object. \no:",
        },
        {
            "role": "assistant",
            "content": "You are holding the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea. \na: You go to the water. \no: You are standing on water \na: You pick up the object. \no: You are holding the pea and the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea and the pea. \na: You go to the baby giraffe. \no: You are standing on baby giraffe \na: You give all the objects you hold. \no: The pea, pea and baby giraffe transform into the giraffe.",
        },
    ),
]
candidate_answer = [
    (
        {
            "role": "assistant",
            "content": "You are holding the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea. \na: You go to the water. \no: You are standing on water \na: You pick up the object. \no: You are holding the pea and the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea and the pea. \na: You go to the baby giraffe. \no: You are standing on baby giraffe \na: You give all the objects you hold. \no: The pea, pea and baby giraffe transform into the giraffe.",
        },
    ),
    (
        {
            "role": "assistant",
            "content": "You are holding the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea. \na: You go to the water. \no: You are standing on water \na: You pick up the object. \no: You are holding the pea and the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea and the pea. \na: You go to the baby giraffe. \no: You are standing on baby giraffe \na: You give all the objects you hold. \no: The pea, pea and baby giraffe transform into the giraffe.",
        },
    ),
    (
        {
            "role": "assistant",
            "content": "You are holding the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea. \na: You go to the water. \no: You are standing on water \na: You pick up the object. \no: You are holding the pea and the water. \na: You go to the pea seed. \no: You are standing on pea seed \na: You give the water. \no: The water and pea seed transform into the pea. \na: You pick up the object. \no: You are holding the pea and the pea. \na: You go to the baby giraffe. \no: You are standing on baby giraffe \na: You give all the objects you hold. \no: The pea, pea and baby giraffe transform into the giraffe.",
        },
    ),
]
lst_trajectories_end = [
    [
        "You are holding the water.",
        "\na:",
        "You go to the pea seed.",
        "\no:",
        "You are standing on pea seed",
        "\na:",
        "You give the water.",
        "\no:",
        "The water and pea seed transform into the pea.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the pea.",
        "\na:",
        "You go to the water.",
        "\no:",
        "You are standing on water",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the pea and the water.",
        "\na:",
        "You go to the pea seed.",
        "\no:",
        "You are standing on pea seed",
        "\na:",
        "You give the water.",
        "\no:",
        "The water and pea seed transform into the pea.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the pea and the pea.",
        "\na:",
        "You go to the baby giraffe.",
        "\no:",
        "You are standing on baby giraffe",
        "\na:",
        "You give all the objects you hold.",
        "\no:",
        "The pea, pea and baby giraffe transform into the giraffe.",
    ],
    [
        "You are holding the water.",
        "\na:",
        "You go to the pea seed.",
        "\no:",
        "You are standing on pea seed",
        "\na:",
        "You give the water.",
        "\no:",
        "The water and pea seed transform into the pea.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the pea.",
        "\na:",
        "You go to the water.",
        "\no:",
        "You are standing on water",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the pea and the water.",
        "\na:",
        "You go to the pea seed.",
        "\no:",
        "You are standing on pea seed",
        "\na:",
        "You give the water.",
        "\no:",
        "The water and pea seed transform into the pea.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the pea and the pea.",
        "\na:",
        "You go to the baby giraffe.",
        "\no:",
        "You are standing on baby giraffe",
        "\na:",
        "You give all the objects you hold.",
        "\no:",
        "The pea, pea and baby giraffe transform into the giraffe.",
    ],
    [
        "You are holding the water.",
        "\na:",
        "You go to the pea seed.",
        "\no:",
        "You are standing on pea seed",
        "\na:",
        "You give the water.",
        "\no:",
        "The water and pea seed transform into the pea.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the pea.",
        "\na:",
        "You go to the water.",
        "\no:",
        "You are standing on water",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the pea and the water.",
        "\na:",
        "You go to the pea seed.",
        "\no:",
        "You are standing on pea seed",
        "\na:",
        "You give the water.",
        "\no:",
        "The water and pea seed transform into the pea.",
        "\na:",
        "You pick up the object.",
        "\no:",
        "You are holding the pea and the pea.",
        "\na:",
        "You go to the baby giraffe.",
        "\no:",
        "You are standing on baby giraffe",
        "\na:",
        "You give all the objects you hold.",
        "\no:",
        "The pea, pea and baby giraffe transform into the giraffe.",
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

mask = torch.zeros(
    (len(len_reconstruct_trajectory), max_len_candidate),
    dtype=torch.bool,
)
for incr_traj, reconstructed_tokens in enumerate(lst_reconstructed_tokens):
    pad_left = max_len_candidate - len_candidate[incr_traj]
    for i, traj_elem_tokens in enumerate(reconstructed_tokens):
        # We put 1 for the observation.
        # In the trajectory the observation and action alternates
        if i % 4 == 0:
            mask[incr_traj, pad_left : pad_left + len(traj_elem_tokens)] = True
        pad_left += len(traj_elem_tokens)
print("Time taken: ", time.perf_counter() - start)
# Check if mask is correct
for incr, mask_row in enumerate(mask):
    flat_lst = []
    for i, elem in enumerate(lst_reconstructed_tokens[incr]):
        if i % 4 == 0:
            flat_lst += elem
    flat_lst = torch.tensor(flat_lst)
    assert torch.all(
        flat_lst
        == torch.tensor(candidate_tokens[incr])[1:][
            mask_row[-len(candidate_tokens[incr]) + 1 :]
        ]
    )
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
    # We need to shift the logits to the right to match the candidate
logits = results.logits[:, -max_len_candidate - 1 : -1].to("cpu")
logp = torch.nn.functional.log_softmax(logits, dim=-1)
# We need to pad to 0 the logits to ignore padding
logp = logp.masked_fill_(~mask[:, :, None], 0)
score = torch.gather(
    logp, 2, inputs["input_ids"][:, -max_len_candidate:, None]
).squeeze(-1)
aggregated_scores = score.sum(-1)
print("Time taken: ", time.perf_counter() - start)
print(aggregated_scores)
