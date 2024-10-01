import time
from typing import Dict, List

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

quantization_config = None
model_name = "microsoft/Phi-3-mini-4k-instruct"
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    device_map="cuda",
    torch_dtype="auto",
    trust_remote_code=True,
    quantization_config=quantization_config,
)
tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side="left")
chat_template = "{{ bos_token }}{% for message in messages %}{% if (message['role'] == 'system') %}{{'<|system|>' + '\n' + message['content'] + '<|end|>' + '\n'}}{% elif (message['role'] == 'user') %}{{'<|user|>' + '\n' + message['content'] + '<|end|>' + '\n' + '<|assistant|>' + '\n'}}{% elif message['role'] == 'assistant' %}{{message['content'] + '<|end|>' + '\n'}}{% endif %}{% endfor %}"

tokenizer.chat_template = chat_template


def generate_answer(msg: List[Dict[str, str]]):
    inputs = tokenizer.apply_chat_template(
        msg,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        return_dict=True,
    ).to(model.device)

    generation_args = {
        "temperature": 1,
        "top_k": None,
        "top_p": 1,
        "max_new_tokens": 200,
        "do_sample": True,
        "output_scores": True,
        "return_dict_in_generate": True,
    }

    outputs = model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"], **generation_args
    )
    generated_sequences = outputs.sequences[:, inputs["input_ids"].shape[-1] :]
    generated_rules = tokenizer.batch_decode(
        generated_sequences, skip_special_tokens=True
    )
    logp = torch.nn.functional.log_softmax(torch.stack(outputs.scores, dim=1), dim=-1)
    # Put the score of the padding token to 0 to ignore(not done by every model)
    logp[:, :, tokenizer.pad_token_id] = 0
    scores = torch.gather(logp, 2, generated_sequences[:, :, None]).squeeze(-1)
    aggregated_scores = scores.sum(-1)
    return generated_rules, aggregated_scores.cpu()


def score_answer(lst_message, lst_candidate):
    candidate = tokenizer.apply_chat_template(
        lst_candidate,
        add_generation_prompt=False,
    )
    # Create mask to get only the generated part
    # We remove the bos token
    len_candidate = torch.tensor([len(c) - 1 for c in candidate])
    max_len_candidate = torch.max(len_candidate).item()
    index_mat = torch.arange(max_len_candidate)
    generation_mask = torch.where(
        index_mat[None, :] >= max_len_candidate - len_candidate[:, None], 1, 0
    )
    with torch.no_grad():
        inputs = tokenizer.apply_chat_template(
            lst_message,
            add_generation_prompt=False,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        ).to(model.device)
        results = model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    # We need to shift the logits to the right to match the candidate
    logits = results.logits[:, -max_len_candidate - 1 : -1]
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    # We need to pad to 0 the logits to ignore padding
    generation_mask = generation_mask.to(logp.device)
    logp = logp.masked_fill_(generation_mask[:, :, None] == 0, 0)
    score = torch.gather(
        logp, 2, inputs["input_ids"][:, -max_len_candidate:, None]
    ).squeeze(-1)
    aggregated_scores = score.sum(-1)
    return aggregated_scores.cpu()


def score_reuse_answer(base_message, lst_candidate):
    """Score and reuse the pas key and values"""
    inputs_candidate = tokenizer.apply_chat_template(
        lst_candidate, add_generation_prompt=False, return_dict=True
    )
    # Pad the candidate to the same length and remove the bos token
    max_len_candidate = max(len(c) - 1 for c in inputs_candidate["input_ids"])
    inputs_candidate_ids = torch.stack(
        [
            torch.nn.functional.pad(
                torch.tensor(input_ids[1:]),
                (0, max_len_candidate - len(input_ids) + 1),
                value=tokenizer.pad_token_id,
            )
            for input_ids in inputs_candidate["input_ids"]
        ]
    ).to("cuda")
    inputs_candidate_mask = torch.stack(
        [
            torch.nn.functional.pad(
                torch.tensor(attention_mask[1:]),
                (0, max_len_candidate - len(attention_mask) + 1),
                value=0,
            )
            for attention_mask in inputs_candidate["attention_mask"]
        ]
    ).to("cuda")
    with torch.no_grad():
        inputs_base = tokenizer.apply_chat_template(
            base_message,
            add_generation_prompt=False,
            return_tensors="pt",
            return_dict=True,
        ).to(model.device)
        results = model(
            inputs_base["input_ids"],
            attention_mask=inputs_base["attention_mask"],
            use_cache=True,
        )
        last_logits = results.logits[0, -1]
        # Create full mask
        full_mask = torch.cat(
            [
                inputs_base["attention_mask"].repeat(len(inputs_candidate_mask), 1),
                inputs_candidate_mask,
            ],
            dim=1,
        )
        # Duplicate the past key and values
        past_key_values = results.past_key_values
        all_keys_values = []
        for past_key_value in past_key_values:
            all_keys_values.append(
                (
                    past_key_value[0].repeat(len(inputs_candidate_mask), 1, 1, 1),
                    past_key_value[1].repeat(len(inputs_candidate_mask), 1, 1, 1),
                )
            )
        past_key_values = tuple(all_keys_values)
        results = model(
            inputs_candidate_ids,
            attention_mask=full_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
    logits = torch.cat(
        (
            last_logits[None, None, :].repeat(len(results.logits), 1, 1),
            results.logits[:, :-1, :],
        ),
        dim=1,
    )
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    # We need to pad to 0 the logits to ignore padding
    logp = logp.masked_fill_(inputs_candidate_mask[:, :, None] == 0, 0)
    score = torch.gather(logp, 2, inputs_candidate_ids[:, :, None]).squeeze(-1)
    aggregated_scores = score.sum(-1)
    return aggregated_scores.cpu()


all_message = [
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
        {"role": "assistant", "content": "Grasp"},
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
        {"role": "assistant", "content": "Go to baby rhinoceros"},
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
        {"role": "assistant", "content": "Go to water"},
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
        {"role": "assistant", "content": "Go to pea seed"},
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
        {"role": "assistant", "content": "Go to water"},
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
        {"role": "assistant", "content": "Go to potato seed"},
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
        {"role": "assistant", "content": "Go to baby rhinoceros"},
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
        {"role": "assistant", "content": "Go to potato seed"},
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
        {"role": "assistant", "content": "Go to beet seed"},
    ),
]
base_message = [
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
    ),
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and be as precise as you can.",
        },
        {
            "role": "user",
            "content": "I am in a space that can contain water, plant seeds(carrot, porato, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). I can move an object, a plant or a herbivore and place it on another object to make them interact. Your objective is to take the best action given the past actions and observations. The possible action are: \nGrasp \nGo to baby rhinoceros \nGo to water \nGo to pea seed \nGo to water \nGo to potato seed \nGo to baby rhinoceros \nGo to potato seed \nGo to beet seed \n\nThe current sequence of experience is: \n\nIn the current space:\nYou see the baby rhinoceros, the water, the pea seed, the water, the potato seed, the baby rhinoceros, the potato seed and the beet seed. You are standing on nothing. Your are holding nothing. \n\nWhat is the best action to take? Answer with just the action.",
        },
    ),
]
true_answer = [
    ({"role": "assistant", "content": "Grasp"},),
    ({"role": "assistant", "content": "Go to baby rhinoceros"},),
    ({"role": "assistant", "content": "Go to water"},),
    ({"role": "assistant", "content": "Go to pea seed"},),
    ({"role": "assistant", "content": "Go to water"},),
    ({"role": "assistant", "content": "Go to potato seed"},),
    ({"role": "assistant", "content": "Go to baby rhinoceros"},),
    ({"role": "assistant", "content": "Go to potato seed"},),
    ({"role": "assistant", "content": "Go to beet seed"},),
]
gen_rules, gen_logp = generate_answer(base_message)

candidate_answer = [
    (
        {
            "role": "assistant",
            "content": gen_rule,
        },
    )
    for gen_rule in gen_rules
]
reconstructed_message = [
    base_message[i] + candidate_answer[i] for i in range(len(candidate_answer))
]
scoring = score_answer(reconstructed_message, candidate_answer)
scoring2 = score_reuse_answer(base_message[0], candidate_answer)

print("rules:", gen_rules)
print("Score:", scoring, "Score2:", scoring2, "generated answer:", gen_logp)
