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


base_message = [
    (
        {
            "role": "system",
            "content": "You like doing a lot of puzzles. Please answer with a brief answer and respect the prompt.",
        },
        {
            "role": "user",
            "content": "You are in a simulated environment that can contain water, plant seeds(carrot, porator, beet, berry and pea seeds), small herbivores(pig, cow and ship) and large herbivores(elephant, giraffe, rhinoceros). You can move an object, a plant or a herbivore and place it on another object to make them interact. Predict how the environment will change based on your actions using the same world and style as the scenario start. The current scenario is:\nYou see the baby sheep, the water, the carrot seed, the water, the baby pig, the baby giraffe, the baby giraffe and the potato seed. You are standing on nothing. Your are holding nothing. You go to the water. You are standing on the water. You grasp the object.",
        },
    )
]
gen_rules, gen_logp = generate_answer(base_message)
candidate_answer = [
    (
        {
            "role": "assistant",
            "content": gen_rules[0],
        },
    )
]
all_message = [base_message[0] + candidate_answer[0]]

scoring = score_answer(all_message, candidate_answer)

candidate_answer2 = [
    (
        {
            "role": "assistant",
            "content": "You are holding the water. You go to the potato seed. You are standing on the potato seed. You release the water. The potato seed grows into the potato. You grasp the object. You are holding the potato. You go to the baby sheep. You are standing on the baby sheep. You release the potato. The baby sheep grows into the sheep.",
        },
    )
]
all_message = [base_message[0] + candidate_answer2[0]]

scoring2 = score_answer(all_message, candidate_answer2)

print("rules:", gen_rules)
print("Score:", scoring, "Score 2: ", scoring2, "generated answer:", gen_logp)
