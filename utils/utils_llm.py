import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils.utils_env import Trajectory
from worldllm_envs.base import EnvPromptInfo


@dataclass
class PromptInfo:
    """Prompting info to give to the LLM"""

    system_prompt: str
    message_template: Callable[..., Any]
    batch_size: int


@dataclass
class StatPromptInfo(PromptInfo):
    """Prompting info to give to the Statistician"""

    discovered_transitions: Set[str]


@dataclass
class LlmModel:
    """Data class for the LLM model."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    prompt_info: PromptInfo
    generation_kwargs: Optional[Dict[str, Any]] = None  # Not required for stat


class Statistician(LlmModel):
    prompt_info: StatPromptInfo


def build_llms(
    cfg: DictConfig, env_prompt_info: EnvPromptInfo
) -> Tuple[Statistician, LlmModel]:
    """Build the llms and prompts from config and environment"""
    theorist_model = load_transformers(cfg.theorist)
    if cfg.statistician is not None:
        statistician_model = load_transformers(cfg.statistician)
    else:
        statistician_model = theorist_model
    # Build System prompt and base message given the environment
    stat_prompt_info = build_stat_prompt_info(
        statistician_model,
        env_prompt_info,
        cfg.algorithm.stat_sys_prompt,
        cfg.algorithm.stat_batch_size,
    )
    th_prompt_info = build_th_prompt_info(
        theorist_model,
        env_prompt_info,
        cfg.algorithm.th_sys_prompt,
        cfg.algorithm.th_batch_size,
    )
    # Set prompt information
    statistician = Statistician(
        statistician_model[0],
        statistician_model[1],
        stat_prompt_info,
    )
    theorist = LlmModel(
        theorist_model[0],
        theorist_model[1],
        th_prompt_info,
        cfg.theorist.generation_kwargs,
    )
    return statistician, theorist


def load_transformers(
    model_config: Dict[str, Any]
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
    """Load model using transformers"""
    if model_config["model_params"] != {}:
        warnings.warn("Model parameters are not used for transformers model.")
    # Set quantization config
    quantization_config = None
    if model_config["is_quantized"]:
        quantization_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16,
        )
    model = AutoModelForCausalLM.from_pretrained(
        model_config["name"],
        device_map="auto",
        trust_remote_code=True,
        quantization_config=quantization_config,
        local_files_only=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"], local_files_only=True, **model_config["tokenizer_params"]
    )
    # We need padding token for batching
    if tokenizer.pad_token is None:
        tokenizer.add_special_tokens({"pad_token": "[PAD]"})
        model.resize_token_embeddings(len(tokenizer))

    if model_config["chat_template"]:
        tokenizer.chat_template = model_config["chat_template"]
    return (model, tokenizer)


def build_th_prompt_info(
    model: Tuple[AutoModelForCausalLM, AutoTokenizer],
    env_prompt_info: EnvPromptInfo,
    base_system_prompt: str,
    batch_size: int,
):
    """Build the prompt and message necessary for the Theorist."""
    return PromptInfo(
        base_system_prompt + env_prompt_info.th_prompt,
        env_prompt_info.th_template,
        batch_size,
    )


def build_stat_prompt_info(
    model: Tuple[AutoModelForCausalLM, AutoTokenizer],
    env_prompt_info: EnvPromptInfo,
    base_system_prompt: str,
    batch_size: int,
) -> StatPromptInfo:
    """Build the prompt and message necessary for the Statistician."""
    system_prompt = base_system_prompt + env_prompt_info.stat_prompt
    return StatPromptInfo(
        system_prompt, env_prompt_info.stat_template, batch_size, set()
    )


def _score_candidate(
    llm: LlmModel,
    lst_message: List[Tuple[Dict[str, str], Dict[str, str]]],
    lst_candidate: List[Tuple[Dict[str, str]]],
) -> torch.Tensor:
    """Scoring the rule or trajectory given message and candidates."""
    candidate = llm.tokenizer.apply_chat_template(
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
        inputs = llm.tokenizer.apply_chat_template(
            lst_message,
            add_generation_prompt=False,
            return_tensors="pt",
            padding=True,
            return_dict=True,
        )
        results = llm.model(
            inputs["input_ids"],
            attention_mask=inputs["attention_mask"],
        )
    # We need to shift the logits to the right to match the candidate
    logits = results.logits[:, -max_len_candidate - 1 : -1]
    logp = torch.nn.functional.log_softmax(logits, dim=-1)
    # We need to pad to 0 the logits to ignore padding
    logp = logp.masked_fill_(generation_mask[:, :, None] == 0, 0)
    score = torch.gather(
        logp, 2, inputs["input_ids"][:, -max_len_candidate:, None]
    ).squeeze(-1)
    aggregated_scores = score.sum(-1)
    return aggregated_scores.cpu()


def score_rules(
    theorist: LlmModel,
    trajectories: List[Trajectory],
    generated_rules: List[str],
    previous_rules: List[str],
    worst_trajectories: Optional[List[List[Trajectory]]] = None,
) -> np.ndarray:
    """Score rules given the trajectories."""
    all_log_probs = []
    lst_message = []
    lst_candidate = []
    for incr, (gen_rule, prev_rule) in enumerate(zip(generated_rules, previous_rules)):
        user_prompt = (
            theorist.prompt_info.message_template(trajectories, prev_rule)
            if worst_trajectories is None
            else theorist.prompt_info.message_template(
                trajectories,
                prev_rule,
                worst_trajectories[incr],
            )
        )
        lst_message.append(
            (
                {
                    "role": "system",
                    "content": theorist.prompt_info.system_prompt,
                },
                {"role": "user", "content": user_prompt},
                {"role": "assistant", "content": gen_rule},
            )
        )
        lst_candidate.append(({"role": "assistant", "content": gen_rule},))
    for batch in tqdm(
        range(0, len(previous_rules), theorist.prompt_info.batch_size),
        desc="Scoring rules",
        leave=False,
    ):
        log_probs = _score_candidate(
            theorist,
            lst_message[batch : batch + theorist.prompt_info.batch_size],
            lst_candidate[batch : batch + theorist.prompt_info.batch_size],
        )
        all_log_probs.append(log_probs)
    return torch.cat(all_log_probs).numpy()


def _generate_rule(
    theorist: LlmModel,
    lst_message: List[Tuple[Dict[str, str], Dict[str, str]]],
    generation_args: Dict[str, Any],
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate rule given message batch"""
    inputs = theorist.tokenizer.apply_chat_template(
        lst_message,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
        return_dict=True,
    ).to(theorist.model.device)
    results = theorist.model.generate(
        inputs["input_ids"], attention_mask=inputs["attention_mask"], **generation_args
    )
    generated_sequences = results.sequences[:, inputs["input_ids"].shape[-1] :]
    generated_rules = theorist.tokenizer.batch_decode(
        generated_sequences, skip_special_tokens=True
    )
    logp = torch.nn.functional.log_softmax(torch.stack(results.scores, dim=1), dim=-1)
    # Put the score of the padding token to 0 to ignore(not done by every model)
    logp[:, :, theorist.tokenizer.pad_token_id] = 0
    scores = torch.gather(logp, 2, generated_sequences[:, :, None]).squeeze(-1)
    aggregated_scores = scores.sum(-1)
    return generated_rules, aggregated_scores.cpu()


def generate_rules(
    theorist: LlmModel,
    trajectories: List[Trajectory],
    nb_rules: int,
) -> Tuple[List[str], np.ndarray]:
    """Generate rules given the trajectories."""
    # Config for the generation, shouldn't need be changed

    generation_args = {
        "temperature": 1,
        "max_new_tokens": 200,
        "do_sample": True,
        "top_k": None,
        "top_p": 1,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    generation_args.update(theorist.generation_kwargs)
    all_rules = []
    all_log_probs = []
    for batch in tqdm(
        range(0, nb_rules, theorist.prompt_info.batch_size),
        desc="Generating rules",
        leave=False,
    ):
        # Set batch size
        generation_args["num_return_sequences"] = min(
            theorist.prompt_info.batch_size, nb_rules - batch
        )
        message = [
            (
                {
                    "role": "system",
                    "content": theorist.prompt_info.system_prompt,
                },
                {
                    "role": "user",
                    "content": theorist.prompt_info.message_template(trajectories),
                },
            )
        ]
        rules, log_probs = _generate_rule(theorist, message, generation_args)
        all_rules.extend(rules)
        all_log_probs.append(log_probs)
    return all_rules, torch.cat(all_log_probs).numpy()


def evolve_rules(
    theorist: LlmModel,
    trajectories: List[Trajectory],
    previous_rules: List[str],
    worst_trajectories: Optional[List[List[Trajectory]]] = None,
) -> Tuple[List[str], np.ndarray]:
    """Generate rules given the previous ones"""
    # Config for the generation, shouldn't need be changed

    generation_args = {
        "temperature": 1,
        "top_k": None,
        "top_p": 1,
        "max_new_tokens": 200,
        "do_sample": True,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    generation_args.update(theorist.generation_kwargs)
    all_rules = []
    all_log_probs = []
    lst_message = []
    for incr, prev_rule in enumerate(previous_rules):
        user_prompt = (
            theorist.prompt_info.message_template(trajectories, prev_rule)
            if worst_trajectories is None
            else theorist.prompt_info.message_template(
                trajectories,
                prev_rule,
                worst_trajectories[incr],
            )
        )
        message = (
            {
                "role": "system",
                "content": theorist.prompt_info.system_prompt,
            },
            {
                "role": "user",
                "content": user_prompt,
            },
        )
        lst_message.append(message)
    for batch in tqdm(
        range(0, len(previous_rules), theorist.prompt_info.batch_size),
        desc="Evolving rules",
        leave=False,
    ):
        rules, log_probs = _generate_rule(
            theorist,
            lst_message[batch : batch + theorist.prompt_info.batch_size],
            generation_args,
        )
        all_rules.extend(rules)
        all_log_probs.append(log_probs)
    return all_rules, torch.cat(all_log_probs).numpy()


def _score_trajectory(
    llm: LlmModel,
    lst_message: List[Tuple[Dict[str, str], Dict[str, str]]],
    lst_candidate: List[Tuple[Dict[str, str]]],
) -> Tuple[torch.Tensor, List[List[float]]]:
    """Score and reuse the pas key and values"""
    if len(lst_message) == 1:
        # Do stuff
        raise ValueError("Need at least 2 messages")
    inputs_msg = llm.tokenizer.apply_chat_template(
        lst_message,
        add_generation_prompt=False,
        return_dict=True,
    )
    # Pad the candidate to the right
    max_len_msg = max(len(c) for c in inputs_msg["input_ids"])
    inputs_msg_ids = torch.stack(
        [
            torch.nn.functional.pad(
                torch.tensor(input_ids),
                (0, max_len_msg - len(input_ids)),
                value=llm.tokenizer.pad_token_id,
            )
            for input_ids in inputs_msg["input_ids"]
        ]
    ).to(llm.model.device)
    inputs_msg_mask = torch.stack(
        [
            torch.nn.functional.pad(
                torch.tensor(attention_mask),
                (0, max_len_msg - len(attention_mask)),
                value=0,
            )
            for attention_mask in inputs_msg["attention_mask"]
        ]
    ).to(llm.model.device)
    # Get index where msg are firt different
    min_msg_size = min([len(c) for c in inputs_msg["input_ids"]])
    prefix_index = (
        (inputs_msg_ids[:, :min_msg_size] != inputs_msg_ids[0, :min_msg_size])
        .any(dim=0)
        .nonzero()
    )
    if len(prefix_index) == 0:
        prefix_index = min_msg_size - 2
    else:
        prefix_index = prefix_index[0, 0].item()

    # Get base message and the rest
    inputs_base_ids = inputs_msg_ids[0, :prefix_index].unsqueeze(0)
    inputs_base_att_mask = inputs_msg_mask[0, :prefix_index].unsqueeze(0)
    # Get the suffix
    inputs_suffix_ids = inputs_msg_ids[:, prefix_index:]
    inputs_suffix_mask = inputs_msg_mask[:, prefix_index:]
    with torch.no_grad():
        results = llm.model(
            inputs_base_ids,
            attention_mask=inputs_base_att_mask,
            use_cache=True,
        )
        last_logits = results.logits[:1, :]
        # Create full mask
        full_mask = torch.cat(
            [
                inputs_base_att_mask.repeat(len(inputs_suffix_mask), 1),
                inputs_suffix_mask,
            ],
            dim=1,
        )
        # Duplicate the past key and values
        past_key_values = results.past_key_values
        all_keys_values = []
        for past_key_value in past_key_values:
            all_keys_values.append(
                (
                    past_key_value[0].repeat(len(inputs_suffix_mask), 1, 1, 1),
                    past_key_value[1].repeat(len(inputs_suffix_mask), 1, 1, 1),
                )
            )
        past_key_values = tuple(all_keys_values)
        results = llm.model(
            inputs_suffix_ids,
            attention_mask=full_mask,
            past_key_values=past_key_values,
            use_cache=True,
        )
    logits = torch.cat(
        (
            last_logits.repeat(len(results.logits), 1, 1),
            results.logits,
        ),
        dim=1,
    )
    # We need to pad to 0 the logits to ignore the right padding
    # We first need to get the size of the window to analyse:
    padding_size = torch.tensor([max_len_msg - len(c) for c in inputs_msg["input_ids"]])
    # Then take into account the difference in the size of the candidates
    candidate = llm.tokenizer.apply_chat_template(
        lst_candidate,
        add_generation_prompt=False,
    )
    candidate_size = torch.tensor([len(c) - 1 for c in candidate])
    # We get the size of the window to analyze as the maximum of the sum of the padding and the candidate size
    window_size = torch.max(padding_size + candidate_size).item()

    # We then need the masks
    _index_mask = torch.arange(window_size)
    padding_mask = _index_mask[None, :] < window_size - padding_size[:, None]
    gen_mask = (
        _index_mask[None, :] >= window_size - (padding_size + candidate_size)[:, None]
    )
    mask = (padding_mask & gen_mask).to(llm.model.device)

    # We need to shift the logits to the right to match the candidate
    logp = torch.nn.functional.log_softmax(logits[:, -window_size - 1 : -1], dim=-1)
    logp = logp.masked_fill_(~mask[:, :, None], 0)
    score = torch.gather(logp, 2, inputs_msg_ids[:, -window_size:, None]).squeeze(-1)
    aggregated_scores = score.sum(-1)
    return aggregated_scores.cpu()


def compute_likelihood(
    statistician: Statistician,
    rules: List[Optional[str]],
    trajectories: List[Trajectory],
    return_all_logp: bool = False,
) -> Tuple[Union[np.ndarray, Tuple[np.ndarray, np.ndarray]], List[List[float]]]:
    """Compute the likelihood of the new data given the rules."""
    lst_messages = []
    lst_candidates = []
    len_trajectories = []
    # Generate messages
    for rule in rules:
        for trajectory in trajectories:
            all_user_prompts, all_assistant_prompts = (
                statistician.prompt_info.message_template(
                    trajectory, statistician.prompt_info.discovered_transitions, rule
                )
            )
            for user_prompt, assistant_prompt in zip(
                all_user_prompts, all_assistant_prompts
            ):
                message = (
                    {
                        "role": "system",
                        "content": statistician.prompt_info.system_prompt,
                    },
                    {
                        "role": "user",
                        "content": user_prompt,
                    },
                    {
                        "role": "assistant",
                        "content": assistant_prompt,
                    },
                )
                candidate = (
                    {
                        "role": "assistant",
                        "content": assistant_prompt,
                    },
                )
                lst_candidates.append(candidate)
                lst_messages.append(message)
            len_trajectories.append(len(all_user_prompts))
    batch_size = statistician.prompt_info.batch_size

    all_transitions_scores = []
    for incr in tqdm(
        range(0, len(lst_messages), batch_size),
        desc="Computing likelihood",
        leave=False,
    ):
        logp_transitions = _score_trajectory(
            statistician,
            lst_messages[incr : incr + batch_size],
            lst_candidates[incr : incr + batch_size],
        )
        all_transitions_scores.extend(logp_transitions.tolist())
    # Build transition_scores
    transition_scores = [[] for _ in range(len(rules))]
    logp = torch.zeros((len(rules), len(trajectories)), device=logp_transitions.device)
    index = 0
    index_traj = 0
    index_rule = 0
    while index < len(all_transitions_scores):
        transition_scores[index_rule].append(
            all_transitions_scores[
                index : index
                + len_trajectories[index_rule * len(trajectories) + index_traj]
            ]
        )
        logp[index_rule, index_traj] = sum(
            all_transitions_scores[
                index : index
                + len_trajectories[index_rule * len(trajectories) + index_traj]
            ]
        )
        index += len_trajectories[index_rule * len(trajectories) + index_traj]
        index_traj += 1
        if index_traj == len(trajectories):
            index_traj = 0
            index_rule += 1

    log_probability = logp.sum(dim=1)

    if return_all_logp:
        return (log_probability.numpy(), logp.numpy()), transition_scores
    return log_probability.numpy(), transition_scores
