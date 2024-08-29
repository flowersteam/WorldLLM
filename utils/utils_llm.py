import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

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
    message_template: Callable[..., str]
    batch_size: int


@dataclass
class LlmModel:
    """Data class for the LLM model."""

    model: AutoModelForCausalLM
    tokenizer: AutoTokenizer
    prompt_info: PromptInfo
    generation_kwargs: Optional[Dict[str, Any]] = None  # Not required for stat


def build_llms(
    cfg: DictConfig, env_prompt_info: EnvPromptInfo
) -> Tuple[LlmModel, LlmModel]:
    """Build the llms and prompts from config and environment"""
    theorist = load_transformers(cfg.theorist)
    if cfg.statistician is not None:
        statistician = load_transformers(cfg.statistician)
    else:
        statistician = theorist
    # Build System prompt and base message given the environment
    stat_prompt_info = build_stat_prompt_info(
        statistician,
        env_prompt_info,
        cfg.algorithm.stat_sys_prompt,
        cfg.algorithm.stat_batch_size,
    )
    th_prompt_info = build_th_prompt_info(
        theorist,
        env_prompt_info,
        cfg.algorithm.th_sys_prompt,
        cfg.algorithm.th_batch_size,
    )
    # Set prompt information
    statistician = LlmModel(
        statistician[0],
        statistician[1],
        stat_prompt_info,
    )
    theorist = LlmModel(
        theorist[0], theorist[1], th_prompt_info, cfg.theorist.generation_kwargs
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
) -> PromptInfo:
    """Build the prompt and message necessary for the Statistician."""
    system_prompt = base_system_prompt + env_prompt_info.stat_prompt
    return PromptInfo(system_prompt, env_prompt_info.stat_template, batch_size)


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
    trajectories = [trajectory.get_full_text() for trajectory in trajectories]
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
                [worst_traj.get_full_text() for worst_traj in worst_trajectories[incr]],
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
    theorist: LlmModel, lst_message: List[str], generation_args: Dict[str, Any]
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
        "max_new_tokens": 100,
        "do_sample": True,
        "top_k": None,
        "top_p": 1,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    generation_args.update(theorist.generation_kwargs)
    trajectories = [trajectory.get_full_text() for trajectory in trajectories]
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
        "max_new_tokens": 100,
        "do_sample": True,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    generation_args.update(theorist.generation_kwargs)
    trajectories = [trajectory.get_full_text() for trajectory in trajectories]
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
                [worst_traj.get_full_text() for worst_traj in worst_trajectories[incr]],
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


def compute_likelihood(
    statistician: LlmModel,
    rules: List[str],
    trajectories: List[Trajectory],
    return_all_logp: bool = False,
) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]:
    """Compute the likelihood of the new data given the rules."""
    assert isinstance(
        trajectories[0].obs[0], int
    ), "We only support discrete observation for the environment for now."
    lst_messages = []
    lst_candidate = []
    # Generate messages
    for rule in rules:
        for trajectory in trajectories:
            message = (
                {
                    "role": "system",
                    "content": statistician.prompt_info.system_prompt,
                },
                {
                    "role": "user",
                    "content": statistician.prompt_info.message_template(
                        rule, " ".join(trajectory.text[:-1])
                    ),
                },
            )
            lst_messages.append(message)
            lst_candidate.append(
                ({"role": "assistant", "content": trajectory.text[-1]},)
            )
    batch_size = statistician.prompt_info.batch_size
    all_logp = []
    for incr in tqdm(
        range(0, len(rules) * len(trajectories), batch_size),
        desc="Computing likelihood",
        leave=False,
    ):
        all_logp.append(
            _score_candidate(
                statistician,
                lst_messages[incr : incr + batch_size],
                lst_candidate[incr : incr + batch_size],
            )
        )
    logp = torch.cat(all_logp, dim=0)
    logp = torch.stack(torch.split(logp, len(trajectories)))

    log_probability = logp.sum(dim=1)

    if return_all_logp:
        return log_probability.numpy(), logp.numpy()
    return log_probability.numpy()
