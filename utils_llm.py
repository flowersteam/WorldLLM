import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

import numpy as np
import torch
from omegaconf import DictConfig
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from utils_env import Trajectory
from worldllm_envs.envs.base import EnvPromptInfo


@dataclass
class PromptInfo:
    """Prompting info to give to the LLM"""

    system_prompt: str
    message_template: Callable[..., str]
    batch_size: int


@dataclass
class StatPromptInfo(PromptInfo):
    """Prompting info to give to the Statistician"""

    tokens: List[int]


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
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_config["name"], **model_config["tokenizer_params"]
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
    llm, tokenizer = model
    lst_token_id = []
    for token in env_prompt_info.tokens:
        tokens_id = tokenizer.encode(token, add_special_tokens=False)
        if len(tokens_id) > 1:
            raise NotImplementedError("Words with multiple tokens not supported yet.")
        lst_token_id.append(tokens_id[0])

    system_prompt = base_system_prompt + env_prompt_info.stat_prompt
    return StatPromptInfo(
        system_prompt, env_prompt_info.stat_template, batch_size, lst_token_id
    )


def _generate_rule(
    theorist: LlmModel, lst_message: List[str], generation_args: Dict[str, Any]
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Generate rule given message batch"""
    inputs = theorist.tokenizer.apply_chat_template(
        lst_message,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(theorist.model.device)
    results = theorist.model.generate(inputs, **generation_args)
    generated_sequences = results.sequences[:, inputs.shape[-1] :]
    generated_rules = theorist.tokenizer.batch_decode(
        generated_sequences, skip_special_tokens=True
    )
    logp = torch.nn.functional.log_softmax(torch.stack(results.scores, dim=1), dim=-1)
    scores = torch.gather(logp, 2, generated_sequences[:, :, None]).squeeze(-1)
    # Change score from -inf to 0 to ignore padding
    scores = scores.masked_fill_(scores == -torch.inf, 0)
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
        "max_new_tokens": 100,
        "do_sample": True,
        "output_scores": True,
        "return_dict_in_generate": True,
    }
    generation_args.update(theorist.generation_kwargs)
    trajectories = [trajectory.get_full_text() for trajectory in trajectories]
    all_rules = []
    all_log_probs = []
    for batch in tqdm(
        range(0, nb_rules, theorist.prompt_info.batch_size), desc="Generating rules"
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


def compute_likelihood_scores(
    statistician: LlmModel,
    generation_args: Dict[str, Any],
    tokens: List[int],
    lst_message: List[str],
) -> torch.Tensor:
    """Compute the score for each message"""
    inputs = statistician.tokenizer.apply_chat_template(
        lst_message, add_generation_prompt=True, return_tensors="pt", padding=True
    ).to(statistician.model.device)
    results = statistician.model.generate(inputs, **generation_args)
    scores = results.scores[0]
    scores = scores[:, tokens]
    scores = scores.softmax(dim=-1).cpu()
    return scores


def compute_likelihood(
    statistician: LlmModel,
    rules: List[str],
    trajectories: List[Trajectory],
) -> np.ndarray:
    """Compute the likelihood of the new data given the rules."""
    assert isinstance(
        trajectories[0].obs[0], int
    ), "We only support discrete observation for the environment for now."
    # Generation args
    generation_args = {
        "temperature": 1,
        "do_sample": True,
        "max_new_tokens": 1,
        "top_k": None,
        "top_p": 1,
        "output_scores": True,
        "return_dict_in_generate": True,
        "pad_token_id": statistician.tokenizer.pad_token_id,
    }
    lst_messages = []
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
    batch_size = statistician.prompt_info.batch_size
    all_scores = []
    for incr in tqdm(
        range(0, len(rules) * len(trajectories), batch_size),
        desc="Computing likelihood",
    ):
        all_scores.append(
            compute_likelihood_scores(
                statistician,
                generation_args,
                statistician.prompt_info.tokens,
                lst_messages[incr : incr + batch_size],
            )
        )
    scores = torch.cat(all_scores, dim=0)

    scores = torch.stack(torch.split(scores, len(trajectories)))
    obs_to_predict = torch.tensor([trajectory.obs[-1] for trajectory in trajectories])

    log_probability = torch.log(
        scores[:, torch.arange(len(trajectories)), obs_to_predict]
    ).sum(dim=1)
    return log_probability.numpy()
