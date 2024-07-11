import warnings
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Tuple

import torch
from omegaconf import DictConfig
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

from worldllm_envs.envs.base import EnvPromptInfo


@dataclass
class PromptInfo:
    """Prompting info to give to the LLM"""

    system_prompt: str
    message_template: Callable[..., str]


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
    stat_prompt_info, th_prompt_info = build_prompt_info(
        statistician,
        theorist,
        env_prompt_info,
        cfg.algorithm.stat_sys_prompt,
        cfg.algorithm.th_sys_prompt,
    )
    # Set prompt information
    statistician = LlmModel(statistician[0], statistician[1], stat_prompt_info)
    theorist = LlmModel(theorist[0], theorist[1], th_prompt_info)
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
        quantization_config = BitsAndBytesConfig(load_in_4bit=True)
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


def build_prompt_info(
    statistician: Tuple[AutoModelForCausalLM, AutoTokenizer],
    theorist: Tuple[AutoModelForCausalLM, AutoTokenizer],
    env_prompt_info: EnvPromptInfo,
    base_stat_prompt: str,
    base_th_prompt: str,
):
    """Build the prompts for the llms"""
    return (
        build_stat_prompt_info(statistician, env_prompt_info, base_stat_prompt),
        build_th_prompt_info(theorist, env_prompt_info, base_th_prompt),
    )


def build_th_prompt_info(
    model: Tuple[AutoModelForCausalLM, AutoTokenizer],
    env_prompt_info: EnvPromptInfo,
    base_system_prompt: str,
):
    """Build the prompt and message necessary for the Theorist."""
    return PromptInfo(
        base_system_prompt + env_prompt_info.th_prompt, env_prompt_info.th_template
    )


def build_stat_prompt_info(
    model: Tuple[AutoModelForCausalLM, AutoTokenizer],
    env_prompt_info: EnvPromptInfo,
    base_system_prompt: str,
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
        system_prompt,
        env_prompt_info.stat_template,
        lst_token_id,
    )


def _generate_rule(theorist: LlmModel, lst_message: List[str]):
    """Generate rule given message batch"""
    generation_args = {
        "temperature": 0.7,
        "do_sample": True,
        "max_new_tokens": 1024,
        "top_p": 0.95,
        "top_k": 50,
        "repetition_penalty": 1.2,
        "output_scores": True,
        "return_dict_in_generate": True,
        "num_return_sequences": 10,
    }


def generate_rules(
    theorist: LlmModel,
    trajectories: List[str],
    nb_rules: int,
    batch_size: int = 20,
):
    """Generate rules given the trajectories."""
    lst_messages = []
    for trajectory in trajectories:
        lst_messages.append(
            [
                {
                    "role": "system",
                    "content": theorist.prompt_info.system_prompt,
                },
                {
                    "role": "user",
                    "content": theorist.prompt_info.message_template(trajectory),
                },
            ]
        )
    all_scores = []
    for incr in range(0, len(lst_test_rule) * len(combinations), batch_size):
        all_scores.append(
            compute_scores(
                llm,
                tokenizer,
                generation_args,
                closed_token_id,
                opened_token_id,
                lst_messages[incr : incr + batch_size],
            )
        )
    scores = torch.cat(all_scores, dim=0)

    scores = torch.stack(torch.split(scores, len(combinations)))
    opened_door_true = torch.tensor(
        [door_rule.is_opened(combination) for combination in combinations]
    ).to("cpu")
    log_probability = torch.cumsum(
        torch.log(torch.where(opened_door_true, scores[:, :, 1], scores[:, :, 0])),
        dim=1,
    )


def compute_likelihood_scores(
    stat_data: LlmModel,
    generation_args: Dict[str, Any],
    tokens: List[int],
    lst_messages: List[str],
) -> torch.Tensor:
    """Compute the score for each message"""
    inputs = stat_data.tokenizer.apply_chat_template(
        lst_messages,
        add_generation_prompt=True,
        return_tensors="pt",
        padding=True,
    ).to(stat_data.model.device)
    scores = stat_data.model.generate(inputs, **generation_args).scores[0]
    scores = scores[:, tokens]
    scores = scores.softmax(dim=-1).cpu()
    return scores


def compute_likelihood(
    statistician: LlmModel,
    rules: List[str],
    trajectories: List[str],
    batch_size: int = 20,
):
    """Compute the likelihood of the new data given the rules."""
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
    all_scores = []
    for incr in range(0, len(rules) * len(trajectories), batch_size):
        all_scores.append(
            compute_likelihood_scores(
                statistician,
                generation_args,
                tokens_to_compare,
                lst_messages[incr : incr + batch_size],
            )
        )
    scores = torch.cat(all_scores, dim=0)

    scores = torch.stack(torch.split(scores, len(trajectories)))
    assert False, "Implement likelihood on random trajs"
    opened_door_true = torch.tensor(
        [door_rule.is_opened(combination) for combination in trajectories]
    ).to("cpu")
    log_probability = torch.cumsum(
        torch.log(torch.where(opened_door_true, scores[:, :, 1], scores[:, :, 0])),
        dim=1,
    )
