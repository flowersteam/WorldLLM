import warnings
from typing import Any, Dict, Tuple

import torch
import yaml
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


def load_transformers(
    model_config: Dict[str, Any]
) -> Tuple[AutoModelForCausalLM, AutoTokenizer]:
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

    if model_config["chat_template"]:
        tokenizer.chat_template = model_config["chat_template"]
    return model, tokenizer
