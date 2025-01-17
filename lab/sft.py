# flake8: noqa
# Copyright 2023 The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from argparse import ArgumentParser

import torch
from datasets import Dataset

from trl import SFTTrainer, DataCollatorForCompletionOnlyLM
from transformers import TrainingArguments, AutoModel, AutoTokenizer

from unsloth import FastLanguageModel, is_bfloat16_supported
import json



if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--seed', type=int)
    parser.add_argument('--dataset_path', type=str)
    parser.add_argument('--model_path', type=str)
    parser.add_argument('--output_dir', type=str)
    args = parser.parse_args()

    ################
    # Model init kwargs & Tokenizer
    ################
    max_seq_length = 4096
    model, tokenizer = FastLanguageModel.from_pretrained(
        args.model_path,
        load_in_4bit=True,
        dtype=None,
        max_seq_length=max_seq_length
    )
    # model = AutoModel.from_pretrained(args.model_path)
    # tokenizer = AutoTokenizer.from_pretrained("microsoft/Phi-3-mini-4k-instruct")
    tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.chat_template = "{% for message in messages %}{% if message['role'] == 'system' and message['content'] %}{{'<|system|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'user' %}{{'<|user|>\n' + message['content'] + '<|end|>\n'}}{% elif message['role'] == 'assistant' %}{{'<|assistant|>\n' + message['content'] + '<|end|>\n'}}{% endif %}{% endfor %}{% if add_generation_prompt %}{{ '<|assistant|>\n' }}{% else %}{{ eos_token }}{% endif %}"

    ################
    # Dataset
    ################
    with open(args.dataset_path, 'r') as file:
        raw_dataset = json.load(file)

    def gen():
        for _row in raw_dataset:
            yield {"messages": _row}

    dataset = Dataset.from_generator(gen)
    # dataset = load_dataset("json", data_files=args.dataset_path)
    response_template = "<|assistant|>\n"  # Warning: this is specific to Phi
    collator = DataCollatorForCompletionOnlyLM(response_template=response_template, tokenizer=tokenizer)
    def formatting_prompts_func(examples):
        texts = []
        for example in examples["messages"]:
            text = tokenizer.apply_chat_template(example, tokenize=False, add_generation_prompt=False)
            assert len(tokenizer.encode(text.split(response_template)[1].split("<|end|>")[0], add_special_tokens=False)) > 0
            texts.append(text)

        return {"text": texts}

    dataset = dataset.map(formatting_prompts_func, batched=True)

    ################
    # Training
    ################
    # Do model patching and add fast LoRA weights
    model = FastLanguageModel.get_peft_model(
        model,
        r=16,
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj",
                        "gate_proj", "up_proj", "down_proj", ],
        lora_alpha=16,
        lora_dropout=0,  # Supports any, but = 0 is optimized
        bias="none",  # Supports any, but = "none" is optimized
        use_gradient_checkpointing=True,
        random_state=args.seed,
        max_seq_length=max_seq_length,
    )

    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        args=TrainingArguments(
            per_device_train_batch_size=16,
            # gradient_accumulation_steps=4,
            learning_rate=2.0e-4,
            num_train_epochs=1,
            fp16=not torch.cuda.is_bf16_supported(),
            bf16=torch.cuda.is_bf16_supported(),
            logging_steps=500,
            output_dir=args.output_dir,
            logging_dir=args.output_dir + "/run/",
            report_to=["tensorboard"],
            save_strategy="epoch",
            save_only_model=True,
            optim="adamw_8bit",
            seed=args.seed,
            eval_strategy="no",
        ),
        data_collator=collator,
        dataset_text_field="text",
        packing=False
    )
    trainer.train()