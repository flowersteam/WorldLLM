# This script is made to construct from the trajectories collected during an
# experiment with the framework, a dataset for fine tunning an LLM.
# To build a test dataset, run the main script of the corresponding environment.
import argparse
import json
import os
import random
import sys

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from utils.utils_llm import build_llms
from worldllm_envs.base import BaseWrapper, Trajectory, build_env

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output_dir",
        type=str,
        help="Directory where the experiment is saved",
        required=True,
    )
    parser.add_argument(
        "--dataset_path",
        type=str,
        help="Path to save the dataset",
        default="datasets/temp",
    )
    output_dir = parser.parse_args().output_dir
    dataset_path = parser.parse_args().dataset_path
    for i_folder, folder in enumerate(os.listdir(output_dir)):
        # Load one of the config
        cfg: DictConfig = OmegaConf.load(
            os.path.join(output_dir, folder, "config.yaml")
        )
        # Instantiate the environment
        # We need to correct the dataset path in the config by going back one directory
        cfg.environment.kwargs.test_dataset_path = (
            "." + cfg.environment.kwargs.test_dataset_path
        )
        env: BaseWrapper = build_env(cfg)
        # Load the statistician
        statistician, _ = build_llms(cfg, env.unwrapped.get_message_info())
        # Load the trajectories
        with open(
            os.path.join(output_dir, folder, "all.json"), "r", encoding="utf-8"
        ) as f:
            data = json.load(f)

        all_trajectories = []
        for data_collection, trajectories_dict in enumerate(
            data["metrics"]["transitions"]
        ):
            all_trajectories.extend(
                [Trajectory.from_dict(traj) for traj in trajectories_dict]
            )
        # Build the dataset
        lst_messages = []
        for trajectory in all_trajectories:
            all_user_prompts, all_assistant_prompts = (
                statistician.prompt_info.message_template(
                    trajectory, env.unwrapped.get_all_transition_to_prompt(), None
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
                lst_messages.append(message)
        # Save the dataset
        with open(
            f"{dataset_path}/dataset_{i_folder}.json", "w", encoding="utf-8"
        ) as f:
            json.dump(lst_messages, f)
        print("Dataset saved at", dataset_path)
