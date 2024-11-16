import os
import random
from functools import partial
from typing import Tuple

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from montecarlo_methods.important_sampling import important_sampling
from montecarlo_methods.metropolis_hastings import metropolis_hastings
from utils.utils_llm import LlmAgent, LlmModel, Statistician, build_llms
from utils.utils_sb3 import SB3Agent
from worldllm_envs.base import BaseAgent, BaseRuleEnv, build_env


def load_modules(
    cfg: DictConfig, env: BaseRuleEnv
) -> Tuple[Statistician, LlmModel, BaseAgent]:
    """Load the modules Statistician, Theorist and Experimenter"""
    statistician, theorist = build_llms(cfg, env)
    if cfg.experimenter.type == "BaseAgent":
        experimenter_config = OmegaConf.to_object(cfg.experimenter)
        del (
            experimenter_config["type"],
        )  # Remove type key to avoid error on instantiation
        experimenter: BaseAgent = hydra.utils.instantiate(
            experimenter_config, action_space=env.action_space
        )

    elif cfg.experimenter.type == "SB3Agent":
        experimenter = SB3Agent.create_agent(
            cfg.experimenter,
            partial(build_env, cfg, rule=env.unwrapped.get_rule()),
            seed=cfg.seed,
        )
        # SB3 include the environment in the experimenter
    elif cfg.experimenter.type == "LLM":
        experimenter = LlmAgent.create_agent(cfg, env, theorist)
    else:
        raise NotImplementedError(f"Agent {cfg.experimenter.type} not implemented.")
    return statistician, theorist, experimenter


# To change the config file: -cn config_name.yaml, to modify the config file: key=value and to add a value: +key=value
@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # Set seed
    if cfg.seed is not None:
        np.random.seed(cfg.seed)
        random.seed(cfg.seed)
        torch.manual_seed(cfg.seed)
        torch.cuda.manual_seed_all(cfg.seed)
    # Instantiate the environment
    env: BaseRuleEnv = build_env(cfg)
    # Load env rules
    if cfg.environment.rule is not None:
        env_rule_info = OmegaConf.to_object(cfg.environment)["rule"]
        env_rule = env.generate_rule(env_rule_info)
    else:
        env_rule = env.generate_rule()
    env.reset(options={"rule": env_rule})
    # Load modules
    statistician, theorist, experimenter = load_modules(cfg, env)
    # Print gpu ram usage
    print(f"GPU RAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # Run the algorithm
    if cfg.algorithm.name == "importance_sampling":
        output = important_sampling(
            env,
            experimenter,
            theorist,
            statistician,
            OmegaConf.to_object(cfg.algorithm),
        )
    elif cfg.algorithm.name == "metropolis_hastings":
        output = metropolis_hastings(
            env,
            experimenter,
            theorist,
            statistician,
            OmegaConf.to_object(cfg.algorithm),
            cfg.output_dir,
        )
    else:
        raise NotImplementedError(f"Algorithm {cfg.algorithm} not implemented.")
    # Save output
    output.to_json(os.path.join(cfg.output_dir, "all.json"))
    # Save experimenter if sb3
    if isinstance(experimenter, SB3Agent):
        experimenter.model.save(os.path.join(cfg.output_dir, "experimenter"))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
