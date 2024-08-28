import os
import random

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from montecarlo_methods.important_sampling import important_sampling
from montecarlo_methods.metropolis_hastings import metropolis_hastings
from utils.utils_env import BaseAgent, build_env
from utils.utils_llm import build_llms
from worldllm_envs.envs.base import BaseRuleEnv


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
    # Set Rule
    if cfg.environment.rule is not None:
        env_rule = env.generate_rule(OmegaConf.to_object(cfg.environment)["rule"])
    else:
        env_rule = env.generate_rule()
    env.reset(options={"rule": env_rule})
    # Set Agent
    agent = hydra.utils.instantiate(cfg.agent, action_space=env.action_space)
    if not isinstance(agent, BaseAgent):
        raise ValueError("The agent must inherit from BaseAgent.")
    # Load LLMs
    statistician, theorist = build_llms(cfg, env.unwrapped.get_message_info())
    # Print gpu ram usage
    print(f"GPU RAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # Run the algorithm
    if cfg.algorithm.name == "importance_sampling":
        output = important_sampling(
            env, agent, theorist, statistician, OmegaConf.to_object(cfg.algorithm)
        )
    elif cfg.algorithm.name == "metropolis_hastings":
        output = metropolis_hastings(
            env, agent, theorist, statistician, OmegaConf.to_object(cfg.algorithm)
        )
    else:
        raise NotImplementedError(f"Algorithm {cfg.algorithm} not implemented.")
    output.to_json(os.path.join(cfg.output_dir, "all.json"))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
