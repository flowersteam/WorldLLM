import os
import random
from functools import partial

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from montecarlo_methods.important_sampling import important_sampling
from montecarlo_methods.metropolis_hastings import metropolis_hastings
from utils.utils_env import BaseAgent, build_env
from utils.utils_llm import build_llms
from utils.utils_sb3 import create_agent
from worldllm_envs.base import BaseRuleEnv


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
    # Set Agent
    if cfg.agent.type == "BaseAgent":
        agent = hydra.utils.instantiate(cfg.agent, action_space=env.action_space)
        # Set rule
        if cfg.environment.rule is not None:
            rules = OmegaConf.to_object(cfg.environment)["rule"]
            if not isinstance(rules, list):
                env_rules = [env.generate_rule(rules)]
            else:
                env_rules = [env.generate_rule(rule) for rule in rules]
        else:
            env_rules = [env.generate_rule()]

    elif cfg.agent.type == "SB3Agent":
        assert isinstance(
            cfg.environment.rule, str
        ), "The rule must be defined for SB3Agent and be unique."
        env_rules = [cfg.environment.rule]
        agent = create_agent(
            cfg.agent, partial(build_env, cfg, rule=env_rules[0]), seed=cfg.seed
        )
    else:
        raise NotImplementedError(
            f"Agent {cfg.agent.type} not implemented. Choose between BaseAgent and SB3Agent."
        )
    # Load LLMs
    statistician, theorist = build_llms(cfg, env.unwrapped.get_message_info())
    # Print gpu ram usage
    print(f"GPU RAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # Run the algorithm
    if cfg.algorithm.name == "importance_sampling":
        output = important_sampling(
            env,
            agent,
            theorist,
            statistician,
            OmegaConf.to_object(cfg.algorithm),
            curriculum_rules=env_rules,
        )
    elif cfg.algorithm.name == "metropolis_hastings":
        output = metropolis_hastings(
            env,
            agent,
            theorist,
            statistician,
            OmegaConf.to_object(cfg.algorithm),
            curriculum_rules=env_rules,
        )
    else:
        raise NotImplementedError(f"Algorithm {cfg.algorithm} not implemented.")
    output.to_json(os.path.join(cfg.output_dir, "all.json"))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
