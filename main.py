import os
import random
from functools import partial
from typing import List

import hydra
import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf

from montecarlo_methods.important_sampling import important_sampling
from montecarlo_methods.metropolis_hastings import metropolis_hastings
from utils.utils_env import BaseAgent, build_env
from utils.utils_llm import build_llms
from utils.utils_sb3 import SB3Agent, create_agent
from worldllm_envs.base import BaseRuleEnv


def load_agent(cfg: DictConfig, env: BaseRuleEnv, env_rule: str) -> BaseAgent:
    """Load the agent used to collect data"""
    if cfg.agent.type == "BaseAgent":
        agent_config = OmegaConf.to_object(cfg.agent)
        del (agent_config["type"],)  # Remove type key to avoid error on instantiation
        agent = hydra.utils.instantiate(agent_config, action_space=env.action_space)
        env.reset(options={"rule": env_rule})

    elif cfg.agent.type == "SB3Agent":
        agent = create_agent(
            cfg.agent, partial(build_env, cfg, rule=env_rule), seed=cfg.seed
        )
        # SB3 include the environment in the agent
    else:
        raise NotImplementedError(
            f"Agent {cfg.agent.type} not implemented. Choose between BaseAgent and SB3Agent."
        )
    return agent


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
    # Set Agent
    agent = load_agent(cfg, env, env_rule)
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
        )
    elif cfg.algorithm.name == "metropolis_hastings":
        output = metropolis_hastings(
            env,
            agent,
            theorist,
            statistician,
            OmegaConf.to_object(cfg.algorithm),
        )
    else:
        raise NotImplementedError(f"Algorithm {cfg.algorithm} not implemented.")
    # Save output
    output.to_json(os.path.join(cfg.output_dir, "all.json"))
    # Save agent if sb3
    if isinstance(agent, SB3Agent):
        agent.model.save(os.path.join(cfg.output_dir, "agent"))


if __name__ == "__main__":
    main()  # pylint: disable=no-value-for-parameter
