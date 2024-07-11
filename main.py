import gymnasium as gym
import hydra
import torch
from omegaconf import DictConfig, OmegaConf

from montecarlo_methods.important_sampling import important_sampling
from utils_env import BaseAgent
from utils_llm import build_llms
from worldllm_envs.envs.base import BaseRuleEnv


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    # Instantiate the environment and the agent
    env = gym.make(cfg.env.id, **cfg.env.kwargs)
    if not isinstance(env.unwrapped, BaseRuleEnv):
        raise ValueError("The environment must be rule based.")
    agent = hydra.utils.instantiate(cfg.agent, action_space=env.action_space)
    if not isinstance(agent, BaseAgent):
        raise ValueError("The agent must inherit from BaseAgent.")
    # Load LLMs
    statistician, theorist = build_llms(cfg, env.unwrapped.get_message_info())
    # Print gpu ram usage
    print(f"GPU RAM usage: {torch.cuda.memory_allocated() / 1e9:.2f} GB")
    # Run the algorithm
    if cfg.algorithm.name == "importance_sampling":
        important_sampling(env, agent, theorist, statistician, cfg.algorithm)
    else:
        raise NotImplementedError(f"Algorithm {cfg.algorithm} not implemented.")


if __name__ == "__main__":
    main()
