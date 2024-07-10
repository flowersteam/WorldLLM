import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf

from montecarlo_methods.important_sampling import important_sampling
from utils_env import BaseAgent
from utils_llm import load_transformers
from worldllm_envs.envs.base import BaseRuleEnv


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))
    theorist = load_transformers(cfg.theorist)
    if cfg.statistician is not None:
        statistician = load_transformers(cfg.statistician)
    else:
        statistician = theorist
    # Instantiate the environment and the agent
    env = gym.make(cfg.env.id, **cfg.env.kwargs)
    if not isinstance(env.unwrapped, BaseRuleEnv):
        raise ValueError("The environment must be rule based.")
    agent = hydra.utils.instantiate(cfg.agent, action_space=env.action_space)
    if not isinstance(agent, BaseAgent):
        raise ValueError("The agent must inherit from BaseAgent.")
    # Run the algorithm
    if cfg.algorithm.name == "importance_sampling":
        important_sampling(env, agent, theorist, statistician, cfg.algorithm)
    else:
        raise NotImplementedError(f"Algorithm {cfg.algorithm} not implemented.")


if __name__ == "__main__":
    main()
