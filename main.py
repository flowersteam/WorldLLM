import gymnasium as gym
import hydra
from omegaconf import DictConfig, OmegaConf

from montecarlo_methods.important_sampling import important_sampling
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
    env = gym.make(cfg.env.id, **cfg.env.kwargs)
    if not isinstance(env, BaseRuleEnv):
        return ValueError("The environment must be rule based.")
    if cfg.algorithm == "importance_sampling":
        important_sampling(env, theorist, statistician, cfg)
    else:
        raise NotImplementedError(f"Algorithm {cfg.algorithm} not implemented.")


if __name__ == "__main__":
    main()
