import hydra
from omegaconf import DictConfig, OmegaConf

from utils_llm import load_transformers


@hydra.main(version_base=None, config_path="configs", config_name="config")
def main(cfg: DictConfig) -> None:
    llm = load_transformers(cfg.llm)
    print(OmegaConf.to_yaml(cfg))


if __name__ == "__main__":
    main()
