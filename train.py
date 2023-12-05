import hydra
from omegaconf import DictConfig

from src.utils import instantiate
from src.tb import train


@hydra.main(version_base="1.3", config_path="configs", config_name="ttt.yaml")
def main(cfg: DictConfig) -> None:
    """Main entry point for training.

    :param cfg: DictConfig configuration composed by Hydra.
    :return: None
    """
    env, model, optimizer, buffer, train_cfg = instantiate(cfg)
    train(env, model, optimizer, buffer, train_cfg)


if __name__ == "__main__":
    main()
