import hydra
import torch
from omegaconf import DictConfig
from torch import nn


def instantiate_optimizer(cfg: DictConfig, model: nn.Module) -> torch.optim.Optimizer:
    lr = cfg.optimizer.lr
    lr_Z_0 = cfg.optimizer.lr_Z_0

    # Hacky delete because of lr_Z_0
    del cfg.optimizer.lr
    del cfg.optimizer.lr_Z_0

    optimizer = hydra.utils.instantiate(
        cfg.optimizer,
        [
            {"params": model.parameters(), "lr": lr},
            {"params": model.log_Z_0, "lr": lr_Z_0},
        ],
    )
    return optimizer


def instantiate(cfg: DictConfig):
    env = hydra.utils.instantiate(cfg.env)
    model = hydra.utils.instantiate(
        cfg.model, state_dims=env.STATE_DIM, action_dims=env.ACTION_DIM
    )

    buffer = hydra.utils.instantiate(cfg.buffer, env=env)

    optimizer = instantiate_optimizer(cfg, model)

    hydra.utils.instantiate(cfg.wandb)

    return env, model, optimizer, buffer, cfg.training
