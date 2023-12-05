import os
import torch
import torch.nn.functional as F
import wandb
from tqdm import tqdm
from torch import nn
from omegaconf import DictConfig

from src.envs.base_env import BaseEnv
from src.data import gen_batch_traj_buffer, TrajectoryBuffer
from src.eval import test_agent, UniformAgent

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


def compute_loss(afn: nn.Module, batch: tuple[torch.Tensor, ...]):
    states, masks, curr_players, actions, dones, log_reward = batch
    batch_size, traj_len, _, _, _ = states.shape

    log_numerator = torch.zeros(batch_size, device=DEVICE)
    log_denominator = torch.zeros(batch_size, device=DEVICE)

    log_numerator += afn.log_Z_0

    for t in range(traj_len):
        curr_states = states[:, t, :, :, :].squeeze(1)
        curr_masks = masks[:, t, :].squeeze(1)
        curr_player = t % 2
        curr_actions = actions[:, t].long()
        curr_not_dones = (dones[:, t].squeeze(1) == 1).bool()
        curr_terminal = (dones[:, t].squeeze(1) == 2).bool()

        if torch.any(curr_not_dones) or torch.any(curr_terminal):
            _, policy = afn(curr_states, curr_player)
            policy = policy * curr_masks - (1 - curr_masks) * 1e9
            probs = nn.functional.softmax(policy, dim=1)
            probs = probs.gather(1, curr_actions).squeeze(1)
            num_children = curr_masks.sum(dim=1)

            if curr_player == 0:
                log_numerator[curr_not_dones] += probs[curr_not_dones].log()
                log_numerator[curr_not_dones] += num_children[curr_not_dones].log()

            if curr_player == 1:
                log_denominator[curr_not_dones] += probs[curr_not_dones].log()
                log_denominator[curr_not_dones] += num_children[curr_not_dones].log()

            if torch.any(curr_terminal):
                log_denominator[curr_terminal] += log_reward[curr_terminal, 0]

    loss = F.mse_loss(log_numerator, log_denominator)

    return loss


def train_afn(
    afn: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: TrajectoryBuffer,
    batch_size=2048,
):
    batch = buffer.sample(batch_size)
    optimizer.zero_grad()

    loss = compute_loss(afn, batch)
    loss.backward()

    optimizer.step()

    return loss.item()


def train(
    env: BaseEnv,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    buffer: TrajectoryBuffer,
    cfg: DictConfig,
):
    # Setup checkpoints dir and path
    run_dir = os.path.join(
        cfg.ckpt_dir, wandb.run.id if wandb.run else wandb.util.generate_id()
    )
    if not os.path.exists(run_dir):
        os.makedirs(run_dir, exist_ok=True)

    print("Generating initial buffer")
    buffer = gen_batch_traj_buffer(
        buffer=buffer,
        env=env,
        afn=model,
        num_trajectories=cfg.num_initial_traj,
        batch_size=cfg.buffer_batch_size,
    )

    train_pbar = tqdm(range(cfg.total_steps), leave=False)
    train_pbar.set_description("Train")
    for step in train_pbar:
        # Train
        loss = train_afn(
            afn=model, optimizer=optimizer, buffer=buffer, batch_size=cfg.batch_size
        )
        wandb.log({"loss": loss, "step": step}) if wandb.run else None

        # Eval
        if step and step % cfg.eval_every == 0:
            print("-" * 10, " Eval ", "-" * 10)
            model.eval()
            test_res = test_agent(env, model, UniformAgent())
            test_res.update({"step": step})
            print(test_res)
            wandb.log(test_res) if wandb.run else None
            model.train()

            # Save the checkpoint
            last_ckpt_path = os.path.join(run_dir, f"ckpt-{step:08}.pt")
            ckpt = {"env": env, "model": model}
            torch.save(ckpt, last_ckpt_path)
            print(f"Saved checkpoint at {last_ckpt_path}")

        # Sample new trajectories
        if step and step % cfg.eval_every == 0:
            print("-" * 10, " Regen ", "-" * 10)
            model.eval()
            buffer = gen_batch_traj_buffer(
                buffer=buffer,
                env=env,
                afn=model,
                num_trajectories=cfg.num_regen_traj,
                batch_size=cfg.buffer_batch_size,
            )
            model.train()

    return model
