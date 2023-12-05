import torch
import copy
import numpy as np
from tqdm import tqdm

from src.envs.base_env import BaseEnv

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


class Trajectory:
    ATTR_ORDER = [
        "states",
        "masks",
        "curr_players",
        "actions",
        "dones",
        "log_reward",
    ]

    def __init__(self, env: BaseEnv) -> None:
        self.states = torch.zeros((env.MAX_TRAJ_LEN, *env.STATE_DIM))
        self.masks = torch.zeros((env.MAX_TRAJ_LEN, env.ACTION_DIM))
        self.curr_players = torch.zeros((env.MAX_TRAJ_LEN, 1))
        self.actions = torch.zeros((env.MAX_TRAJ_LEN, 1))
        self.dones = torch.zeros((env.MAX_TRAJ_LEN, 1))  # 0 for N/A
        self.log_reward = torch.zeros(2)

        self.length = 0

    def add_step(self, env: BaseEnv, action):
        self.states[self.length] = torch.from_numpy(env.obs())
        self.masks[self.length] = torch.from_numpy(env.get_masks())
        self.curr_players[self.length] = env.get_curr_player()
        self.actions[self.length] = action
        self.dones[self.length] = 1  # 1 for not done

        if env.done:
            self.log_reward = torch.tensor(env.get_log_reward())
            self.dones[self.length] = 2  # 2 for just terminated

        self.length += 1

    def __len__(self):
        return self.length


class TrajectoryBuffer:
    def __init__(self, max_capacity: int, env: BaseEnv) -> None:
        sample_traj = Trajectory(env)

        for attr in sample_traj.ATTR_ORDER:
            val = getattr(sample_traj, attr)
            self.__dict__[attr] = torch.zeros((max_capacity, *val.shape))

        self.max_capacity = max_capacity
        self.size = 0

    def add_traj(self, traj: Trajectory):
        index = self.size % self.max_capacity
        for attr in traj.ATTR_ORDER:
            self.__dict__[attr][index] = getattr(traj, attr)

        self.size += 1

    def sample(self, sample_size):
        idx = np.random.choice(min(self.max_capacity, self.size), sample_size)
        return (getattr(self, attr)[idx].to(DEVICE) for attr in Trajectory.ATTR_ORDER)


def gen_traj(env: BaseEnv):
    traj = Trajectory(env)

    while not env.done:
        masks = env.get_masks()
        action = torch.from_numpy(masks).float().multinomial(1).item()
        traj.add_step(env, action)
        env.step(action)

    # Add reward transition
    traj.add_step(env, 0)

    env.reset()
    return traj


def gen_batch_traj_buffer(
    buffer: TrajectoryBuffer,
    env: BaseEnv,
    afn: torch.nn.Module,
    num_trajectories: int,
    batch_size: int,
    # sampling: str = "",
):
    def gen_batch():
        envs = [copy.deepcopy(env) for _ in range(batch_size)]
        trajs = [Trajectory(curr_env) for curr_env in envs]

        for t in range(env.MAX_TRAJ_LEN):
            curr_player = t % 2
            batch_obs, batch_masks = [], []
            for curr_env in envs:
                batch_obs.append(torch.from_numpy(curr_env.obs()))
                batch_masks.append(torch.from_numpy(curr_env.get_masks()))

            batch_obs = torch.stack(batch_obs, dim=0).to(DEVICE)
            batch_masks = torch.stack(batch_masks, dim=0).float().to(DEVICE)

            # TODO: Support UniformAgent to speed up initial data collection
            with torch.no_grad():
                _, logits = afn(batch_obs, curr_player)
                masked_logits = batch_masks * logits + (1 - batch_masks) * (-1e9)
                masked_probs = torch.nn.functional.softmax(masked_logits / 2, dim=1)
                actions = masked_probs.multinomial(1)

            for action, curr_env, curr_traj in zip(actions, envs, trajs):
                if not curr_env.done:
                    curr_traj.add_step(curr_env, action)
                    curr_env.step(action)

        for curr_env, curr_traj in zip(envs, trajs):
            curr_traj.add_step(curr_env, 0)

        return trajs

    n_batches = int(num_trajectories / batch_size + 1)
    for _ in tqdm(range(n_batches)):
        batch_traj = gen_batch()
        for traj in batch_traj:
            buffer.add_traj(traj)

    env.reset()
    return buffer
