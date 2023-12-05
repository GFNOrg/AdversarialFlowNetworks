import torch
import numpy as np
from torch import nn

from src.envs.base_env import BaseEnv, Player

DEVICE = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")


""" AlphaZero Architecture """


class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels=256):
        super(ConvBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(out_channels, affine=False),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        return self.block(x)


class ResBlock(nn.Module):
    def __init__(self, num_filters=256):
        super(ResBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_filters, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Conv2d(num_filters, num_filters, kernel_size=3, stride=1, padding=1),
            # nn.BatchNorm2d(num_filters, affine=False),
        )
        self.LeakyReLU = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        out = self.block(x)
        out += x
        return self.LeakyReLU(out)


class PolicyBlock(nn.Module):
    def __init__(self, state_size, action_dim, in_channels=256, num_filters=32):
        super(PolicyBlock, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=1, stride=1),
            # nn.BatchNorm2d(num_filters, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(state_size * num_filters, action_dim),
        )

    def forward(self, x):
        return self.block(x)


class ValueBlock(nn.Module):
    def __init__(self, state_size, in_channels=256, num_filters=1, num_hidden=256):
        super(ValueBlock, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(in_channels, num_filters, kernel_size=1, stride=1, padding=0),
            # nn.BatchNorm2d(num_filters, affine=False),
            nn.LeakyReLU(inplace=True),
            nn.Flatten(),
            nn.Linear(state_size * num_filters, num_hidden),
            nn.LeakyReLU(inplace=True),
            nn.Linear(num_hidden, 1),
        )

    def forward(self, x):
        return self.block(x)


class AZResNet(nn.Module):
    def __init__(
        self,
        state_dims: tuple[int, ...],
        action_dims: int,
        res_filters=128,
        res_layers=10,
        head_filters=32,
    ):
        super(AZResNet, self).__init__()

        self.conv = ConvBlock(state_dims[0] + 1, res_filters)
        self.res_layers = nn.Sequential(*[ResBlock(res_filters) for _ in range(res_layers)])

        self.policy_head_0 = PolicyBlock(np.prod(state_dims[1:]), action_dims, res_filters, head_filters)
        self.value_head_0 = ValueBlock(np.prod(state_dims[1:]), res_filters, head_filters)

        self.policy_head_1 = PolicyBlock(np.prod(state_dims[1:]), action_dims, res_filters, head_filters)
        self.value_head_1 = ValueBlock(np.prod(state_dims[1:]), res_filters, head_filters)
        self.log_Z_0 = torch.zeros(1, requires_grad=True, device=DEVICE)
        self.log_Z_1 = torch.zeros(1, requires_grad=True, device=DEVICE)

        self.to(DEVICE)

    def forward(self, x: torch.tensor, side: Player):
        batch_size, _, w, h = x.shape
        sides = torch.ones((batch_size, 1, w, h), dtype=torch.float32, device=DEVICE)

        x = torch.cat((x, sides), dim=1)
        x = self.conv(x)
        x = self.res_layers(x)

        if side == 0:
            return self.value_head_0(x), self.policy_head_0(x)
        else:
            return self.value_head_1(x), self.policy_head_1(x)

    def sample_actions(self, env: BaseEnv, side: Player):
        states = torch.from_numpy(env.obs()).unsqueeze(0).to(DEVICE)
        masks = torch.from_numpy(env.get_masks()).float().to(DEVICE)

        masks = masks.unsqueeze(0)
        with torch.no_grad():
            _, policy = self(states, side)
            return (masks * policy + (1 - masks) * -(1e9)).argmax(dim=1).unsqueeze(1)
