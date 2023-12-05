from __future__ import annotations
import numpy as np
from abc import ABC, abstractmethod
from enum import IntEnum, Enum


class Outcome(Enum):
    WIN_P1 = 0
    WIN_P2 = 1
    DRAW = 2
    NOT_DONE = 3


class Player(IntEnum):
    ONE = 0
    TWO = 1

    def switch(self) -> Player:
        return Player.ONE if self == Player.TWO else Player.TWO

    def clone(self) -> Player:
        return Player.ONE if self == Player.ONE else Player.TWO


class BaseEnv(ABC):
    # State vars
    board: np.ndarray
    turns: int
    curr_player: Player
    done: bool
    outcome: Outcome

    # NEED TO BE SET BY EACH ENV
    name: str
    BOARD_SHAPE: tuple[int, ...]
    NUM_EXTRA_INFO: int
    ACTION_DIM: int
    MAX_TRAJ_LEN: int  # Number of possible turns + 1

    def __init__(
        self,
        initial_board: np.ndarray | None = None,
        use_conv: bool = True,
        lambd: float = 2.0,
    ) -> None:
        # Get user specified vars
        self.initial_board = initial_board
        self.use_conv = use_conv
        self.lambd = lambd

        # Get env shapes/dims
        self.BOARD_DIM = np.prod(self.BOARD_SHAPE).astype(int)
        self.FLAT_STATE_DIM = self.BOARD_DIM + self.NUM_EXTRA_INFO

        self.CONV_SHAPE = (1, *self.BOARD_SHAPE[1:])
        self.CONV_STATE_DIM = (
            self.BOARD_SHAPE[0] + self.NUM_EXTRA_INFO,
            *self.BOARD_SHAPE[1:],
        )

        if self.use_conv:
            self.STATE_DIM = self.CONV_STATE_DIM
        else:
            self.STATE_DIM = self.FLAT_STATE_DIM

    @abstractmethod
    def reset(self) -> np.ndarray:
        ...

    @abstractmethod
    def place_piece(self, action: int) -> None:
        ...

    def evaluate_outcome(self) -> Outcome:
        ...

    def step(self, action: int) -> bool:
        self.place_piece(action)

        self.outcome = self.evaluate_outcome()
        if self.outcome in [Outcome.WIN_P1, Outcome.WIN_P2, Outcome.DRAW]:
            self.done = True
        else:
            self.turns += 1
            self.curr_player = self.curr_player.switch()

        return self.done

    @abstractmethod
    def get_masks(self) -> np.ndarray:
        ...

    def get_curr_player(self) -> Player:
        return self.curr_player

    def get_log_reward(self) -> tuple[float, float]:
        assert self.done

        if self.outcome == Outcome.DRAW:
            return (0, 0)

        if self.outcome == Outcome.WIN_P1:
            return (self.lambd, -self.lambd)

        return (-self.lambd, self.lambd)

    @abstractmethod
    def get_extra_info(self) -> np.ndarray:
        ...

    def conv_obs(self) -> np.ndarray:
        extra_info = self.get_extra_info()
        obs = np.concatenate(
            [self.board] + [np.ones(self.CONV_SHAPE) * info for info in extra_info],
            axis=0,
        ).astype(np.float32)

        return obs

    def flat_obs(self) -> np.ndarray:
        obs = np.empty(self.FLAT_STATE_DIM, dtype=np.float32)
        obs[: self.BOARD_DIM] = self.board.flatten()

        extra_info = self.get_extra_info()
        for i, info in enumerate(extra_info):
            obs[self.BOARD_DIM + i] = info

        return obs

    def obs(self) -> np.ndarray:
        return self.conv_obs() if self.use_conv else self.flat_obs()

    @abstractmethod
    def render(self) -> None:
        ...
