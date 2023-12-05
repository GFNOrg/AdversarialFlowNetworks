import numpy as np
from src.envs.base_env import BaseEnv, Outcome, Player


def create_ttt_win_masks() -> np.ndarray:
    row_wins = np.zeros((3, 3, 3))
    for row in range(3):
        row_wins[row, row, :] = 1

    col_wins = np.zeros((3, 3, 3))
    for col in range(3):
        col_wins[col, :, col] = 1

    diagonal_wins = np.zeros((2, 3, 3))
    diagonal_wins[0] = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    diagonal_wins[1] = np.array([[0, 0, 1], [0, 1, 0], [1, 0, 0]])

    win_masks = np.concatenate((row_wins, col_wins, diagonal_wins), axis=0)

    return win_masks


Xs = [[1, 0, 1], [0, 1, 0], [0, 0, 0]]
Os = [[0, 0, 0], [0, 0, 0], [1, 0, 1]]
SMALL_INITIAL_STATE = np.array([Xs, Os], dtype=np.float32)


class TicTacToeEnv(BaseEnv):
    def __init__(self, initial_board=None, use_conv=True, lambd=2.0) -> None:
        self.name = "TicTacToe"
        self.BOARD_SHAPE = (2, 3, 3)
        self.NUM_EXTRA_INFO = 2
        self.ACTION_DIM = 9
        self.MAX_TRAJ_LEN = 9 + 1
        self.WIN_MASKS = create_ttt_win_masks()

        super().__init__(initial_board, use_conv, lambd)
        self.reset()

    def reset(self) -> None:
        if self.initial_board is not None:
            self.board = self.initial_board.copy()
            self.turns = self.board.sum()
            self.curr_player = Player.ONE if self.turns % 2 == 0 else Player.TWO
        else:
            self.board = np.zeros(self.BOARD_SHAPE, dtype=np.float32)
            self.turns = 0
            self.curr_player = Player.ONE

        self.done = False
        self.outcome = Outcome.NOT_DONE

    def get_extra_info(self) -> np.ndarray:
        return [int(self.curr_player), int(self.turns)]

    def place_piece(self, action: int) -> None:
        row = action // 3
        col = action % 3

        assert ~np.any(
            self.board[:, row, col]
        ), f"Cannot override a placed piece at ({row}, {col})"

        # Update the internal state
        self.board[self.curr_player, row, col] = 1

        return

    def evaluate_outcome(self):
        if np.any(
            (self.board[self.curr_player] * self.WIN_MASKS).sum(axis=(1, 2)) == 3
        ):
            return Outcome(self.curr_player)
        elif self.turns == 8:
            return Outcome.DRAW
        else:
            return Outcome.NOT_DONE

    def get_masks(self) -> np.ndarray:
        return self.board.sum(axis=0).flatten().astype(bool) == False

    def render(self) -> None:
        board = self.board
        for row in range(3):
            print("|", end=" ")
            for col in range(3):
                if board[0, row, col] == 1:
                    print("X", end=" | ")
                elif board[1, row, col] == 1:
                    print("O", end=" | ")
                else:
                    print(" ", end=" | ")
            print()
        print()


def test():
    env = TicTacToeEnv()

    for i in range(9):
        if not env.done:
            print(env.obs().shape)
            env.step(i)
            env.render()

    print(env.outcome)


if __name__ == "__main__":
    test()
