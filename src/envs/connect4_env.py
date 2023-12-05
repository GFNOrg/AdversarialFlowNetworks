import numpy as np
from src.envs.base_env import BaseEnv, Player, Outcome


def create_win_masks(num_rows: int, num_cols: int) -> np.ndarray:
    # Horizontal wins
    horizontal_wins = []
    for row in range(num_rows):
        for col in range(num_cols - 3):
            curr_win = np.zeros((num_rows, num_cols))
            curr_win[row, col : col + 4] = 1
            horizontal_wins.append(curr_win)

    # Vertical wins
    vertical_wins = []
    for row in range(num_rows - 3):
        for col in range(num_cols):
            curr_win = np.zeros((num_rows, num_cols))
            curr_win[row : row + 4, col] = 1
            vertical_wins.append(curr_win)

    # Diagonal wins
    diagonal_wins = []
    for row in range(num_rows - 3):
        for col in range(num_cols - 3):
            # Top left to bottom right
            curr_win = np.zeros((num_rows, num_cols))
            curr_win[np.arange(row, row + 4), np.arange(col, col + 4)] = 1
            diagonal_wins.append(curr_win)

            # Top right to bottom left
            curr_win = np.zeros((num_rows, num_cols))
            curr_win[np.arange(row, row + 4)[::-1], np.arange(col, col + 4)] = 1
            diagonal_wins.append(curr_win)

    wins = horizontal_wins + vertical_wins + diagonal_wins
    wins = np.concatenate([np.expand_dims(win, axis=0) for win in wins], axis=0)
    return wins


class Connect4Env(BaseEnv):
    def __init__(
        self,
        num_rows: int = 6,
        num_cols: int = 7,
        initial_board=None,
        use_conv=True,
        lambd=2,
    ) -> None:
        self.name = "Connect4"
        self.num_rows = 6
        self.num_cols = 7

        self.BOARD_SHAPE = (2, num_rows, num_cols)
        self.NUM_EXTRA_INFO = 2
        self.ACTION_DIM = num_cols
        self.MAX_TRAJ_LEN = num_rows * num_cols + 1
        self.WIN_MASKS = create_win_masks(num_rows, num_cols)

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

    def get_extra_info(self):
        return [
            int(self.curr_player),
            int(self.turns),
        ]

    def place_piece(self, action: int) -> None:
        num_pieces_column = int(self.board.sum(axis=(0, 1))[action])
        row = (
            self.num_rows - num_pieces_column - 1
        )  # E.g. if there are 6 rows and 2 pieces already, place at row with index 3
        col = action
        assert ~np.any(
            self.board[:, row, col]
        ), f"Cannot override a placed piece at ({row}, {col})"

        # Update the internal state
        self.board[self.curr_player, row, col] = 1

    def won(self, player: Player):
        x = (self.WIN_MASKS * self.board[player]).sum(axis=(1, 2))
        x = x.max()
        return x == 4

    def evaluate_outcome(self):
        if self.won(Player.ONE):
            return Outcome.WIN_P1
        elif self.won(Player.TWO):
            return Outcome.WIN_P2
        elif self.turns + 1 >= self.num_rows * self.num_cols:
            return Outcome.DRAW
        else:
            return Outcome.NOT_DONE

    def get_masks(self) -> np.ndarray:
        return self.board.sum(axis=(0, 1)) < self.num_rows

    def render(self) -> None:
        board = self.board[0] - self.board[1]

        print()

        for row in range(board.shape[0]):
            for col in range(board.shape[1]):
                if board[row, col] == 1:
                    print("X", end=" ")
                elif board[row, col] == -1:
                    print("O", end=" ")
                else:
                    print("-", end=" ")
            print()

        # Print column numbers
        for col in range(board.shape[1]):
            print(col, end=" ")

        print()
        print()


def test():
    env = Connect4Env()

    for i in range(25):
        if not env.done:
            print(env.obs().shape)
            env.step(i % 7)
            env.render()

    print(env.outcome)


if __name__ == "__main__":
    test()
