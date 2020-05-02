import numpy as np
from typing import Optional, Tuple


PlayerAction = np.int8
BoardPiece = np.int8
SavedState = np.int8


def generate_move_random(
    board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]
) -> Tuple[PlayerAction, int]:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID
    :param saved_state:
    :return: Column in which player wants to make his move (chosen randomly)
    """

    free_columns = []
    for i, column in enumerate(board.T):
        if 0 in set(column):
            free_columns.append(i)
    if not free_columns:
        return "All states are occupied - Game ends without winner"
    else:
        action = np.random.choice(free_columns)
        saved_state = 0
        return action, saved_state