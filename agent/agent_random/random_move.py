import numpy as np
import time
from typing import Optional, Tuple


PlayerAction = np.int8
BoardPiece = np.int8
SavedState = np.int8


def generate_move_random(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) \
        -> Tuple[PlayerAction, int]:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID
    :param saved_state:
    :return: Column in which player wants to make his move (chosen randomly)
    """

    free_columns = []
    for i, column in enumerate(board.T):
        if 0 in column:
            free_columns.append(i)
    action = np.random.choice(free_columns)
    time.sleep(1)
    return action+1, saved_state