import numpy as np
from typing import Optional, Tuple
from agents.common import PlayerAction, BoardPiece, SavedState, NO_PLAYER


def generate_move_random(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) \
        -> Tuple[PlayerAction, int]:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID
    :param saved_state:
    :return: Column in which player wants to make his move (chosen randomly)
    """
    # Get column indexes where there is no player and choose one empty column randomly
    action = np.random.choice(np.unique(np.where(board == NO_PLAYER)[1]))
    return action, saved_state