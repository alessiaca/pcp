import numpy as np
from typing import Optional, Callable, Tuple
from enum import Enum
from numba import njit

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece
CONNECT_N = 4  # Number of connected board pieces needed for a win
PlayerAction = np.int8  # The column to be played


# Class indicating whether the game is still going on, ended in a draw (full board) or one of the players won
class GameState(Enum):
    IS_WIN = 1
    IS_DRAW = -1
    STILL_PLAYING = 0


class SavedState:
    pass


GenMove = Callable[
    [np.ndarray, BoardPiece, Optional[SavedState]],  # Arguments for the generate_move function
    Tuple[PlayerAction, Optional[SavedState]]  # Return type of the generate_move function
]


def initialize_game_state() -> np.ndarray:
    """
    :return: initial board state,  6 x 7 array of zeros
    """
    return np.zeros((6, 7), dtype=BoardPiece)


def pretty_print_board(board: np.ndarray) -> str:
    """
    :param board: State of board , 6 x 7 array
    :return: String which shows the state of the board in a human readable way
    """
    states = [' ', 'X', 'O']
    pp_board = '| ============= |\n'
    for row in board:
        states_row = [states[int(i)] for i in row]
        pp_row = ' '.join(states_row)
        pp_board += '| ' + pp_row + ' |\n'
    pp_board += ('| ============= |\n'
                 '| 0 1 2 3 4 5 6 |\n')
    return pp_board


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece) -> np.ndarray:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param action: Column where player should be dropped
    :param player: player ID for which action should be applied [1, 2]
    :return: New state of board after action of the player was applied
    """
    max_free_ind = np.max(np.where(board[:, action] == NO_PLAYER))
    board[max_free_ind, action] = player
    return board


@njit()
def connect_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None) \
        -> bool:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID for which victory should be checked
    :param last_action: last column where player was dropped
    :return: Decision on whether the player won (whether he has N adjacent pieces on the board)
    """
    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1
    for i in range(rows):
        for j in range(cols_edge):
            if np.all(board[i, j:j+CONNECT_N] == player):
                return True
    for i in range(rows_edge):
        for j in range(cols):
            if np.all(board[i:i+CONNECT_N, j] == player):
                return True
    for i in range(rows_edge):
        for j in range(cols_edge):
            block = board[i:i+CONNECT_N, j:j+CONNECT_N]
            if np.all(np.diag(block) == player):
                return True
            if np.all(np.diag(block[::-1, :]) == player):
                return True
    return False


def check_end_state(
    board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None,
) -> GameState:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID for which GameState should be checked
    :param last_action: last column where player was dropped
    :return: State of Game: Either the player won, the game is drawn or is still going on
    """

    if connect_four(board, player):
        return GameState.IS_WIN
    else:
        if 0 in board:
            return GameState.STILL_PLAYING
        else:
            return GameState.IS_DRAW


