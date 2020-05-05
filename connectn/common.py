import numpy as np
from typing import Optional, Callable, Tuple
from enum import Enum

BoardPiece = np.int8  # The data type (dtype) of the board
NO_PLAYER = BoardPiece(0)  # board[i, j] == NO_PLAYER where the position is empty
PLAYER1 = BoardPiece(1)  # board[i, j] == PLAYER1 where player 1 has a piece
PLAYER2 = BoardPiece(2)  # board[i, j] == PLAYER2 where player 2 has a piece

PlayerAction = np.int8  # The column to be played


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
    :return: String which shows the state of the board in a nicer way
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


def string_to_board(pp_board: str) -> np.ndarray:
    return np.ones((6, 7), dtype=BoardPiece)
# Not necessary yet


def apply_player_action(
        board: np.ndarray, action: PlayerAction, player: BoardPiece, copy: bool = False) \
        -> np.ndarray:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param action: Column where player should be dropped
    :param player: player ID [1, 2]
    :param copy:
    :return: New state of board after player was dropped in column
    """
    max_free_ind = np.max(np.where(board[:, action] == NO_PLAYER))
    board[max_free_ind, action] = player
    return board


def connect_four(
        board: np.ndarray, player: BoardPiece, last_action: Optional[PlayerAction] = None) \
        -> bool:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID for which victory should be checked
    :param last_action: last column where player was dropped
    :return: Decision on whether the player won
    """
    # Compare two states and update the counter accordingly
    def compare_states_and_count(state_1: BoardPiece, state_2: BoardPiece, player_func: BoardPiece, counter_func: int):
        if int(state_1) == player_func & int(state_2) == player_func:
            counter_func += 1
        else:
            counter_func = 0
        return counter_func

    # Check if there are 4 adjacent players in either rows or columns of board
    for board_tmp in [board, board.T]:
        for row in board_tmp:
            counter = 0
            for i in range(len(row)-1):
                counter = compare_states_and_count(row[i], row[i+1], player, counter)
                if counter == 3:
                    return True

        # Check if there are 4 adjacent players in a diagonal
        n_rows = board.shape[0]
        for board_tmp in [board, np.flip(board)]:
            for row_i in range(n_rows):
                counter = 0
                for j, i in enumerate(range(row_i, n_rows-1)):
                    counter = compare_states_and_count(board[i, j], board[i+1, j+1], player, counter)
                    if counter == 3:
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


