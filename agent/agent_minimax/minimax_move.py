import numpy as np
from typing import Optional, Tuple
#from ..connectn.common import apply_player_action # Fix import and use functions from common.py


PlayerAction = np.int8
BoardPiece = np.int8
SavedState = np.int8


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
    max_free_ind = np.max(np.where(board[:, action] == 0))
    board[max_free_ind, action] = player
    return board


def eval_board(board: np.ndarray, player: BoardPiece) -> int:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID for which victory should be checked
    :param last_action: last column where player was dropped
    :return: Max number of adjacent pieces of the player
    """

    max_counter = - np.inf

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
            max_counter = max(counter, max_counter)

        # Check if there are 4 adjacent players in a diagonal
        n_rows = board.shape[0]
        for board_tmp in [board, np.flip(board)]:
            for row_i in range(n_rows):
                counter = 0
                for j, i in enumerate(range(row_i, n_rows-1)):
                    counter = compare_states_and_count(board[i, j], board[i+1, j+1], player, counter)
                max_counter = max(counter, max_counter)

    return max_counter


def minimax(board: np.ndarray, player: BoardPiece, depth: int, MaxPlayer: bool) \
        -> Tuple[PlayerAction, np.ndarray]:
    if depth == 0:
        return 0, eval_board(board, player)
    if MaxPlayer:
        max_counter = - np.inf
        for action, column in enumerate(board):
            if 0 in column:
                board = apply_player_action(board, action, player)
                action, counter = minimax(board, player, depth - 1, False)
                if counter >= max_counter:
                    max_counter = counter
                    max_action = action
            return max_counter, max_action
    else:
        min_counter = np.inf
        for action, column in enumerate(board):
            if 0 in column:
                board = apply_player_action(board, action, player)
                action, counter = minimax(board, player, depth - 1, True)
                if counter <= min_counter:
                    min_counter = counter
                    min_action = action
            return min_counter, min_action


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) \
        -> Tuple[PlayerAction, int]:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID
    :param saved_state:
    :return: Column in which player wants to make his move (chosen using the minimax algorithm)
    """

    action, counter = minimax(board, player, 4, True)
    return action, saved_state

print(minimax(np.zeros((6,7)), BoardPiece(1), 2, True))