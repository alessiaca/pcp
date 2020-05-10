import numpy as np
from typing import Optional, Tuple
from connectn.common import PlayerAction, BoardPiece, SavedState, apply_player_action


# TODO: Implement a better evaluation of the board state: Look at the literature of connect4 heuristics, Mixture of hard
#  rules and heuristic !?
def eval_board(board: np.ndarray, players: BoardPiece) -> int:

    def compute_max_adjacent_players(board: np.ndarray, player: BoardPiece) -> int:

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

    counter_1 = compute_max_adjacent_players(board, players[0])
    counter_2 = compute_max_adjacent_players(board, players[1])
    if counter_2 == 3:
        return -np.inf
    return counter_1


def minimax(board: np.ndarray, alpha: int, beta: int, players: list, depth: int, MaxPlayer: bool) \
        -> Tuple[PlayerAction, np.ndarray]:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param alpha: the best value that maximizer can guarantee in the current state or before in the maximizer turn
    :param beta: the best value that minimizer can guarantee in the current state or before it in the minimizer turn
    :param players: List of players with maximizer first
    :param depth: Steps again that should be evaluated
    :param MaxPlayer: Bool if maximizers turn
    :return: Best value for maximizer or minimizer and the corresponding action
    """

    if depth == 0:
        return eval_board(board, players), 1

    if MaxPlayer:
        best_value = -np.inf
        player = players[0]
    else:
        best_value = np.inf
        player = players[1]

    free_columns = [i for i, column in enumerate(board.T) if 0 in column]
    np.random.shuffle(free_columns)
    for action in free_columns:
        board_new = apply_player_action(board.copy(), action, player)
        value, _ = minimax(board_new, alpha, beta, players, depth - 1, not MaxPlayer)

        if MaxPlayer and value >= best_value:
            best_value = value
            best_action = action
            alpha = max(alpha, best_value)
            if beta <= alpha:
                break
        if not MaxPlayer and value <= best_value:
            best_value = value
            best_action = action
            beta = min(beta, best_value)
            if beta <= alpha:
                break

    return best_value, best_action


def generate_move_minimax(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) \
        -> Tuple[PlayerAction, int]:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID
    :param saved_state:
    :return: Column in which player wants to make his move (chosen using the minimax algorithm)
    """
    # Create a list that holds the player first, and the opponent second
    from connectn.common import PLAYER1, PLAYER2
    players = [PLAYER1, PLAYER2]
    players.remove(player)
    ordered_players = [player] + players

    value, action = minimax(board, -np.inf, np.inf, ordered_players, 2, True)
    return action, saved_state
