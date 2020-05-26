import numpy as np
from typing import Optional, Tuple
from agents.common import PlayerAction, BoardPiece, SavedState, apply_player_action,connect_four, CONNECT_N


def eval_board(board: np.ndarray, players: BoardPiece) -> int:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param players: List of players with maximizer first
    :return: Evaluation of the board
    """
    # Return a infinitely negative evaluation for a board in which the rival player won - regardless
    # of whether the max player could theoretically also connect N in his next move
    if connect_four(board, players[1]):
        return -np.infty

    def eval_board_part(board_part: np.ndarray, players_eval: BoardPiece) -> int:
        """
        :param board_part: 4 adjacent board pieces
        :param players_eval: List of players with maximizer first
        :return: Value of the board_part
        """
        # Count the occurrences of the players in CONNECT_N adjacent cells, where a win could potentially occur
        player_check_n = np.count_nonzero(row_part == players_eval[0])
        player_riv_n = np.count_nonzero(row_part == players_eval[1])
        if player_check_n == 4:
            return np.infty
        elif player_riv_n == 0:
            return player_check_n ** 2
        else:
            return 0

    rows, cols = board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1
    players_rev = players[::-1]  # Players with minimizing player first
    value_max_player = 0
    value_min_player = 0
    for i in range(rows):
        for j in range(cols_edge):
            row_part = board[i, j:j + CONNECT_N]
            value_max_player += eval_board_part(row_part, players)
            value_min_player += eval_board_part(row_part, players_rev)
    for i in range(rows_edge):
        for j in range(cols):
            col_part = board[i:i + CONNECT_N, j]
            value_max_player += eval_board_part(col_part, players)
            value_min_player += eval_board_part(col_part, players_rev)
    for i in range(rows_edge):
        for j in range(cols_edge):
            block = board[i:i + CONNECT_N, j:j + CONNECT_N]
            value_max_player += eval_board_part(np.diag(block), players)
            value_min_player += eval_board_part(np.diag(block), players_rev)
            value_max_player += eval_board_part(np.diag(block[::-1, :]), players)
            value_min_player += eval_board_part(np.diag(block[::-1, :]), players_rev)

    return value_max_player - value_min_player


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
    # If the minimax agent can do the first move, make sure it is always in the middle
    if not board.any():
        return 3, None

    # Create a list that holds the player first, and the opponent second
    from agents.common import PLAYER1, PLAYER2
    players = [PLAYER1, PLAYER2]
    players.remove(player)
    ordered_players = [player] + players

    value, action = minimax(board, -np.inf, np.inf, ordered_players, 4, True)
    return action, None
