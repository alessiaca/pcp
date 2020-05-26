import numpy as np
from typing import Optional, Tuple, List, Union
from agents.common import PlayerAction, BoardPiece, SavedState, apply_player_action, check_end_state,\
    GameState, CONNECT_N, PLAYER1, PLAYER2


def eval_board(board: np.ndarray, players: List[BoardPiece]) -> float:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param players: List of players with maximizer first
    :return: Evaluation of the board
    """

    def eval_board_part(board_part: np.ndarray, players_eval: List[BoardPiece]) -> float:
        """
        :param board_part: 4 adjacent board pieces
        :param players_eval: List of players with maximizer first
        :return: Value of the board_part
        """
        # Count the occurrences of the players in CONNECT_N adjacent cells, where a win could potentially occur
        player_check_n = np.count_nonzero(board_part == players_eval[0])
        player_riv_n = np.count_nonzero(board_part == players_eval[1])
        if player_riv_n == 0:
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


def minimax(board: np.ndarray, alpha: int, beta: int, players: List[BoardPiece], depth: int, MaxPlayer: bool) \
        -> Tuple[any, Union[PlayerAction, None]]:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param alpha: the best value that maximizer can guarantee in the current state or before in the maximizer turn
    :param beta: the best value that minimizer can guarantee in the current state or before it in the minimizer turn
    :param players: List of players with maximizer first
    :param depth: Steps again that should be evaluated
    :param MaxPlayer: Bool if maximizers turn
    :return: Best value for maximizer or minimizer and the corresponding action
    """
    # Check endstate of the game after last players move
    end_state = check_end_state(board, players[0] if not MaxPlayer else players[1])
    # Return maximally positive/negative value if the move of the last player won the game
    if end_state == GameState.IS_WIN:
        if MaxPlayer:
            return -np.infty, None
        else:
            return np.infty, None
    if end_state == GameState.IS_DRAW:
        return 0, None
    # Only evaluate the board if the game is still on and the bottom of the tree is reached
    if end_state == GameState.STILL_PLAYING and depth == 0:
        return eval_board(board, players), None

    if MaxPlayer:
        best_value = -np.inf
        player = players[0]
    else:
        best_value = np.inf
        player = players[1]

    free_columns = [i for i, column in enumerate(board.T) if 0 in column]
    np.random.shuffle(free_columns)
    for action in free_columns:
        board_new = apply_player_action(board.copy(), PlayerAction(action), player)
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
        -> Tuple[PlayerAction, SavedState]:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID
    :param saved_state:
    :return: Column in which player wants to make his move (chosen using the minimax algorithm)
    """
    # If the minimax agent can do the first move, make sure it is always in the middle
    if not board.any():
        return PlayerAction(3), SavedState()

    # Create a list that holds the player first, and the opponent second
    players = [PLAYER1, PLAYER2]
    players.remove(player)
    ordered_players = [player] + players

    _, action = minimax(board, -np.inf, np.inf, ordered_players, 5, True)
    return PlayerAction(action), SavedState()
