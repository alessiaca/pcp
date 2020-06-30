import numpy as np
from typing import Optional, Callable
from agents.common import PlayerAction, BoardPiece, SavedState, GenMove, apply_player_action
from agents.agent_random import random_move
from agents.agent_minimax import minimax_move
from agents.agent_MCTS import MCTS_move


def user_move(board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState]):
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param _player: Player ID of the user
    :param saved_state: not used this implementation of the user move generation
    :return: Column the user wants to drop his player
    """
    action = PlayerAction(-1)
    move_worked = None
    # Make sure that a column is selected which is in the range of the board and is not already full
    while not 0 <= action < board.shape[1] or move_worked is None:
        try:
            action = PlayerAction(input("Column? "))
            move_worked = apply_player_action(board, action, _player)
        except:
            pass
    return action, SavedState()


def human_vs_agent(
        generate_move_1: GenMove = MCTS_move,
        generate_move_2: GenMove = user_move,
        player_1: str = "Player 1",
        player_2: str = "Player 2",
        args_1: tuple = (),
        args_2: tuple = (),
        init_1: Callable = lambda board, player: None,
        init_2: Callable = lambda board, player: None,
):
    """
    :param generate_move_1: Function which is used for the move generation of player 1
    (either random_move, minimax_move or user_move)
    :param generate_move_2: Function which is used for the move generation of player 2
    :param player_1: Name of player 1
    :param player_2: Name of player 2
    :param args_1: Additional parameters for player 1
    :param args_2: Additional parameters for player 2
    :param init_1: /
    :param init_2: /
    :return: No return type, the function controls the CONNECT_N (here N=4) game between two players
    """
    import time
    from agents.common import PLAYER1, PLAYER2, GameState
    from agents.common import initialize_game_state, pretty_print_board, apply_player_action, check_end_state

    players = (PLAYER1, PLAYER2)
    # Two rounds in which the player that makes the first move is switched
    for play_first in (1, -1):
        for init, player in zip((init_1, init_2)[::play_first], players):
            init(initialize_game_state(), player)

        saved_state = {PLAYER1: None, PLAYER2: None}
        board = initialize_game_state()
        gen_moves = (generate_move_1, generate_move_2)[::play_first]
        player_names = (player_1, player_2)[::play_first]
        gen_args = (args_1, args_2)[::play_first]

        playing = True
        while playing:
            for player, player_name, gen_move, args in zip(
                    players, player_names, gen_moves, gen_args,
            ):
                t0 = time.time()
                print(pretty_print_board(board))
                print(
                    f'{player_name} you are playing with {"X" if player == PLAYER1 else "O"}'
                )
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], *args
                )
                print(f"Move time: {time.time() - t0:.3f}s")
                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board))
                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw")
                    else:
                        print(
                            f'{player_name} won playing {"X" if player == PLAYER1 else "O"}'
                        )

                    playing = False
                    break


if __name__ == "__main__":
    human_vs_agent()
