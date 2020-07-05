import numpy as np
from typing import Optional, Callable, Union
from agents.common import PlayerAction, BoardPiece, SavedState, GenMove, apply_player_action
from agents.agent_random import random_move
from agents.agent_minimax import minimax_move
from agents.agent_MCTS import MCTS_move
from matplotlib import pyplot as plt


def user_move(board: np.ndarray, _player: BoardPiece, saved_state: Optional[SavedState], args):
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param _player: Player ID of the user
    :param saved_state: not used this implementation of the user move generation
    :param args: Optional parameter
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


def play_one_round(
        generate_move_1: GenMove = MCTS_move,
        generate_move_2: GenMove = user_move,
        player_1: str = "Player 1",
        player_2: str = "Player 2",
        args_1: Union[int, float, None] = None,
        args_2: Union[int, float, None] = None,
        init_1: Callable = lambda board, player: None,
        init_2: Callable = lambda board, player: None,
        print_board = True
):
    """
    :param generate_move_1: Function which is used for the move generation of player 1
    (here either random_move, minimax_move or user_move)
    :param generate_move_2: Function which is used for the move generation of player 2
    :param player_1: Name of player 1
    :param player_2: Name of player 2
    :param args_1: Additional parameters for player 1 (e.g. depth of minimax or maximal MCTS time)
    :param args_2: Additional parameters for player 2
    :param init_1: /
    :param init_2: /
    :param print_board: True if board state should be printed to console, False otherwise
    :return: No return type, the function controls the CONNECT_N (here N=4) game between two players
    """
    import time
    from agents.common import PLAYER1, PLAYER2, GameState
    from agents.common import initialize_game_state, pretty_print_board, apply_player_action, check_end_state

    players = (PLAYER1, PLAYER2)
    result = np.zeros((2, 1))  # Save the number of wins per agent
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
                if print_board:
                    print(pretty_print_board(board))
                    print(
                    f'{player_name} you are playing with {"X" if player == PLAYER1 else "O"}'
                    )
                action, saved_state[player] = gen_move(
                    board.copy(), player, saved_state[player], args
                )
                print(f"Move time: {time.time() - t0:.3f}s") if print_board else None
                apply_player_action(board, action, player)
                end_state = check_end_state(board, player)
                if end_state != GameState.STILL_PLAYING:
                    print(pretty_print_board(board)) if print_board else None
                    if end_state == GameState.IS_DRAW:
                        print("Game ended in draw") if print_board else None
                    else:
                        print(
                            f'{player_name} won playing {"X" if player == PLAYER1 else "O"}'
                        ) if print_board else None
                        # Save which player won, the result array saves in the first row the number of wins
                        # of the first agent and in the second row the number of wins of the second agent
                        result[player_names.index(player_name)] += 1

                    playing = False
                    break
    return result


def agent_vs_agent_often(n_iterations: int, plot_res: bool):
    """
    Lets minimax and MCTS agent play against each other for a lot of rounds, tracks the winning of the two
    agents and creates a pie chart of the winning proportions
    :param n_iterations: Number of rounds the agents should play against each other
    :param plot_res: True if results (winning proportions) should be plotted, False if not wanted
    """
    # Change here the variations of MCTS time and minimax depth (no alpha-beta pruning used - caution
    # when increasing the search depth)
    MCTS_time = 5
    minimax_depth = 4
    result = np.zeros((2, 1)) # Stores the number of wins for each agent
    for i in range(n_iterations):
        print(f"Start round {i}")
        # Let the agents play one round (consisting of each agent starting one game)
        # and add the wins to the result array
        result += play_one_round(generate_move_1=minimax_move, generate_move_2=MCTS_move,
                       args_1=minimax_depth, args_2=MCTS_time, print_board=False)
    # Calculate the percentage of wins and add the percentage of draws
    perc_win = result / (n_iterations * 2)  # * 2 as a round consists of two plays
    draw =  1 - np.sum(perc_win)
    perc_win = np.vstack((perc_win, [[draw]]))
    if plot_res:
        # Plot a pie chart of the winning percentages
        labels = [f"Minimax with depth {minimax_depth}", f"MCTS with max time {MCTS_time}", "Draw"]
        print(perc_win)
        plt.pie(np.squeeze(perc_win), labels=labels)
        plt.show()


if __name__ == "__main__":
    agent_vs_agent_often(5, True)
    #play_one_round()  # Either human vs. agent or agent vs. agent