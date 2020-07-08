import numpy as np
import pytest
from agents.common import initialize_game_state, pretty_print_board, apply_player_action, connect_four, \
     string_to_board, check_end_state, PLAYER1, PLAYER2, PlayerAction, CONNECT_N, GameState
from agents.agent_minimax import minimax_move
from agents.agent_MCTS import MCTS_move

move_agents = [minimax_move, MCTS_move]

players = [PLAYER1, PLAYER2]

#TODO Write test for the agents

def test_initialize_game_state():
    """Test the initialization of the board"""

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)
    assert np.all(ret == 0)


def test_pretty_print_board_and_string_to_board():
    """Test the printing of a board and the transformation back to an array"""

    # Check that the print of the board to a string and the transformation back to the board works for every position on
    # the board for each player
    for player in players:
        board = initialize_game_state()
        n_rows, n_cols = board.shape
        for i in range(n_rows):
            for j in range(n_cols):
                board[i, j] = player
                # Transform the board into a pretty string
                pp_board = pretty_print_board(board)
                # Transform it back to an ndarray
                trans_board = string_to_board(pp_board)

                # Check that everything worked
                assert isinstance(pp_board, str)
                assert len(pp_board) > 42
                assert np.array_equal(board, trans_board)


def test_apply_player_action_success():
    """Test for successful application of actions"""

    # Test if application of action (drop of the board piece) is possible for every cell and player
    for player in players:
        board = initialize_game_state()
        n_rows = board.shape[0]
        for action, column in enumerate(board.T):
            for i in range(n_rows):
                board = apply_player_action(board=board, action=PlayerAction(action), player=player)
                assert isinstance(board, np.ndarray)
                assert board.dtype == np.int8
                assert board.shape == (6, 7)
                assert not np.all(board == 0)
                assert board[n_rows - 1 - i, action] == player


def test_apply_player_action_fail():
    """Test that an error is raised if an action in a full or not existent column is applied"""

    # Test for the insertion in a already full column
    full_board = initialize_game_state()
    full_board[:] = PLAYER1  # Fill the board completely with one player
    n_cols = full_board.shape[1]
    for i in range(n_cols):  # Check that the exception is raised in every column
        with pytest.raises(Exception) as e:
            assert apply_player_action(full_board, PlayerAction(i), PLAYER1)
            assert str(e.value) == "Tried to apply an action in a non existent or full column"
    # Test for a non existent column
    with pytest.raises(Exception) as e:
        assert apply_player_action(initialize_game_state(), PlayerAction(100), PLAYER1)
        assert str(e.value) == "Tried to apply an action in a non existent or full column"


def test_connect_four():
    """Test that 4 adjacent pieces at every position are detected"""

    # Get the shape of the board
    empty_board = initialize_game_state()
    rows, cols = empty_board.shape
    rows_edge = rows - CONNECT_N + 1
    cols_edge = cols - CONNECT_N + 1
    # Test that every possible win for each player is detected
    for player in players:
        # Test for wins in all rows
        for i in range(rows):
            for j in range(cols_edge):
                board = empty_board.copy()
                for h in range(CONNECT_N):
                    # For less than CONNECT_N (here 4) pieces no win should be detected
                    assert not connect_four(board, player)
                    board[i, j + h] = player
                # After CONNECT_N pieces were placed a win should be detected
                assert connect_four(board, player)

        # Test for wins in all columns
        for i in range(rows_edge):
            for j in range(cols):
                board = empty_board.copy()
                for h in range(CONNECT_N):
                    # For less than CONNECT_N (here 4) pieces no win should be detected
                    assert not connect_four(board, player)
                    board[i + h, j] = player
                # After CONNECT_N pieces were placed a win should be detected
                assert connect_four(board, player)

        # Test for wins in the diagonals
        for i in range(rows_edge):
            for j in range(cols_edge):
                board_1 = empty_board.copy()
                board_2 = empty_board.copy()
                for h in range(1, CONNECT_N + 1):
                    block = np.zeros((h, h))
                    np.fill_diagonal(block, player)
                    board_1[i:i + h, j:j + h] = block
                    board_2[i:i + h, j:j + h] = block[::-1, :]
                    if h < CONNECT_N:
                        assert not connect_four(board_1, player)
                        assert not connect_four(board_2, player)
                    else:
                        assert connect_four(board_1, player)
                        assert connect_four(board_2, player)


def test_check_end_state():
    """Check that a win, draw or loss is correctly detected"""

    for player in players:
        board = initialize_game_state()
        assert check_end_state(board, player) == GameState.STILL_PLAYING
        board[:, :] = player
        assert check_end_state(board, player) == GameState.IS_WIN
        # Create a draw board which is valid (contains only player pieces)
        n_rows, n_cols = board.shape
        board_flat = players * int((n_cols * n_rows) / 2)
        draw_board = np.reshape(board_flat, (n_rows, n_cols))
        draw_board[:, 1] = draw_board[:, 1][::-1]
        draw_board[:, 5] = draw_board[:, 5][::-1]
        assert check_end_state(draw_board, player) == GameState.IS_DRAW


def test_agents():
    """ Test that the agents minimax and MCTS take immediate wins and block immediate losses"""

    empty_board = initialize_game_state()
    n_rows, n_cols = empty_board.shape
    for player in players:
        opponent = PLAYER1 if player == PLAYER2 else PLAYER2
        # Test for immediate wins (p=player) and immediate losses (p=opponent)
        for p in (player, opponent):
            board_col = empty_board.copy()
            board_row = empty_board.copy()
            # Check for win and loss in a row and column
            for i in range(CONNECT_N - 1):
                board_row = apply_player_action(board_row, PlayerAction(i), p)
                board_col = apply_player_action(board_col, PlayerAction(CONNECT_N - 1), p)
            # Check that both agents make the right move (always column CONNECT_N -1 = 3)
            for move_agent in move_agents:
                for board in [board_row, board_col]:
                    action = move_agent(board, player, None)[0]
                    assert action == PlayerAction(CONNECT_N - 1)

        # Test that the agent blocks a certain win of the opponent (two free player pieces in the middle of
        # the board generate a certain win if the player does not put his piece to the right of left)
        board = empty_board.copy()
        board[-1, 1:3] = opponent
        for move_agent in move_agents:
            action = move_agent(board, player, None)[0]
            assert action == PlayerAction(0) or action == PlayerAction(3)


# Run the tests when executing the script
test_pretty_print_board_and_string_to_board()
test_initialize_game_state()
test_apply_player_action_success()
test_apply_player_action_fail()
test_connect_four()
test_check_end_state()
test_agents()
