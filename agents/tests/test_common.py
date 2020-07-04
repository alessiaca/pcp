import numpy as np
import agents.common as common
from agents.common import initialize_game_state, pretty_print_board, apply_player_action, connect_four,\
     check_end_state, PLAYER1, PLAYER2, NO_PLAYER, PlayerAction, CONNECT_N, GameState

players = [PLAYER1, PLAYER2]


def test_initialize_game_state():

    ret = initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)
    assert np.all(ret == 0)


def test_pretty_print_board():

    board = initialize_game_state()
    pp_board = pretty_print_board(board)

    assert isinstance(pp_board, str)
    assert len(pp_board) > 42


def test_apply_player_action():

    # Test if application of action (drop of the board piece) is possible for every cell and player
    for player in players:
        board = initialize_game_state()
        n_rows = board.shape[0]
        for action, column in enumerate(board.T):
            for i in range(n_rows):
                player = common.PLAYER1
                board = apply_player_action(board=board, action=action, player=player)
                assert isinstance(board, np.ndarray)
                assert board.dtype == np.int8
                assert board.shape == (6, 7)
                assert not np.all(board == 0)
                assert board[n_rows - 1 - i, action] == player

    # Testing for an insertion in a full board/column is not necessary, as this is already taken care of in the
    # user_move, minimax_move and random_move function


def test_connect_four():

    # Get the shape of the board
    board = initialize_game_state()
    n_rows, n_cols = board.shape

    # Check that the expected return type bool is correct
    assert isinstance(connect_four(board, PLAYER1), bool)

    # Make sure that for neither player a win is detected if he has less than CONNECT_N (4) adjacent pieces on the
    # board
    for player in players:

        # Make sure that a win in a column is detected
        for action in range(n_cols):
            board = initialize_game_state()
            for i in range(CONNECT_N):
                assert not connect_four(board, player)
                board = apply_player_action(board, PlayerAction(action), player)
            assert connect_four(board, player)

        # Make sure that a win in a row is detected
        for row in range(n_rows):
            board = initialize_game_state()
            for i in range(CONNECT_N):
                assert not connect_four(board, player)
                board = apply_player_action(board, PlayerAction(i), player)
            assert connect_four(board, player)

        # Make sure that a win in a diagonal is detected
        board = initialize_game_state()
        np.fill_diagonal(board, player)
        assert connect_four(board, player)
        assert connect_four(board[::-1, :], player)


def test_check_end_state():

    for player in players:
        board = initialize_game_state()
        assert check_end_state(board, player) == GameState.STILL_PLAYING
        board[:, :] = player
        assert check_end_state(board, player) == GameState.IS_WIN
        board[:, :] = 3
        assert check_end_state(board, player) == GameState.IS_DRAW


# Run the tests when executing the script
test_pretty_print_board()
test_initialize_game_state()
test_apply_player_action()
test_connect_four()
test_check_end_state()
