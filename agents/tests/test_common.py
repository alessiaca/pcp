import numpy as np
import agents.common as common


def test_initialize_game_state():
    ret = common.initialize_game_state()

    assert isinstance(ret, np.ndarray)
    assert ret.dtype == np.int8
    assert ret.shape == (6, 7)
    assert np.all(ret == 0)


def test_pretty_print_board():
    board = common.initialize_game_state()
    pp_board = common.pretty_print_board(board)

    assert isinstance(pp_board, str)
    assert len(pp_board) > 42


def test_string_to_board():
    # not implemented yet
    pass


def test_apply_player_action():
    board = common.initialize_game_state()
    player = 1
    for action, column in enumerate(board.T):
        for i in range(2):
            player = common.PLAYER1
            board = common.apply_player_action(board=board, action=action, player=player)
            assert isinstance(board, np.ndarray)
            assert board.dtype == np.int8
            assert board.shape == (6, 7)
            assert not np.all(board == 0)
            assert board[5 - i, action] == player
    # Testing for an insertion in a full board/column is not necessary, as this is already taken care of in the
    # user_move and generate_move_minimax/random function


def test_connect_four():
    board = common.initialize_game_state()
    player = 1
    n_rows = board.shape[0]
    n_columns = board.shape[1]
    board1 = board.copy()
    board2 = board.copy()
    board3 = board.copy()
    board4 = board.copy()
    board1[1, :] = 1
    board2[:, 3] = 1
    board3[1, 5] = 1
    np.fill_diagonal(board4, 1)
    win_board = common.connect_four(board, player)
    win_board1 = common.connect_four(board1, player)
    win_board2 = common.connect_four(board2, player)
    win_board3 = common.connect_four(board3, player)
    win_board4 = common.connect_four(board4, player)

    assert isinstance(win_board, bool)
    assert not win_board
    assert win_board1
    assert win_board2
    assert not win_board3
    assert win_board4


test_pretty_print_board()
test_string_to_board()
test_initialize_game_state()
test_apply_player_action()
test_connect_four()
