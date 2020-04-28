import numpy as np
import connectn.common as common


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
    assert pp_board
    assert len(pp_board) > 42


def test_string_to_board():

    board = common.initialize_game_state()
    pp_board = common.pretty_print_board(board)
    st_board = common.string_to_board(pp_board)

    assert isinstance(st_board, np.ndarray)
    assert st_board.dtype == np.int8
    assert st_board.shape == (6, 7)
    # assert not np.all(st_board == 0)


def test_apply_player_action():

    board = common.initialize_game_state()  # Try different boards
    action = 0  # Try different actions
    player = 1  # Try different players
    new_board1 = common.apply_player_action(board=board, action=action, player=player)
    new_board2 = common.apply_player_action(board=new_board1, action=action, player=player)
    assert isinstance(new_board1, np.ndarray)
    assert new_board1.dtype == np.int8
    assert new_board1.shape == (6, 7)
    assert not np.all(new_board1 == 0)
    assert new_board1[5, action] == player
    assert new_board2[4, action] == player


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
    board4 = np.fill_diagonal(board4, 1)
    win_board = common.connect_four(board,player)
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



