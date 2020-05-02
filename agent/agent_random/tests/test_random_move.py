import numpy as np
from  agent_random.random_move import generate_move_random
from connectn.common import initialize_game_state
from connectn.common import pretty_print_board


def test_generate_move_random():

    board = initialize_game_state()
    player = 1
    savedstate = 1
    generate_move_random(board, 1, 1)

    pass

