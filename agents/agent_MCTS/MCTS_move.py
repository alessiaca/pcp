import numpy as np
import time
from typing import Optional, Tuple, List, Union
from agents.common import PlayerAction, BoardPiece, SavedState, apply_player_action, connect_four,\
    check_end_state, GameState, CONNECT_N, PLAYER1, PLAYER2, NO_PLAYER


def poss_actions(board: np.ndarray) -> np.ndarray:
    """
    :param state: Board
    :return: Array of possible actions/free columns
    """
    return np.where(board[0, :] == NO_PLAYER)[0]


class Node:
    def __init__(self, action=None, parent=None, board=None, player=None):
        self.parent = parent
        self.action = action
        self.board = board
        self.player = player
        self.wins = 0
        self.visits = 0
        self.children = []
        self.untried_actions = poss_actions(board) # Array of free columns

    def selection(self, c=1):
        """
        c: Exploration parameter
        :return: Returns the child node with the largest UCB1 value
        """
        # Define a function which returns the UCB1 value for each child
        ucb_func = lambda child: child.wins / child.visits + c * np.sqrt(np.log(self.visits) / child.visits)
        # Return the child with the largest score
        return sorted(self.children, key=ucb_func)[-1]

    def expansion(self, action: PlayerAction):
        """
        :param action: Action to apply in order to expand a node
        :return: Child after node expansion: The board of the child node
        corresponds to the board of the parent after the action was applied
        """
        # The parent node was created by taking an action for a specific player. For
        # the expansion choose the opponent , as the players have alternating turns
        player_exp = PLAYER1 if self.player == PLAYER2 else PLAYER1
        # Apply the action to the board of the parent node
        new_board = apply_player_action(self.board.copy(), action, player_exp)
        # Create a new child node with that action and board
        child = Node(action=action, parent=self, board=new_board, player=player_exp)
        # Append the created child to the children of the parent
        self.children.append(child)
        # Remove the action form the untried actions from the parent
        self.untried_actions = np.setdiff1d(self.untried_actions, action)

        return child

    def update(self, result: int):
        """
        Updated the win and visits value of a node
        :param result: 0 or 1 indicating whether the player won the simulation
        """
        self.visits += 1
        self.wins += result



def MCTS(board: np.ndarray, player: BoardPiece, max_time: float) -> PlayerAction:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID of the player for which a good move (that generates a win most probably)
     has to be chosen
    :param max_time: Number of seconds until an action need to be chosen
    :return: Column in which player wants to make his move (chosen using MCTS)
    """
    # Initialize the root of the search tree with the current board state (based on
    # which an action needs to be found) and the player of the opponent
    root_node = Node(board=board, player=PLAYER1 if player == PLAYER2 else PLAYER2)

    # Perform as many iterations of MCTS as allowed by the maximal time/maximum number of iterations
    end_time = time.time() + max_time
    while time.time() < end_time:

        # Start at the root node at each iteration
        node = root_node

        # Selection
        # Go down the tree until a terminal node or a node with untried moves is reached
        while not np.any(node.untried_actions) and node.children != []:
            node = node.selection()

        # Expansion
        # If not all actions were tried, choose a random action and append a child node
        if np.any(node.untried_actions):
            # Choose a random action from the untried actions
            action = np.random.choice(node.untried_actions)
            # Expand the current node generating a new child node with a board after the
            # application of the action and remove the action from the untried actions
            # of the current node
            node = node.expansion(action)

        # Simulation
        # Generate random moves of the players until the board is full
        board = node.board.copy()
        win = False
        player_sim = player
        while np.any(poss_actions(board)) and not win:
            # Choose the other player for the first/next random move
            player_sim = PLAYER1 if player_sim == PLAYER2 else PLAYER2
            # Choose a random action
            action = np.random.choice(poss_actions(board))
            # Apply the action
            board = apply_player_action(board, action, player_sim)
            # Check if the game is won
            win = connect_four(board, player_sim)

        # Backpropagation
        # Update the number of visits and wins for each node
        # Check if player won the random simulation (win will be false if the game ended in a draw)
        result = 1 if win and player_sim == player else 0
        # Go up the tree until reaching the root node
        while node is not None:
            node.update(result)
            node = node.parent


    # Choose the best action based on the ratio of wins and visits
    eval_func = lambda child: child.wins / child.visits
    # Get the child with the best value
    best_child = sorted(root_node.children, key=eval_func)[-1]
    # Return the action of the child
    return best_child.action


def generate_move_MCTS(board: np.ndarray, player: BoardPiece, saved_state: Optional[SavedState]) \
        -> Tuple[PlayerAction, SavedState]:
    """
    :param board: State of board, 6 x 7 with either 0 or player ID [1, 2]
    :param player: Player ID
    :param saved_state: Not used in this implementation of the move generation
    :return: Column in which player wants to make his move (chosen using MCTS)
    """
    action = MCTS(board, player, 5)
    return PlayerAction(action), SavedState()