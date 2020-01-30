"""
Mybot -- A simple strategy: looks if it has a trump card otherwise plays the lowest card
"""

# Import the API objects
from api import State
from api import State
from api import Deck
import random


class Bot:

    def __init__(self):
        pass

    def get_move(self, state):

        # Get a list of all legal moves
        moves = state.moves()
        chosen_move = moves[0]
        moves_trump_suit = []

        # Get all trump suit moves available
        for index, move in enumerate(moves):

            if move[0] is not None and Deck.get_suit(move[0]) == state.get_trump_suit():
                moves_trump_suit.append(move)

        if len(moves_trump_suit) > 0:
            chosen_move = moves_trump_suit[0]
            return chosen_move

        # Get move with lowest rank available, of any suit
        for index, move in enumerate(moves):
            if move[0] is not None and move[0] % 5 > chosen_move[0] % 5:
                chosen_move = move
        return chosen_move

