#!/usr/bin/env python
"""
A basic adaptive bot. This is part of the third worksheet.

"""

from api import State, util, Deck
import random, os
from itertools import chain

from sklearn.externals import joblib

# Path of the model we will use. If you make a model
# with a different name, point this line to its path.
DEFAULT_MODEL = os.path.dirname(os.path.realpath(__file__)) + '/model.pkl'


class Bot:
    __randomize = True

    __model = None

    def __init__(self, randomize=True, model_file=DEFAULT_MODEL):

        print(model_file)
        self.__randomize = randomize

        # Load the model
        self.__model = joblib.load(model_file)

    def get_move(self, state):

        val, move = self.value(state)

        return move

    def value(self, state):
        """
        Return the value of this state and the associated move
        :param state:
        :return: val, move: the value of the state, and the best move.
        """

        best_value = float('-inf') if maximizing(state) else float('inf')
        best_move = None

        moves = state.moves()

        if self.__randomize:
            random.shuffle(moves)

        for move in moves:

            next_state = state.next(move)

            # IMPLEMENT: Add a function call so that 'value' will
            # contain the predicted value of 'next_state'
            # NOTE: This is different from the line in the minimax/alphabeta bot
            value = self.heuristic(next_state)

            if maximizing(state):
                if value > best_value:
                    best_value = value
                    best_move = move
            else:
                if value < best_value:
                    best_value = value
                    best_move = move

        return best_value, best_move

    def heuristic(self, state):

        # Convert the state to a feature vector
        feature_vector = [features(state)]

        # These are the classes: ('won', 'lost')
        classes = list(self.__model.classes_)

        # Ask the model for a prediction
        # This returns a probability for each class
        prob = self.__model.predict_proba(feature_vector)[0]

        # Weigh the win/loss outcomes (-1 and 1) by their probabilities
        res = -1.0 * prob[classes.index('lost')] + 1.0 * prob[classes.index('won')]

        return res


def maximizing(state):
    """
    Whether we're the maximizing player (1) or the minimizing player (2).
    :param state:
    :return:
    """
    return state.whose_turn() == 1


def features(state):
    # type: (State) -> tuple[float, ...]
    """
    Extract features from this state. Remember that every feature vector returned should have the same length.

    :param state: A state to be converted to a feature vector
    :return: A tuple of floats: a feature vector representing this state.
    """

    feature_set = []

    # Add player 1's points to feature set
    p1_points = state.get_points(1)
    #feature_set.append(p1_points)

    # Add player 2's points to feature set
    p2_points = state.get_points(2)
    #feature_set.append(p2_points)

    # Add player 1's pending points to feature set
    p1_pending_points = state.get_pending_points(1)
    #feature_set.append(p1_pending_points)

    # Add player 2's pending points to feature set
    p2_pending_points = state.get_pending_points(2)
    #feature_set.append(p2_pending_points)

    # Get trump suit
    trump_suit = state.get_trump_suit()

    # Add phase to feature set
    phase = state.get_phase()
    #feature_set.append(phase)

    # Add stock size to feature set
    stock_size = state.get_stock_size()
    #feature_set.append(stock_size)

    # Add leader to feature set
    leader = state.leader()
    #feature_set.append(leader)

    # Add whose turn it is to feature set
    whose_turn = state.whose_turn()
    #feature_set.append(whose_turn)

    # Add opponent's played card to feature set
    opponents_played_card = state.get_opponents_played_card()

    ###   Added features  ###

    # Points in hand
    hand = state.hand()
    feature_set.append(hand)
    num_cards = state.hand()
    point_state = 0
    if opponents_played_card is not None:
        if card in num_cards:
            if opponents_played_card == 4 or opponents_played_card  == 9 or opponents_played_card  == 14 or opponents_played_card == 19:
                if hand != 4 or hand != 9 or hand != 14 or hand != 19:
                    point_state += 1
            elif opponents_played_card == 3 or opponents_played_card == 8 or opponents_played_card == 13 or opponents_played_card == 18:
                if hand != 3 or hand != 8 or hand != 13 or hand != 18 or hand != 4 or hand != 9 or hand != 14 or hand != 19:
                    point_state += 1
            elif opponents_played_card == 2 or opponents_played_card == 7 or opponents_played_card == 12 or opponents_played_card == 17:
                if hand != 3 or hand != 8 or hand != 13 or hand != 18 or hand != 4 or hand != 9 or hand != 14 or hand != 19 or hand != 2 or hand != 7 or hand != 12 or hand != 17:
                    point_state += 1
            elif opponents_played_card == 1 or opponents_played_card == 6 or opponents_played_card == 11 or opponents_played_card == 16:
                if hand != 3 or hand != 8 or hand != 13 or hand != 18 or hand != 4 or hand != 9 or hand != 14 or hand != 19 or hand != 2 or hand != 7 or hand != 12 or hand != 17 or hand != 1 or hand != 6 or hand != 11 or hand != 16:
                    point_state += 1
            elif opponents_played_card == 0 or opponents_played_card == 5 or opponents_played_card == 10 or opponents_played_card == 15:
                if hand != 0 or hand != 5 or hand != 10 or hand != 15:
                    point_state += 0
        # elif Deck.get_rank(num_cards[0]) == "10":
        #     num_points += 10
        # elif Deck.get_rank(num_cards[0]) == "K":
        #     num_points += 3
        # elif Deck.get_rank(num_cards[0]) == "Q":
        #     num_points += 2
        # elif Deck.get_rank(num_cards[0]) == "J":
        #     num_points += 1
    print(num_cards)
    print(point_state)
    feature_set.append(point_state)


    ################## You do not need to do anything below this line ########################

    perspective = state.get_perspective()

    # Perform one-hot encoding on the perspective.
    # Learn more about one-hot here: https://machinelearningmastery.com/how-to-one-hot-encode-sequence-data-in-python/
    perspective = [card if card != 'U' else [1, 0, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'S' else [0, 1, 0, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'P1H' else [0, 0, 1, 0, 0, 0] for card in perspective]
    perspective = [card if card != 'P2H' else [0, 0, 0, 1, 0, 0] for card in perspective]
    perspective = [card if card != 'P1W' else [0, 0, 0, 0, 1, 0] for card in perspective]
    perspective = [card if card != 'P2W' else [0, 0, 0, 0, 0, 1] for card in perspective]

    # Append one-hot encoded perspective to feature_set
    feature_set += list(chain(*perspective))

    # Append normalized points to feature_set
    total_points = p1_points + p2_points
    feature_set.append(p1_points / total_points if total_points > 0 else 0.)
    feature_set.append(p2_points / total_points if total_points > 0 else 0.)

    # Append normalized pending points to feature_set
    total_pending_points = p1_pending_points + p2_pending_points
    feature_set.append(p1_pending_points / total_pending_points if total_pending_points > 0 else 0.)
    feature_set.append(p2_pending_points / total_pending_points if total_pending_points > 0 else 0.)

    # Convert trump suit to id and add to feature set
    # You don't need to add anything to this part
    suits = ["C", "D", "H", "S"]
    trump_suit_onehot = [0, 0, 0, 0]
    trump_suit_onehot[suits.index(trump_suit)] = 1
    feature_set += trump_suit_onehot

    # Append one-hot encoded phase to feature set
    feature_set += [1, 0] if phase == 1 else [0, 1]

    # Append normalized stock size to feature set
    feature_set.append(stock_size / 10)

    # Append one-hot encoded leader to feature set
    feature_set += [1, 0] if leader == 1 else [0, 1]

    # Append one-hot encoded whose_turn to feature set
    feature_set += [1, 0] if whose_turn == 1 else [0, 1]

    # Append one-hot encoded opponent's card to feature set
    opponents_played_card_onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    opponents_played_card_onehot[opponents_played_card if opponents_played_card is not None else 20] = 1
    feature_set += opponents_played_card_onehot

    # Return feature set
    return feature_set
