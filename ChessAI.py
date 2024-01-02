"""The chess AI, which is a work in progress"""

# Imports
import random

from ChessEngine import Move


def find_random_move(valid_moves: list) -> Move:
    """
    Pick a random valid move
    :param valid_moves: list of valid moves
    :return: a random move
    """

    return valid_moves[random.randint(0, len(valid_moves) - 1)]
