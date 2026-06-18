"""A simple chess move finder using negamax search with alpha-beta pruning.

The evaluation combines material values with piece-square tables so that the
engine prefers active, well-placed pieces as well as a healthy material
balance. The search is intentionally compact: a single negamax routine serves
both sides, which keeps the code short and easy to reason about.
"""

from __future__ import annotations

import math
import random

from chess_mini_me import constants
from chess_mini_me.engine import GameState, Move


def find_best_move(
    gamestate: GameState,
    valid_moves: list[Move],
    depth: int = constants.DEFAULT_SEARCH_DEPTH,
) -> Move | None:
    """Choose the best move for the side to move.

    Args:
        gamestate: The current game state. It is restored to its original form
            before this function returns.
        valid_moves: The legal moves available to the side to move.
        depth: The search depth in plies (half-moves).

    Returns:
        The chosen move, or ``None`` when no legal moves are available.
    """
    if not valid_moves:
        return None

    # The perspective sign: evaluations are written from White's point of view,
    # so White maximises a positive score and Black maximises its negation.
    perspective = 1 if gamestate.white_to_move else -1

    # Shuffling avoids always picking the first of several equally good moves,
    # which makes the opponent feel less predictable.
    ordered_moves = list(valid_moves)
    random.shuffle(ordered_moves)

    best_move = ordered_moves[0]
    best_score = -math.inf
    alpha, beta = -math.inf, math.inf

    for move in ordered_moves:
        gamestate.make_move(move)
        opponent_moves = gamestate.get_valid_moves()
        score = -_negamax(
            gamestate, opponent_moves, depth - 1, -beta, -alpha, -perspective
        )
        gamestate.undo_move()

        if score > best_score:
            best_score = score
            best_move = move
        alpha = max(alpha, best_score)

    return best_move


def _negamax(
    gamestate: GameState,
    valid_moves: list[Move],
    depth: int,
    alpha: float,
    beta: float,
    perspective: int,
) -> float:
    """Return the negamax value of a position from the mover's perspective.

    Args:
        gamestate: The current game state, restored before returning.
        valid_moves: The legal moves for the side to move in this position.
        depth: The remaining search depth in plies.
        alpha: The best score the maximising side can already guarantee.
        beta: The best score the minimising side can already guarantee.
        perspective: ``1`` when the side to move is White, otherwise ``-1``.

    Returns:
        The evaluation of the position from the side-to-move's perspective.
    """
    # A position with no legal moves is checkmate or stalemate; the flags were
    # set by the ``get_valid_moves`` call that produced ``valid_moves``.
    if depth == 0 or not valid_moves or gamestate.draw:
        return perspective * evaluate_board(gamestate)

    best_score = -math.inf
    for move in valid_moves:
        gamestate.make_move(move)
        next_moves = gamestate.get_valid_moves()
        score = -_negamax(
            gamestate, next_moves, depth - 1, -beta, -alpha, -perspective
        )
        gamestate.undo_move()

        best_score = max(best_score, score)
        alpha = max(alpha, best_score)
        if alpha >= beta:
            # The opponent would never allow this line, so stop searching it.
            break

    return best_score


def evaluate_board(gamestate: GameState) -> int:
    """Score a position from White's perspective, in centipawns.

    A positive score favours White and a negative score favours Black. Terminal
    positions return the checkmate or stalemate score.

    Args:
        gamestate: The game state to evaluate.

    Returns:
        The position's evaluation, where larger is better for White.
    """
    if gamestate.checkmate:
        # The side to move has been mated, so the score favours the other side.
        return (
            -constants.CHECKMATE_SCORE
            if gamestate.white_to_move
            else constants.CHECKMATE_SCORE
        )
    if gamestate.stalemate or gamestate.draw:
        return constants.STALEMATE_SCORE

    score = 0
    for row in range(constants.BOARD_DIMENSION):
        for column in range(constants.BOARD_DIMENSION):
            piece = gamestate.board[row][column]
            if piece == constants.EMPTY_SQUARE:
                continue
            score += _piece_score(piece, row, column)
    return score


def _piece_score(piece: str, row: int, column: int) -> int:
    """Return a single piece's signed contribution to the evaluation.

    Args:
        piece: The two-character piece code, for example ``wQ``.
        row: The piece's row.
        column: The piece's column.

    Returns:
        The piece's material plus positional value, positive for White and
        negative for Black.
    """
    colour, piece_type = piece[0], piece[1]

    # White reads the piece-square table directly; Black reads the vertically
    # mirrored square so that both sides are rewarded for advancing.
    table = constants.PIECE_POSITION_VALUE[piece_type]
    if colour == constants.WHITE:
        positional_value = table[row][column]
        sign = 1
    else:
        positional_value = table[constants.BOARD_DIMENSION - 1 - row][column]
        sign = -1

    return sign * (constants.PIECE_MATERIAL_VALUE[piece_type] + positional_value)
