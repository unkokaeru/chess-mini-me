"""Encode chess positions and moves as tensors for the Mini-Me network.

The Mini-Me opponent learns to imitate a player, so it needs the board turned
into numbers a neural network can consume, and a way to turn the network's
output back into a move. Both directions live here so that the live game (which
uses :class:`chess_mini_me.engine.GameState`) and the Lichess importer (which
uses python-chess) can share exactly the same representation.

The representation follows the convention used by modern board-game networks:

* The position is a stack of 8x8 "planes", each a small picture of one feature
  (where the white knights are, whose turn it is, and so on).
* A move is an index into a flat policy vector of size 64 x 64, formed from the
  origin square and the destination square. Promotions are predicted by a small
  separate head because, in this engine, the promotion piece is chosen when the
  move is made rather than being part of the move's identity.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from chess_mini_me import constants

if TYPE_CHECKING:
    from chess_mini_me.engine import GameState, Move

# The order of the twelve piece planes. The plane index of a piece is its
# position in this tuple.
PIECE_PLANE_ORDER: tuple[str, ...] = (
    "wP", "wN", "wB", "wR", "wQ", "wK",
    "bP", "bN", "bB", "bR", "bQ", "bK",
)
PIECE_TO_PLANE: dict[str, int] = {
    piece: plane for plane, piece in enumerate(PIECE_PLANE_ORDER)
}

# Twelve piece planes, one side-to-move plane, four castling-right planes and
# one en passant plane.
NUMBER_OF_PIECE_PLANES = len(PIECE_PLANE_ORDER)
NUMBER_OF_INPUT_PLANES = NUMBER_OF_PIECE_PLANES + 1 + 4 + 1

# The policy predicts an (origin square, destination square) pair.
NUMBER_OF_SQUARES = constants.BOARD_DIMENSION * constants.BOARD_DIMENSION
POLICY_SIZE = NUMBER_OF_SQUARES * NUMBER_OF_SQUARES

# The promotion head predicts which piece a promoting pawn becomes.
PROMOTION_PIECES: tuple[str, ...] = (
    constants.QUEEN,
    constants.ROOK,
    constants.BISHOP,
    constants.KNIGHT,
)
PROMOTION_TO_INDEX: dict[str, int] = {
    piece: index for index, piece in enumerate(PROMOTION_PIECES)
}
NUMBER_OF_PROMOTION_CLASSES = len(PROMOTION_PIECES)


def square_index(row: int, column: int) -> int:
    """Return the flat index of a board square.

    Args:
        row: The square's row (0 is the eighth rank, the top of the board).
        column: The square's column (0 is the a-file).

    Returns:
        The flattened index ``row * 8 + column``.
    """
    return row * constants.BOARD_DIMENSION + column


def move_to_policy_index(
    start_row: int, start_column: int, end_row: int, end_column: int
) -> int:
    """Return the policy-vector index for a move between two squares.

    Args:
        start_row: The origin row.
        start_column: The origin column.
        end_row: The destination row.
        end_column: The destination column.

    Returns:
        The index into the flat policy vector identifying this move.
    """
    origin = square_index(start_row, start_column)
    destination = square_index(end_row, end_column)
    return origin * NUMBER_OF_SQUARES + destination


def encode_planes(
    board: list[list[str]],
    white_to_move: bool,
    castling_rights: tuple[bool, bool, bool, bool],
    en_passant_square: tuple[int, int] | None,
) -> np.ndarray:
    """Encode a position as a stack of feature planes.

    Args:
        board: The 8x8 board in the engine's ``"wP"`` style notation.
        white_to_move: Whether it is White's turn.
        castling_rights: The four flags
            ``(white_king_side, white_queen_side, black_king_side,
            black_queen_side)``.
        en_passant_square: The square a pawn may capture into by en passant, or
            ``None``.

    Returns:
        A float32 array of shape ``(NUMBER_OF_INPUT_PLANES, 8, 8)``.
    """
    planes = np.zeros(
        (NUMBER_OF_INPUT_PLANES, constants.BOARD_DIMENSION, constants.BOARD_DIMENSION),
        dtype=np.float32,
    )

    for row in range(constants.BOARD_DIMENSION):
        for column in range(constants.BOARD_DIMENSION):
            piece = board[row][column]
            if piece in PIECE_TO_PLANE:
                planes[PIECE_TO_PLANE[piece], row, column] = 1.0

    if white_to_move:
        planes[NUMBER_OF_PIECE_PLANES, :, :] = 1.0

    for offset, has_right in enumerate(castling_rights):
        if has_right:
            planes[NUMBER_OF_PIECE_PLANES + 1 + offset, :, :] = 1.0

    if en_passant_square is not None:
        en_passant_row, en_passant_column = en_passant_square
        planes[NUMBER_OF_INPUT_PLANES - 1, en_passant_row, en_passant_column] = 1.0

    return planes


def encode_gamestate(gamestate: "GameState") -> np.ndarray:
    """Encode a live :class:`GameState` as feature planes.

    Args:
        gamestate: The current game state.

    Returns:
        The encoded position, as returned by :func:`encode_planes`.
    """
    rights = gamestate.current_castling_rights
    castling_rights = (
        rights.white_king_side,
        rights.white_queen_side,
        rights.black_king_side,
        rights.black_queen_side,
    )
    en_passant_square = (
        gamestate.en_passant_target if gamestate.en_passant_target else None
    )
    return encode_planes(
        gamestate.board,
        gamestate.white_to_move,
        castling_rights,
        en_passant_square,
    )


def build_legal_move_lookup(
    legal_moves: list["Move"],
) -> tuple[np.ndarray, dict[int, "Move"]]:
    """Build a legality mask and an index-to-move lookup for a position.

    Args:
        legal_moves: The legal moves in the current position.

    Returns:
        A tuple ``(mask, index_to_move)`` where ``mask`` is a float32 vector of
        length ``POLICY_SIZE`` holding 1 for legal moves and 0 elsewhere, and
        ``index_to_move`` maps a policy index back to its move.
    """
    mask = np.zeros(POLICY_SIZE, dtype=np.float32)
    index_to_move: dict[int, "Move"] = {}
    for move in legal_moves:
        index = move_to_policy_index(
            move.start_row, move.start_column, move.end_row, move.end_column
        )
        mask[index] = 1.0
        index_to_move[index] = move
    return mask, index_to_move
