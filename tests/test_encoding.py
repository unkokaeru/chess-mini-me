"""Tests for encoding positions and moves as tensors."""

from __future__ import annotations

from chess_mini_me import constants, encoding
from chess_mini_me.engine import GameState, Move


def test_starting_position_planes() -> None:
    """The encoded starting position should have the expected plane shape."""
    planes = encoding.encode_gamestate(GameState())
    assert planes.shape == (
        encoding.NUMBER_OF_INPUT_PLANES,
        constants.BOARD_DIMENSION,
        constants.BOARD_DIMENSION,
    )
    # Eight white pawns sit on the white-pawn plane.
    assert planes[encoding.PIECE_TO_PLANE["wP"]].sum() == 8
    # White is to move, so the side-to-move plane is fully set.
    assert planes[encoding.NUMBER_OF_PIECE_PLANES].all()


def test_policy_index_round_trip() -> None:
    """Every legal move should appear exactly once in the lookup."""
    gamestate = GameState()
    legal_moves = gamestate.get_valid_moves()
    mask, index_to_move = encoding.build_legal_move_lookup(legal_moves)
    assert mask.sum() == len(legal_moves)
    for move in legal_moves:
        index = encoding.move_to_policy_index(
            move.start_row, move.start_column, move.end_row, move.end_column
        )
        assert index_to_move[index] == move


def test_en_passant_plane_is_set() -> None:
    """A double pawn advance should set the en passant plane."""
    gamestate = GameState()
    for legal_move in gamestate.get_valid_moves():
        if legal_move == Move((6, 4), (4, 4), gamestate.board):
            gamestate.make_move(legal_move)
            break
    planes = encoding.encode_gamestate(gamestate)
    en_passant_plane = planes[encoding.NUMBER_OF_INPUT_PLANES - 1]
    # The target square is e3 (row 5, column 4).
    assert en_passant_plane[5][4] == 1.0
    assert en_passant_plane.sum() == 1.0


def test_castling_planes_clear_when_rights_lost() -> None:
    """Losing castling rights should clear the matching castling plane."""
    gamestate = GameState()
    gamestate.current_castling_rights.white_king_side = False
    planes = encoding.encode_gamestate(gamestate)
    # The white king-side plane is the first castling plane.
    white_king_side_plane = planes[encoding.NUMBER_OF_PIECE_PLANES + 1]
    assert white_king_side_plane.sum() == 0.0
