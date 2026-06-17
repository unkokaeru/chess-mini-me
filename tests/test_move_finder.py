"""Tests for the move finder's search and evaluation."""

from __future__ import annotations

from chess_mini_me import constants, move_finder
from chess_mini_me.engine import CastleRights, GameState


def empty_game() -> GameState:
    """Return a game state with an empty board and no castling rights.

    Returns:
        A blank ``GameState`` ready for pieces to be placed.
    """
    gamestate = GameState()
    for row in range(constants.BOARD_DIMENSION):
        for column in range(constants.BOARD_DIMENSION):
            gamestate.board[row][column] = constants.EMPTY_SQUARE
    rights = CastleRights(False, False, False, False)
    gamestate.current_castling_rights = rights
    gamestate.castle_rights_log = [rights.copy()]
    return gamestate


def test_starting_position_is_balanced() -> None:
    """The evaluation of the starting position should be exactly level."""
    assert move_finder.evaluate_board(GameState()) == 0


def test_finds_free_capture() -> None:
    """The move finder should capture an undefended queen when it can."""
    gamestate = empty_game()
    gamestate.board[7][4] = "wK"
    gamestate.board[0][4] = "bK"
    gamestate.board[7][0] = "wR"
    gamestate.board[0][0] = "bQ"
    gamestate.white_king_location = (7, 4)
    gamestate.black_king_location = (0, 4)

    best_move = move_finder.find_best_move(
        gamestate, gamestate.get_valid_moves(), depth=2
    )
    assert best_move is not None
    assert (best_move.end_row, best_move.end_column) == (0, 0)


def test_finds_mate_in_one() -> None:
    """The move finder should choose the move that delivers checkmate."""
    gamestate = empty_game()
    for square, piece in {
        (0, 6): "bK",
        (1, 5): "bP",
        (1, 6): "bP",
        (1, 7): "bP",
        (7, 4): "wR",
        (7, 7): "wK",
    }.items():
        gamestate.board[square[0]][square[1]] = piece
    gamestate.white_king_location = (7, 7)
    gamestate.black_king_location = (0, 6)

    best_move = move_finder.find_best_move(
        gamestate, gamestate.get_valid_moves(), depth=3
    )
    assert best_move is not None
    gamestate.make_move(best_move)
    gamestate.get_valid_moves()
    assert gamestate.checkmate is True


def test_returns_none_without_moves() -> None:
    """With no legal moves the move finder should return ``None``."""
    assert move_finder.find_best_move(GameState(), [], depth=2) is None
