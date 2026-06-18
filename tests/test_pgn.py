"""Tests for PGN export and SAN generation."""

from __future__ import annotations

import io

import pytest

from chess_mini_me import constants, pgn
from chess_mini_me.engine import CastleRights, GameState, Move


def _play(gamestate: GameState, coordinates) -> None:
    """Play a sequence of (start, end) coordinate moves.

    Args:
        gamestate: The game state to advance.
        coordinates: An iterable of (start_square, end_square) pairs.
    """
    for start_square, end_square in coordinates:
        intended = Move(start_square, end_square, gamestate.board)
        for legal_move in gamestate.get_valid_moves():
            if legal_move == intended:
                gamestate.make_move(legal_move)
                break
        else:
            raise AssertionError(f"illegal move {start_square}->{end_square}")


def test_scholars_mate_movetext() -> None:
    """Scholar's mate should produce the expected SAN and result."""
    gamestate = GameState()
    _play(
        gamestate,
        [((6, 4), (4, 4)), ((1, 4), (3, 4)), ((7, 5), (4, 2)), ((0, 1), (2, 2)),
         ((7, 3), (3, 7)), ((0, 6), (2, 5)), ((3, 7), (1, 5))],
    )
    gamestate.get_valid_moves()
    text = pgn.game_to_pgn(gamestate, "Alice", "Bob")
    assert '[White "Alice"]' in text
    assert '[Result "1-0"]' in text
    assert "1. e4 e5 2. Bc4 Nc6 3. Qh5 Nf6 4. Qxf7#" in text


def test_promotion_san_directly() -> None:
    """A promoting pawn move should be written with the chosen piece."""
    gamestate = GameState()
    for row in range(constants.BOARD_DIMENSION):
        for column in range(constants.BOARD_DIMENSION):
            gamestate.board[row][column] = constants.EMPTY_SQUARE
    gamestate.board[1][7] = "wP"
    gamestate.board[7][0] = "wK"
    gamestate.board[0][4] = "bK"
    gamestate.white_king_location = (7, 0)
    gamestate.black_king_location = (0, 4)
    gamestate.current_castling_rights = CastleRights(False, False, False, False)

    legal_moves = gamestate.get_valid_moves()
    promotion_move = next(move for move in legal_moves if move.is_pawn_promotion)
    promotion_move.promotion_choice = constants.ROOK
    assert pgn.move_to_san(promotion_move, legal_moves) == "h8=R"


def test_disambiguation_san() -> None:
    """Two knights able to reach a square should be disambiguated by file."""
    gamestate = GameState()
    for row in range(constants.BOARD_DIMENSION):
        for column in range(constants.BOARD_DIMENSION):
            gamestate.board[row][column] = constants.EMPTY_SQUARE
    # White knights on b1 and f1 can both reach d2.
    gamestate.board[7][1] = "wN"
    gamestate.board[7][5] = "wN"
    gamestate.board[7][4] = "wK"
    gamestate.board[0][4] = "bK"
    gamestate.white_king_location = (7, 4)
    gamestate.black_king_location = (0, 4)
    gamestate.current_castling_rights = CastleRights(False, False, False, False)

    legal_moves = gamestate.get_valid_moves()
    knight_to_d2 = next(
        move
        for move in legal_moves
        if (move.start_row, move.start_column) == (7, 1)
        and (move.end_row, move.end_column) == (6, 3)
    )
    assert pgn.move_to_san(knight_to_d2, legal_moves) == "Nbd2"


def test_en_passant_and_round_trip() -> None:
    """En passant should be written as a pawn capture and replay correctly."""
    chess = pytest.importorskip("chess")
    import chess.pgn

    gamestate = GameState()
    _play(
        gamestate,
        [((6, 4), (4, 4)), ((1, 0), (2, 0)), ((4, 4), (3, 4)), ((1, 3), (3, 3)),
         ((3, 4), (2, 3))],
    )
    text = pgn.game_to_pgn(gamestate)
    assert "exd6" in text

    game = chess.pgn.read_game(io.StringIO(text))
    board = chess.Board()
    for move in game.mainline_moves():
        assert move in board.legal_moves
        board.push(move)
