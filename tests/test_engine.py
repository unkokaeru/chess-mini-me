"""Tests for the chess engine's rules and move generation."""

from __future__ import annotations

import pytest

from chess_mini_me import constants
from chess_mini_me.engine import CastleRights, GameState, Move


def perft(gamestate: GameState, depth: int) -> int:
    """Count the leaf nodes of the move tree to a fixed depth.

    Perft (performance test) is the standard way to check a move generator: the
    node counts from the starting position are well known, so any error in
    move generation shows up as a wrong total.

    Args:
        gamestate: The position to search from.
        depth: The number of plies to search.

    Returns:
        The number of distinct legal move sequences of the given length.
    """
    if depth == 0:
        return 1
    total = 0
    for move in gamestate.get_valid_moves():
        gamestate.make_move(move)
        total += perft(gamestate, depth - 1)
        gamestate.undo_move()
    return total


def play_moves(gamestate: GameState, coordinates: list[tuple[tuple[int, int], tuple[int, int]]]) -> None:
    """Play a sequence of moves given by their start and end coordinates.

    Args:
        gamestate: The game state to advance.
        coordinates: A list of (start_square, end_square) pairs.
    """
    for start_square, end_square in coordinates:
        intended_move = Move(start_square, end_square, gamestate.board)
        for legal_move in gamestate.get_valid_moves():
            if legal_move == intended_move:
                gamestate.make_move(legal_move)
                break
        else:
            raise AssertionError(f"Illegal move in sequence: {start_square}->{end_square}")


@pytest.mark.parametrize(
    "depth, expected_nodes",
    [(1, 20), (2, 400), (3, 8902)],
)
def test_perft_from_starting_position(depth: int, expected_nodes: int) -> None:
    """The move generator should match the known perft node counts."""
    assert perft(GameState(), depth) == expected_nodes


def test_fools_mate_is_checkmate() -> None:
    """The fastest checkmate should be reported as checkmate, not stalemate."""
    gamestate = GameState()
    play_moves(
        gamestate,
        [((6, 5), (5, 5)), ((1, 4), (3, 4)), ((6, 6), (4, 6)), ((0, 3), (4, 7))],
    )
    gamestate.get_valid_moves()
    assert gamestate.checkmate is True
    assert gamestate.stalemate is False


def test_back_rank_mate_leaves_no_legal_moves() -> None:
    """A back-rank mate must leave the defending side with no legal escape."""
    gamestate = empty_game()
    place(gamestate, {(0, 6): "bK", (1, 5): "bP", (1, 6): "bP", (1, 7): "bP",
                      (7, 4): "wR", (7, 7): "wK"}, black_king=(0, 6), white_king=(7, 7))
    play_moves(gamestate, [((7, 4), (0, 4))])
    assert gamestate.get_valid_moves() == []
    assert gamestate.checkmate is True


def test_stalemate_is_detected() -> None:
    """A position with no legal move but no check should be a stalemate."""
    gamestate = empty_game()
    place(gamestate, {(0, 7): "bK", (1, 5): "wK", (2, 6): "wQ"},
          black_king=(0, 7), white_king=(1, 5))
    gamestate.white_to_move = False
    gamestate.get_valid_moves()
    assert gamestate.stalemate is True
    assert gamestate.checkmate is False


def test_en_passant_capture_removes_the_pawn() -> None:
    """Capturing en passant should remove the pawn that advanced two squares."""
    gamestate = GameState()
    play_moves(
        gamestate,
        [((6, 4), (4, 4)), ((1, 0), (2, 0)), ((4, 4), (3, 4)), ((1, 3), (3, 3))],
    )
    # White pawn on e5 captures the d5 pawn en passant, landing on d6.
    play_moves(gamestate, [((3, 4), (2, 3))])
    assert gamestate.board[2][3] == "wP"
    assert gamestate.board[3][3] == constants.EMPTY_SQUARE
    assert gamestate.board[3][4] == constants.EMPTY_SQUARE


def test_king_side_castle_moves_king_and_rook() -> None:
    """Castling king-side should move both the king and the rook."""
    gamestate = GameState()
    gamestate.board[7][5] = constants.EMPTY_SQUARE
    gamestate.board[7][6] = constants.EMPTY_SQUARE
    castle_move = next(
        move
        for move in gamestate.get_valid_moves()
        if move.is_castle_move and move.end_column == 6
    )
    gamestate.make_move(castle_move)
    assert gamestate.board[7][6] == "wK"
    assert gamestate.board[7][5] == "wR"
    assert gamestate.board[7][7] == constants.EMPTY_SQUARE


def test_pawn_promotion_uses_chosen_piece() -> None:
    """A promoting pawn should become the requested piece."""
    gamestate = empty_game()
    place(gamestate, {(1, 0): "wP", (7, 4): "wK", (0, 7): "bK"},
          white_king=(7, 4), black_king=(0, 7))
    promotion_move = next(
        move for move in gamestate.get_valid_moves() if move.is_pawn_promotion
    )
    gamestate.make_move(promotion_move, promotion_piece=constants.ROOK)
    assert gamestate.board[0][0] == "wR"


def test_undo_restores_the_previous_position() -> None:
    """Undoing a capture should restore the board and the side to move."""
    gamestate = GameState()
    play_moves(gamestate, [((6, 4), (4, 4)), ((1, 3), (3, 3))])
    board_before = [row[:] for row in gamestate.board]
    white_to_move_before = gamestate.white_to_move
    play_moves(gamestate, [((4, 4), (3, 3))])  # exd5 capture
    gamestate.undo_move()
    assert gamestate.board == board_before
    assert gamestate.white_to_move == white_to_move_before


# ---------------------------------------------------------------------------
# Helpers for building tailored positions
# ---------------------------------------------------------------------------


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


def place(
    gamestate: GameState,
    pieces: dict[tuple[int, int], str],
    white_king: tuple[int, int],
    black_king: tuple[int, int],
) -> None:
    """Place pieces on the board and record the king locations.

    Args:
        gamestate: The game state to populate.
        pieces: A mapping from (row, column) to piece code.
        white_king: The white king's location.
        black_king: The black king's location.
    """
    for square, piece in pieces.items():
        gamestate.board[square[0]][square[1]] = piece
    gamestate.white_king_location = white_king
    gamestate.black_king_location = black_king
