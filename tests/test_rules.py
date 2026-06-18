"""Tests for the drawing rules and resignation."""

from __future__ import annotations

from chess_mini_me import constants
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


def _bare_board(pieces, white_king, black_king) -> GameState:
    """Return a game state with only the given pieces on the board.

    Args:
        pieces: A mapping from (row, column) to piece code.
        white_king: The white king's square.
        black_king: The black king's square.

    Returns:
        The constructed game state with castling rights cleared.
    """
    gamestate = GameState()
    for row in range(constants.BOARD_DIMENSION):
        for column in range(constants.BOARD_DIMENSION):
            gamestate.board[row][column] = constants.EMPTY_SQUARE
    for square, piece in pieces.items():
        gamestate.board[square[0]][square[1]] = piece
    gamestate.white_king_location = white_king
    gamestate.black_king_location = black_king
    rights = CastleRights(False, False, False, False)
    gamestate.current_castling_rights = rights
    gamestate.castle_rights_log = [rights.copy()]
    return gamestate


def test_threefold_repetition() -> None:
    """Shuffling knights back to the same position three times is a draw."""
    gamestate = GameState()
    cycle = [((7, 6), (5, 5)), ((0, 6), (2, 5)), ((5, 5), (7, 6)), ((2, 5), (0, 6))]
    _play(gamestate, cycle * 2)
    gamestate.get_valid_moves()
    assert gamestate.draw is True
    assert gamestate.draw_reason == "threefold repetition"


def test_fifty_move_rule() -> None:
    """Reaching a hundred plies without a capture or pawn move is a draw."""
    gamestate = GameState()
    gamestate.halfmove_clock = 100
    gamestate.get_valid_moves()
    assert gamestate.draw is True
    assert gamestate.draw_reason == "the fifty-move rule"


def test_insufficient_material_king_versus_king() -> None:
    """King against king is a draw by insufficient material."""
    gamestate = _bare_board({(7, 4): "wK", (0, 4): "bK"}, (7, 4), (0, 4))
    gamestate.get_valid_moves()
    assert gamestate.draw is True
    assert gamestate.draw_reason == "insufficient material"


def test_insufficient_material_same_colour_bishops() -> None:
    """King and bishop against king and bishop draw on same-coloured bishops."""
    same_colour = _bare_board(
        {(7, 4): "wK", (0, 4): "bK", (7, 2): "wB", (0, 1): "bB"}, (7, 4), (0, 4)
    )
    same_colour.get_valid_moves()
    assert same_colour.draw is True

    opposite_colour = _bare_board(
        {(7, 4): "wK", (0, 4): "bK", (7, 2): "wB", (0, 2): "bB"}, (7, 4), (0, 4)
    )
    opposite_colour.get_valid_moves()
    assert opposite_colour.draw is False


def test_queen_is_sufficient_material() -> None:
    """A queen on the board is enough material, so it is not a draw."""
    gamestate = _bare_board(
        {(7, 4): "wK", (0, 4): "bK", (7, 3): "wQ"}, (7, 4), (0, 4)
    )
    gamestate.get_valid_moves()
    assert gamestate.draw is False


def test_resignation_sets_the_result() -> None:
    """Resigning ends the game and awards it to the opponent."""
    gamestate = GameState()
    gamestate.resign(constants.WHITE)
    assert gamestate.is_game_over() is True
    assert gamestate.result_string() == "0-1"
    assert "resigned" in gamestate.outcome_description()


def test_undo_restores_repetition_and_clock_state() -> None:
    """Undoing a move restores the half-move clock and repetition counts."""
    gamestate = GameState()
    initial_key = gamestate._position_key()
    _play(gamestate, [((6, 4), (4, 4))])
    gamestate.undo_move()
    assert gamestate.position_counts.get(initial_key) == 1
    assert gamestate.halfmove_clock == 0
