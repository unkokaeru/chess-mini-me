"""Export a finished or in-progress game to PGN.

PGN (Portable Game Notation) is the standard text format for recording chess
games. The moves are written in Standard Algebraic Notation (SAN), which this
module generates directly from the engine's move log, including the
disambiguation and check or mate markers SAN requires. No third-party library
is needed, so saving a game works with only the core dependencies installed.
"""

from __future__ import annotations

import datetime
import pathlib

from chess_mini_me import constants
from chess_mini_me.engine import GameState, Move


def _square_name(row: int, column: int) -> str:
    """Return the file-and-rank name of a square, such as ``e4``.

    Args:
        row: The board row (0 is the eighth rank).
        column: The board column (0 is the a-file).

    Returns:
        The square name.
    """
    return constants.COLUMNS_TO_FILES[column] + str(
        constants.BOARD_DIMENSION - row
    )


def _disambiguation(
    move: Move, legal_moves: list[Move]
) -> str:
    """Return the SAN disambiguation for a piece move, if any is needed.

    When two identical pieces could move to the same square, SAN adds the
    origin file, rank or both to make the move unambiguous.

    Args:
        move: The move being described.
        legal_moves: All legal moves in the position before the move.

    Returns:
        The disambiguation string, which may be empty.
    """
    rivals = [
        other
        for other in legal_moves
        if other.piece_moved == move.piece_moved
        and (other.end_row, other.end_column) == (move.end_row, move.end_column)
        and (other.start_row, other.start_column)
        != (move.start_row, move.start_column)
    ]
    if not rivals:
        return ""

    shares_file = any(rival.start_column == move.start_column for rival in rivals)
    shares_rank = any(rival.start_row == move.start_row for rival in rivals)
    if not shares_file:
        return constants.COLUMNS_TO_FILES[move.start_column]
    if not shares_rank:
        return str(constants.BOARD_DIMENSION - move.start_row)
    return constants.COLUMNS_TO_FILES[move.start_column] + str(
        constants.BOARD_DIMENSION - move.start_row
    )


def move_to_san(move: Move, legal_moves: list[Move]) -> str:
    """Return the Standard Algebraic Notation for a move, without suffixes.

    The check or checkmate suffix is added by the caller, which knows the
    resulting position.

    Args:
        move: The move to describe.
        legal_moves: All legal moves in the position before the move, used for
            disambiguation.

    Returns:
        The SAN string for the move, for example ``Nf3`` or ``exd5``.
    """
    if move.is_castle_move:
        return "O-O" if move.end_column > move.start_column else "O-O-O"

    destination = _square_name(move.end_row, move.end_column)
    is_capture = (
        move.piece_captured != constants.EMPTY_SQUARE or move.is_en_passant_move
    )

    if move.piece_moved[1] == constants.PAWN:
        notation = ""
        if is_capture:
            notation = constants.COLUMNS_TO_FILES[move.start_column] + "x"
        notation += destination
        if move.is_pawn_promotion:
            notation += "=" + (move.promotion_choice or constants.QUEEN)
        return notation

    notation = move.piece_moved[1]
    notation += _disambiguation(move, legal_moves)
    if is_capture:
        notation += "x"
    notation += destination
    return notation


def build_san_moves(gamestate: GameState) -> list[str]:
    """Return the game's moves in SAN, with check and mate markers.

    The move log is replayed on a fresh game so that each move's SAN can be
    generated with the correct disambiguation and check or mate suffix.

    Args:
        gamestate: The game whose moves should be converted.

    Returns:
        The list of SAN strings, one per ply.
    """
    replay = GameState()
    san_moves: list[str] = []
    for played_move in gamestate.move_log:
        legal_moves = replay.get_valid_moves()
        notation = move_to_san(played_move, legal_moves)

        promotion_piece = played_move.promotion_choice or constants.QUEEN
        replay.make_move(played_move, promotion_piece)
        replay.get_valid_moves()
        if replay.checkmate:
            notation += "#"
        elif replay.in_check:
            notation += "+"
        san_moves.append(notation)
    return san_moves


def _format_movetext(san_moves: list[str], result: str) -> str:
    """Lay out SAN moves as numbered PGN movetext.

    Args:
        san_moves: The moves in SAN.
        result: The game result token to append.

    Returns:
        The movetext, wrapped so that lines do not grow too long.
    """
    tokens: list[str] = []
    for index, san in enumerate(san_moves):
        if index % 2 == 0:
            tokens.append(f"{index // 2 + 1}.")
        tokens.append(san)
    tokens.append(result)

    lines: list[str] = []
    current = ""
    for token in tokens:
        candidate = token if not current else f"{current} {token}"
        if len(candidate) > 78:
            lines.append(current)
            current = token
        else:
            current = candidate
    if current:
        lines.append(current)
    return "\n".join(lines)


def game_to_pgn(
    gamestate: GameState,
    white_name: str = "White",
    black_name: str = "Black",
    event: str = "Chess Mini-Me game",
    site: str = "Chess Mini-Me",
    game_date: datetime.date | None = None,
) -> str:
    """Return a complete PGN document for a game.

    Args:
        gamestate: The game to export.
        white_name: The name to record for White.
        black_name: The name to record for Black.
        event: The event name for the header.
        site: The site name for the header.
        game_date: The date for the header; today's date is used if omitted.

    Returns:
        The PGN document as a string.
    """
    if game_date is None:
        game_date = datetime.date.today()

    result = gamestate.result_string()
    termination = gamestate.outcome_description() or "Game in progress"
    headers = [
        ("Event", event),
        ("Site", site),
        ("Date", game_date.strftime("%Y.%m.%d")),
        ("Round", "-"),
        ("White", white_name),
        ("Black", black_name),
        ("Result", result),
        ("Termination", termination),
    ]
    header_text = "\n".join(f'[{name} "{value}"]' for name, value in headers)
    movetext = _format_movetext(build_san_moves(gamestate), result)
    return f"{header_text}\n\n{movetext}\n"


def save_pgn(
    gamestate: GameState,
    path: pathlib.Path,
    white_name: str = "White",
    black_name: str = "Black",
) -> pathlib.Path:
    """Write a game to a PGN file, creating parent directories as needed.

    Args:
        gamestate: The game to export.
        path: The file path to write to.
        white_name: The name to record for White.
        black_name: The name to record for Black.

    Returns:
        The path that was written.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        game_to_pgn(gamestate, white_name=white_name, black_name=black_name),
        encoding="ascii",
    )
    return path
