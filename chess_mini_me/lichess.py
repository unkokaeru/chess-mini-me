"""Build a Mini-Me from a player's Lichess game history.

This module downloads a player's games from the public Lichess API, replays
them with python-chess, analyses the player's style with pandas, and turns the
player's own moves into training examples for the Mini-Me network.

Only the target player's moves are used: the Mini-Me copies the requested
player, not their opponents.
"""

from __future__ import annotations

import io
from typing import Iterator

import numpy as np
import pandas as pd

from chess_mini_me import constants, encoding
from chess_mini_me.training import (
    BLACK_WIN_VALUE,
    DRAW_VALUE,
    WHITE_WIN_VALUE,
    TrainingExamples,
)

LICHESS_GAMES_URL = "https://lichess.org/api/games/user/{username}"

# Map a python-chess promotion piece type to this engine's letter.
_PROMOTION_LETTERS = {2: constants.KNIGHT, 3: constants.BISHOP, 4: constants.ROOK, 5: constants.QUEEN}


def _import_chess():
    """Import and return python-chess, with a helpful error if missing.

    Returns:
        The imported ``chess`` module.

    Raises:
        ImportError: If python-chess is not installed.
    """
    try:
        import chess
        import chess.pgn  # noqa: F401 - ensures the submodule is available

        return chess
    except ImportError as error:  # pragma: no cover - only without python-chess
        raise ImportError(
            "Importing from Lichess requires python-chess. Install it with "
            "'pip install chess'."
        ) from error


def _import_requests():
    """Import and return requests, with a helpful error if missing.

    Returns:
        The imported ``requests`` module.

    Raises:
        ImportError: If requests is not installed.
    """
    try:
        import requests

        return requests
    except ImportError as error:  # pragma: no cover - only without requests
        raise ImportError(
            "Downloading from Lichess requires the requests library. Install "
            "it with 'pip install requests'."
        ) from error


def download_games(
    username: str,
    max_games: int = 200,
    rated_only: bool = True,
    token: str | None = None,
    timeout_seconds: float = 30.0,
) -> str:
    """Download a player's games from Lichess as PGN text.

    Args:
        username: The Lichess username to download.
        max_games: The most recent number of games to fetch.
        rated_only: Whether to fetch only rated games.
        token: An optional Lichess API token, which raises the rate limit.
        timeout_seconds: The network timeout.

    Returns:
        The downloaded games as a single PGN string.

    Raises:
        ValueError: If the username is unknown or the request fails.
    """
    requests = _import_requests()
    parameters = {
        "max": max_games,
        "rated": str(rated_only).lower(),
        "clocks": "false",
        "evals": "false",
        "opening": "true",
    }
    headers = {"Accept": "application/x-chess-pgn"}
    if token:
        headers["Authorization"] = f"Bearer {token}"

    response = requests.get(
        LICHESS_GAMES_URL.format(username=username),
        params=parameters,
        headers=headers,
        timeout=timeout_seconds,
    )
    if response.status_code == 404:
        raise ValueError(f"Lichess has no player named '{username}'.")
    if not response.ok:
        raise ValueError(
            f"Lichess request failed with status {response.status_code}."
        )
    return response.text


def iterate_games(pgn_text: str) -> Iterator["object"]:
    """Yield each game parsed from a PGN string.

    Args:
        pgn_text: The PGN text holding one or more games.

    Yields:
        Each parsed python-chess game.
    """
    chess = _import_chess()
    stream = io.StringIO(pgn_text)
    while True:
        game = chess.pgn.read_game(stream)
        if game is None:
            break
        yield game


def _player_colour(game: "object", username: str) -> str | None:
    """Return the colour the target player had in a game.

    Args:
        game: A python-chess game.
        username: The target player's username.

    Returns:
        ``"w"`` or ``"b"`` for the player's colour, or ``None`` if the player
        did not feature in the game.
    """
    lowered = username.lower()
    if game.headers.get("White", "").lower() == lowered:
        return constants.WHITE
    if game.headers.get("Black", "").lower() == lowered:
        return constants.BLACK
    return None


def _result_value(game: "object") -> float:
    """Return the game result from White's perspective.

    Args:
        game: A python-chess game.

    Returns:
        ``1`` for a White win, ``-1`` for a Black win and ``0`` otherwise.
    """
    result = game.headers.get("Result", "*")
    if result == "1-0":
        return WHITE_WIN_VALUE
    if result == "0-1":
        return BLACK_WIN_VALUE
    return DRAW_VALUE


def _board_to_matrix(board: "object") -> list[list[str]]:
    """Convert a python-chess board into this engine's board matrix.

    Args:
        board: A python-chess board.

    Returns:
        An 8x8 matrix in ``"wP"`` style notation.
    """
    chess = _import_chess()
    matrix = [
        [constants.EMPTY_SQUARE] * constants.BOARD_DIMENSION
        for _ in range(constants.BOARD_DIMENSION)
    ]
    for square in chess.SQUARES:
        piece = board.piece_at(square)
        if piece is None:
            continue
        symbol = piece.symbol()
        colour = constants.WHITE if symbol.isupper() else constants.BLACK
        row = constants.BOARD_DIMENSION - 1 - chess.square_rank(square)
        column = chess.square_file(square)
        matrix[row][column] = colour + symbol.upper()
    return matrix


def _encode_board(board: "object") -> np.ndarray:
    """Encode a python-chess board into the network's feature planes.

    Args:
        board: A python-chess board.

    Returns:
        The encoded position, matching :func:`encoding.encode_planes`.
    """
    chess = _import_chess()
    matrix = _board_to_matrix(board)
    castling_rights = (
        board.has_kingside_castling_rights(chess.WHITE),
        board.has_queenside_castling_rights(chess.WHITE),
        board.has_kingside_castling_rights(chess.BLACK),
        board.has_queenside_castling_rights(chess.BLACK),
    )
    en_passant_square = None
    if board.ep_square is not None:
        en_passant_square = (
            constants.BOARD_DIMENSION - 1 - chess.square_rank(board.ep_square),
            chess.square_file(board.ep_square),
        )
    return encoding.encode_planes(
        matrix,
        board.turn == chess.WHITE,
        castling_rights,
        en_passant_square,
    )


def _move_to_indices(move: "object") -> tuple[int, int]:
    """Return the policy index and promotion class of a python-chess move.

    Args:
        move: A python-chess move.

    Returns:
        A tuple ``(policy_index, promotion_index)`` where the promotion index
        is ``-1`` for non-promotions.
    """
    chess = _import_chess()
    start_row = constants.BOARD_DIMENSION - 1 - chess.square_rank(move.from_square)
    start_column = chess.square_file(move.from_square)
    end_row = constants.BOARD_DIMENSION - 1 - chess.square_rank(move.to_square)
    end_column = chess.square_file(move.to_square)
    policy_index = encoding.move_to_policy_index(
        start_row, start_column, end_row, end_column
    )
    promotion_index = -1
    if move.promotion is not None:
        promotion_index = encoding.PROMOTION_TO_INDEX[
            _PROMOTION_LETTERS[move.promotion]
        ]
    return policy_index, promotion_index


def extract_examples(pgn_text: str, username: str) -> TrainingExamples:
    """Build training examples from a player's own moves in their games.

    Args:
        pgn_text: The PGN text of the player's games.
        username: The player to imitate.

    Returns:
        The training examples drawn from the player's moves.
    """
    chess = _import_chess()
    planes_list: list[np.ndarray] = []
    policy_indices: list[int] = []
    promotion_indices: list[int] = []
    values: list[float] = []

    for game in iterate_games(pgn_text):
        colour = _player_colour(game, username)
        if colour is None:
            continue
        result_value = _result_value(game)
        player_is_white = colour == constants.WHITE

        board = game.board()
        for move in game.mainline_moves():
            it_is_players_move = (board.turn == chess.WHITE) == player_is_white
            if it_is_players_move:
                planes_list.append(_encode_board(board))
                policy_index, promotion_index = _move_to_indices(move)
                policy_indices.append(policy_index)
                promotion_indices.append(promotion_index)
                values.append(result_value)
            board.push(move)

    if not planes_list:
        return TrainingExamples.empty()
    return TrainingExamples(
        planes=np.stack(planes_list).astype(np.float32),
        policy_indices=np.array(policy_indices, dtype=np.int64),
        promotion_indices=np.array(promotion_indices, dtype=np.int64),
        values=np.array(values, dtype=np.float32),
    )


def analyse_games(pgn_text: str, username: str) -> pd.DataFrame:
    """Summarise a player's style as a tidy table of their games.

    Args:
        pgn_text: The PGN text of the player's games.
        username: The player to analyse.

    Returns:
        A DataFrame with one row per game the player featured in, holding the
        colour played, the result for the player, the opening, the number of
        plies and the player's first move.
    """
    chess = _import_chess()
    records: list[dict[str, object]] = []
    for game in iterate_games(pgn_text):
        colour = _player_colour(game, username)
        if colour is None:
            continue
        result_value = _result_value(game)
        player_is_white = colour == constants.WHITE
        if result_value == DRAW_VALUE:
            outcome = "draw"
        elif (result_value == WHITE_WIN_VALUE) == player_is_white:
            outcome = "win"
        else:
            outcome = "loss"

        moves = list(game.mainline_moves())
        first_move = ""
        if moves:
            first_move = game.board().san(moves[0])

        records.append(
            {
                "colour": "white" if player_is_white else "black",
                "outcome": outcome,
                "opening": game.headers.get("Opening", "Unknown"),
                "eco": game.headers.get("ECO", "?"),
                "plies": len(moves),
                "first_move": first_move,
            }
        )
    return pd.DataFrame.from_records(records)


def summarise_style(games: pd.DataFrame) -> str:
    """Produce a short, human-readable summary of a player's style.

    Args:
        games: The DataFrame returned by :func:`analyse_games`.

    Returns:
        A multi-line summary string.
    """
    if games.empty:
        return "No games were found for this player."

    lines = [f"Analysed {len(games)} games."]
    colour_counts = games["colour"].value_counts()
    lines.append(
        "Colours: "
        + ", ".join(f"{count} as {colour}" for colour, count in colour_counts.items())
    )

    outcome_counts = games["outcome"].value_counts()
    lines.append(
        "Results: "
        + ", ".join(f"{count} {outcome}" for outcome, count in outcome_counts.items())
    )
    lines.append(f"Average game length: {games['plies'].mean():.0f} plies.")

    top_openings = games["opening"].value_counts().head(5)
    lines.append("Favourite openings:")
    for opening, count in top_openings.items():
        lines.append(f"  {count:>3}  {opening}")

    top_first_moves = (
        games.loc[games["first_move"] != "", "first_move"].value_counts().head(5)
    )
    if not top_first_moves.empty:
        lines.append("Most common first moves: " + ", ".join(
            f"{move} ({count})" for move, count in top_first_moves.items()
        ))
    return "\n".join(lines)
