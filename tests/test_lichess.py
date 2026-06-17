"""Tests for the Lichess import pipeline.

These tests build their own PGN with python-chess, so they need no network.
They are skipped if python-chess is not installed.
"""

from __future__ import annotations

import pytest

chess = pytest.importorskip("chess")
import chess.pgn  # noqa: E402

from chess_mini_me import lichess  # noqa: E402


def _build_pgn() -> str:
    """Build a two-game PGN for a player named ``target``.

    Returns:
        The PGN text of the two synthetic games.
    """

    def make_game(white, black, ucis, result, opening):
        game = chess.pgn.Game()
        game.headers["White"] = white
        game.headers["Black"] = black
        game.headers["Result"] = result
        game.headers["Opening"] = opening
        node = game
        board = chess.Board()
        for uci in ucis:
            move = chess.Move.from_uci(uci)
            board.push(move)
            node = node.add_variation(move)
        return game

    first = make_game(
        "target", "rival", ["e2e4", "e7e5", "g1f3", "b8c6", "f1b5"], "1-0", "Ruy Lopez"
    )
    second = make_game(
        "rival", "target", ["d2d4", "g8f6", "c2c4", "e7e6"], "0-1", "Indian Defence"
    )
    return f"{first}\n\n{second}"


def test_extract_examples_uses_only_the_target_player() -> None:
    """Only the target player's own moves should become examples."""
    examples = lichess.extract_examples(_build_pgn(), "target")
    # Three moves as White in game one, two as Black in game two.
    assert len(examples) == 5
    assert examples.planes.shape[1:] == (18, 8, 8)


def test_analyse_games_summarises_style() -> None:
    """The analysis should describe the games the player featured in."""
    games = lichess.analyse_games(_build_pgn(), "target")
    assert len(games) == 2
    assert set(games["colour"]) == {"white", "black"}
    summary = lichess.summarise_style(games)
    assert "Analysed 2 games" in summary
    assert "Ruy Lopez" in summary
