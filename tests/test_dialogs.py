"""Tests for the native dialog helpers and their fallbacks."""

from __future__ import annotations

import sys

from chess_mini_me import dialogs


def test_save_path_falls_back_without_a_toolkit(tmp_path, monkeypatch) -> None:
    """Without a GUI toolkit, a default path inside the folder is returned."""
    # Make importing tkinter fail, simulating a headless environment.
    monkeypatch.setitem(sys.modules, "tkinter", None)
    path, cancelled = dialogs.ask_save_pgn_path("game.pgn", tmp_path)
    assert cancelled is False
    assert path is not None
    assert path.parent == tmp_path
    assert path.name == "game.pgn"
