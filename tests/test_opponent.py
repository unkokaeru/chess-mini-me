"""Tests for the opponent controller and the move recorder.

These tests do not require PyTorch: with no trained Mini-Me the controller
falls back to the classical search, which is the behaviour exercised here.
"""

from __future__ import annotations

from chess_mini_me import opponent
from chess_mini_me.engine import GameState
from chess_mini_me.training import StyleRecorder, StyleStore


def test_player_type_cycle() -> None:
    """Cycling through player types should visit each and wrap around."""
    assert opponent.next_player_type(opponent.PLAYER_HUMAN) == opponent.PLAYER_COMPUTER
    assert opponent.next_player_type(opponent.PLAYER_COMPUTER) == opponent.PLAYER_MINI_ME
    assert opponent.next_player_type(opponent.PLAYER_MINI_ME) == opponent.PLAYER_HUMAN


def test_controller_falls_back_without_a_mini_me(tmp_path) -> None:
    """Without a trained Mini-Me, the controller should still return a move."""
    store = StyleStore(tmp_path)
    controller = opponent.OpponentController(store)
    assert controller.is_mini_me_ready() is False

    gamestate = GameState()
    legal_moves = gamestate.get_valid_moves()
    for player_type in (opponent.PLAYER_MINI_ME, opponent.PLAYER_COMPUTER):
        move, promotion_piece = controller.choose_move(
            player_type, gamestate, legal_moves
        )
        assert move in legal_moves
        assert promotion_piece == "Q"


def test_recorder_builds_examples() -> None:
    """The recorder should turn recorded moves into labelled examples."""
    gamestate = GameState()
    recorder = StyleRecorder()
    first_move = gamestate.get_valid_moves()[0]
    recorder.record(gamestate, first_move, "Q")
    assert recorder.has_moves()

    examples = recorder.finalise(result_value=1.0)
    assert len(examples) == 1
    assert examples.planes.shape[1:] == (18, 8, 8)
    assert examples.values[0] == 1.0
