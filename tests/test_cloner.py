"""Tests for the Mini-Me network, trainer and agent.

These tests need PyTorch and are skipped where it is not installed. A small
network and a handful of self-play positions keep them fast.
"""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("torch")

from chess_mini_me import cloner, encoding  # noqa: E402
from chess_mini_me.engine import GameState  # noqa: E402
from chess_mini_me.training import (  # noqa: E402
    StyleStore,
    TrainingExamples,
    learn_from_examples,
    train_network,
)

SMALL_CHANNELS = 8
SMALL_BLOCKS = 1


def _self_play_examples(number_of_positions: int) -> TrainingExamples:
    """Build training examples by playing random legal moves.

    Args:
        number_of_positions: How many positions to record.

    Returns:
        A set of training examples with arbitrary but legal target moves.
    """
    planes_list = []
    policy_indices = []
    gamestate = GameState()
    for _ in range(number_of_positions):
        legal_moves = gamestate.get_valid_moves()
        if not legal_moves:
            gamestate = GameState()
            continue
        move = legal_moves[np.random.randint(len(legal_moves))]
        planes_list.append(encoding.encode_gamestate(gamestate))
        policy_indices.append(
            encoding.move_to_policy_index(
                move.start_row, move.start_column, move.end_row, move.end_column
            )
        )
        gamestate.make_move(move)
    count = len(planes_list)
    return TrainingExamples(
        planes=np.stack(planes_list).astype(np.float32),
        policy_indices=np.array(policy_indices, dtype=np.int64),
        promotion_indices=np.full((count,), -1, dtype=np.int64),
        values=np.zeros((count,), dtype=np.float32),
    )


def test_network_output_shapes() -> None:
    """The network should produce policy, promotion and value outputs."""
    import torch

    model = cloner.StyleNetwork(channels=SMALL_CHANNELS, residual_blocks=SMALL_BLOCKS)
    batch = torch.zeros((3, encoding.NUMBER_OF_INPUT_PLANES, 8, 8))
    policy_logits, promotion_logits, value = model(batch)
    assert policy_logits.shape == (3, encoding.POLICY_SIZE)
    assert promotion_logits.shape == (3, encoding.NUMBER_OF_PROMOTION_CLASSES)
    assert value.shape == (3, 1)


def test_training_runs_and_reports_losses() -> None:
    """Training should run and report a finite loss for each epoch."""
    examples = _self_play_examples(40)
    model = cloner.StyleNetwork(channels=SMALL_CHANNELS, residual_blocks=SMALL_BLOCKS)
    losses = train_network(model, examples, epochs=3, batch_size=16)
    assert len(losses) == 3
    assert all(np.isfinite(loss) for loss in losses)


def test_mini_me_selects_a_legal_move() -> None:
    """A Mini-Me wrapping a network should always choose a legal move."""
    model = cloner.StyleNetwork(channels=SMALL_CHANNELS, residual_blocks=SMALL_BLOCKS)
    mini_me = cloner.MiniMe(model, temperature=0.5)
    gamestate = GameState()
    legal_moves = gamestate.get_valid_moves()
    move, promotion_piece = mini_me.select_move(gamestate, legal_moves)
    assert move in legal_moves
    assert promotion_piece in encoding.PROMOTION_PIECES


def test_learn_and_reload_round_trip(tmp_path) -> None:
    """Learning should save a model that can be loaded back as a Mini-Me."""
    store = StyleStore(tmp_path)
    examples = _self_play_examples(30)
    learn_from_examples(
        store, examples, epochs=2, channels=SMALL_CHANNELS, residual_blocks=SMALL_BLOCKS
    )
    assert store.model_path.exists()
    assert store.dataset_path.exists()

    mini_me = cloner.load_mini_me(store.model_path)
    assert mini_me is not None
    gamestate = GameState()
    move, _ = mini_me.select_move(gamestate, gamestate.get_valid_moves())
    assert move in gamestate.get_valid_moves()
