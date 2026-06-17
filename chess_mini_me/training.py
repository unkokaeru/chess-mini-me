"""Training and persistence for the Mini-Me network.

This module turns recorded moves into training data, trains the network by
behavioural cloning, and stores the growing dataset and the latest model on
disk so that the Mini-Me keeps improving as it sees more of the player.

The dataset is kept as plain NumPy arrays in a compressed ``.npz`` file, which
is small, portable and easy to inspect. The model is stored as a PyTorch
checkpoint. Both live under a single data directory.
"""

from __future__ import annotations

import dataclasses
import pathlib
from typing import TYPE_CHECKING, Callable

import numpy as np

from chess_mini_me import cloner, constants, encoding

if TYPE_CHECKING:
    from torch import nn

    from chess_mini_me.engine import GameState, Move

# The most recent examples to keep on disk, so the dataset does not grow without
# bound during long-running online learning.
MAXIMUM_STORED_EXAMPLES = 200_000

# A draw, a White win and a Black win expressed as a value target.
DRAW_VALUE = 0.0
WHITE_WIN_VALUE = 1.0
BLACK_WIN_VALUE = -1.0


@dataclasses.dataclass
class TrainingExamples:
    """A batch of positions paired with the move the player made.

    Attributes:
        planes: Encoded positions, shaped ``(count, planes, 8, 8)``.
        policy_indices: The played move as a policy index, shaped ``(count,)``.
        promotion_indices: The promotion class, or ``-1`` for non-promotions,
            shaped ``(count,)``.
        values: The game result from White's perspective, shaped ``(count,)``.
    """

    planes: np.ndarray
    policy_indices: np.ndarray
    promotion_indices: np.ndarray
    values: np.ndarray

    @classmethod
    def empty(cls) -> "TrainingExamples":
        """Return an empty set of examples.

        Returns:
            An empty :class:`TrainingExamples`.
        """
        return cls(
            planes=np.zeros(
                (0, encoding.NUMBER_OF_INPUT_PLANES, constants.BOARD_DIMENSION,
                 constants.BOARD_DIMENSION),
                dtype=np.float32,
            ),
            policy_indices=np.zeros((0,), dtype=np.int64),
            promotion_indices=np.zeros((0,), dtype=np.int64),
            values=np.zeros((0,), dtype=np.float32),
        )

    def __len__(self) -> int:
        """Return the number of examples.

        Returns:
            The example count.
        """
        return int(self.planes.shape[0])

    def concatenate(self, other: "TrainingExamples") -> "TrainingExamples":
        """Return these examples followed by another set.

        Args:
            other: The examples to append.

        Returns:
            A new :class:`TrainingExamples` containing both sets.
        """
        return TrainingExamples(
            planes=np.concatenate([self.planes, other.planes]),
            policy_indices=np.concatenate(
                [self.policy_indices, other.policy_indices]
            ),
            promotion_indices=np.concatenate(
                [self.promotion_indices, other.promotion_indices]
            ),
            values=np.concatenate([self.values, other.values]),
        )

    def keep_most_recent(self, limit: int) -> "TrainingExamples":
        """Return at most ``limit`` of the most recent examples.

        Args:
            limit: The maximum number of examples to keep.

        Returns:
            The trimmed examples (or all of them if already within the limit).
        """
        if len(self) <= limit:
            return self
        return TrainingExamples(
            planes=self.planes[-limit:],
            policy_indices=self.policy_indices[-limit:],
            promotion_indices=self.promotion_indices[-limit:],
            values=self.values[-limit:],
        )


class StyleStore:
    """The on-disk location of the Mini-Me dataset and model."""

    def __init__(self, directory: pathlib.Path) -> None:
        """Create a store rooted at a directory, creating it if needed.

        Args:
            directory: The directory holding the dataset and model.
        """
        self.directory = directory
        self.directory.mkdir(parents=True, exist_ok=True)
        self.dataset_path = self.directory / "style_dataset.npz"
        self.model_path = self.directory / "mini_me.pt"

    def load_examples(self) -> TrainingExamples:
        """Load the stored examples, or an empty set if none exist.

        Returns:
            The stored :class:`TrainingExamples`.
        """
        if not self.dataset_path.exists():
            return TrainingExamples.empty()
        with np.load(self.dataset_path) as stored:
            return TrainingExamples(
                planes=stored["planes"].astype(np.float32),
                policy_indices=stored["policy_indices"].astype(np.int64),
                promotion_indices=stored["promotion_indices"].astype(np.int64),
                values=stored["values"].astype(np.float32),
            )

    def save_examples(self, examples: TrainingExamples) -> None:
        """Store examples on disk, compressing the planes to save space.

        Args:
            examples: The examples to save.
        """
        np.savez_compressed(
            self.dataset_path,
            planes=examples.planes.astype(np.int8),
            policy_indices=examples.policy_indices.astype(np.int64),
            promotion_indices=examples.promotion_indices.astype(np.int64),
            values=examples.values.astype(np.float32),
        )

    def append_examples(self, examples: TrainingExamples) -> TrainingExamples:
        """Append new examples to the stored dataset and save it.

        Args:
            examples: The new examples to add.

        Returns:
            The full, trimmed dataset after appending.
        """
        combined = self.load_examples().concatenate(examples).keep_most_recent(
            MAXIMUM_STORED_EXAMPLES
        )
        self.save_examples(combined)
        return combined


def default_store() -> StyleStore:
    """Return the default data store in the user's home directory.

    Returns:
        A :class:`StyleStore` rooted at ``~/.chess_mini_me``.
    """
    import os

    override = os.environ.get("CHESS_MINI_ME_DATA")
    directory = (
        pathlib.Path(override)
        if override
        else pathlib.Path.home() / ".chess_mini_me"
    )
    return StyleStore(directory)


class StyleRecorder:
    """Collect the player's moves during a live game as training examples."""

    def __init__(self) -> None:
        """Start with no recorded moves."""
        self._planes: list[np.ndarray] = []
        self._policy_indices: list[int] = []
        self._promotion_indices: list[int] = []

    def record(
        self, gamestate: "GameState", move: "Move", promotion_piece: str
    ) -> None:
        """Record one move the player made.

        Args:
            gamestate: The game state as the player saw it, before the move.
            move: The move the player chose.
            promotion_piece: The piece chosen for a promotion (ignored when the
                move is not a promotion).
        """
        self._planes.append(encoding.encode_gamestate(gamestate))
        self._policy_indices.append(
            encoding.move_to_policy_index(
                move.start_row, move.start_column, move.end_row, move.end_column
            )
        )
        if move.is_pawn_promotion:
            self._promotion_indices.append(
                encoding.PROMOTION_TO_INDEX[promotion_piece]
            )
        else:
            self._promotion_indices.append(-1)

    def finalise(self, result_value: float) -> TrainingExamples:
        """Turn the recorded moves into examples labelled with the result.

        Args:
            result_value: The game result from White's perspective
                (``1`` White win, ``-1`` Black win, ``0`` draw).

        Returns:
            The recorded :class:`TrainingExamples`.
        """
        if not self._planes:
            return TrainingExamples.empty()
        count = len(self._planes)
        return TrainingExamples(
            planes=np.stack(self._planes).astype(np.float32),
            policy_indices=np.array(self._policy_indices, dtype=np.int64),
            promotion_indices=np.array(self._promotion_indices, dtype=np.int64),
            values=np.full((count,), result_value, dtype=np.float32),
        )

    def has_moves(self) -> bool:
        """Return whether any moves have been recorded.

        Returns:
            True if at least one move has been recorded.
        """
        return bool(self._planes)


def train_network(
    model: "nn.Module",
    examples: TrainingExamples,
    epochs: int = 5,
    batch_size: int = 64,
    learning_rate: float = 1e-3,
    value_loss_weight: float = 0.5,
    promotion_loss_weight: float = 0.5,
    device: str = "cpu",
    progress_callback: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Train the network on examples by behavioural cloning.

    The loss combines three parts: cross-entropy that pushes the policy towards
    the move the player made, cross-entropy on the promotion choice (only for
    promotion examples), and mean-squared error on the value head against the
    game result.

    Args:
        model: The network to train, modified in place.
        examples: The training examples.
        epochs: How many passes to make over the data.
        batch_size: The mini-batch size.
        learning_rate: The optimiser learning rate.
        value_loss_weight: The weight on the value loss.
        promotion_loss_weight: The weight on the promotion loss.
        device: The PyTorch device to train on.
        progress_callback: An optional function called with
            ``(epoch, mean_loss)`` after each epoch.

    Returns:
        The mean loss for each epoch.
    """
    torch = cloner._import_torch()
    from torch import nn
    from torch.utils.data import DataLoader, TensorDataset

    # Batch normalisation needs at least two examples to estimate statistics, so
    # there is nothing meaningful to train on below that.
    if len(examples) < 2:
        return []

    planes = torch.from_numpy(examples.planes.astype(np.float32))
    policy_targets = torch.from_numpy(examples.policy_indices)
    promotion_targets = torch.from_numpy(examples.promotion_indices)
    value_targets = torch.from_numpy(examples.values)

    # Drop a trailing batch only when it would hold a single example, which
    # batch normalisation cannot handle while training.
    drop_last = len(examples) > batch_size and len(examples) % batch_size == 1
    loader = DataLoader(
        TensorDataset(planes, policy_targets, promotion_targets, value_targets),
        batch_size=batch_size,
        shuffle=True,
        drop_last=drop_last,
    )

    model = model.to(device)
    model.train()
    optimiser = torch.optim.Adam(model.parameters(), lr=learning_rate)
    policy_loss_function = nn.CrossEntropyLoss()
    promotion_loss_function = nn.CrossEntropyLoss()
    value_loss_function = nn.MSELoss()

    epoch_losses: list[float] = []
    for epoch in range(epochs):
        batch_losses: list[float] = []
        for plane_batch, policy_batch, promotion_batch, value_batch in loader:
            plane_batch = plane_batch.to(device)
            policy_batch = policy_batch.to(device)
            promotion_batch = promotion_batch.to(device)
            value_batch = value_batch.to(device)

            policy_logits, promotion_logits, value = model(plane_batch)
            loss = policy_loss_function(policy_logits, policy_batch)

            is_promotion = promotion_batch >= 0
            if bool(is_promotion.any()):
                loss = loss + promotion_loss_weight * promotion_loss_function(
                    promotion_logits[is_promotion], promotion_batch[is_promotion]
                )

            loss = loss + value_loss_weight * value_loss_function(
                value.squeeze(1), value_batch
            )

            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
            batch_losses.append(float(loss.item()))

        mean_loss = float(np.mean(batch_losses)) if batch_losses else 0.0
        epoch_losses.append(mean_loss)
        if progress_callback is not None:
            progress_callback(epoch, mean_loss)

    return epoch_losses


def load_or_build_model(
    store: StyleStore, channels: int = 64, residual_blocks: int = 4, device: str = "cpu"
) -> "nn.Module":
    """Load the stored model, or build a fresh one when none exists.

    Args:
        store: The data store to look in.
        channels: The trunk width for a freshly built model.
        residual_blocks: The trunk depth for a freshly built model.
        device: The PyTorch device to place the model on.

    Returns:
        A network ready for training.
    """
    torch = cloner._import_torch()
    if store.model_path.exists():
        checkpoint = torch.load(store.model_path, map_location=device)
        model = cloner.StyleNetwork(
            channels=checkpoint.get("channels", channels),
            residual_blocks=checkpoint.get("residual_blocks", residual_blocks),
        )
        model.load_state_dict(checkpoint["model_state"])
        return model.to(device)
    return cloner.StyleNetwork(
        channels=channels, residual_blocks=residual_blocks
    ).to(device)


def save_model(
    store: StyleStore, model: "nn.Module", channels: int = 64, residual_blocks: int = 4
) -> None:
    """Save the model and the shape needed to rebuild it.

    Args:
        store: The data store to save into.
        model: The network to save.
        channels: The trunk width to record.
        residual_blocks: The trunk depth to record.
    """
    torch = cloner._import_torch()
    torch.save(
        {
            "model_state": model.state_dict(),
            "channels": channels,
            "residual_blocks": residual_blocks,
        },
        store.model_path,
    )


def learn_from_examples(
    store: StyleStore,
    examples: TrainingExamples,
    epochs: int = 5,
    channels: int = 64,
    residual_blocks: int = 4,
    device: str = "cpu",
    progress_callback: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Add examples to the store and train the model on the full dataset.

    This is the single entry point used both by online learning after a game
    and by the Lichess importer.

    Args:
        store: The data store holding the dataset and model.
        examples: The new examples to learn from.
        epochs: How many passes to make over the dataset.
        channels: The trunk width when building a fresh model.
        residual_blocks: The trunk depth when building a fresh model.
        device: The PyTorch device to train on.
        progress_callback: An optional per-epoch callback.

    Returns:
        The mean loss for each epoch.
    """
    dataset = store.append_examples(examples)
    model = load_or_build_model(store, channels, residual_blocks, device)
    losses = train_network(
        model,
        dataset,
        epochs=epochs,
        device=device,
        progress_callback=progress_callback,
    )
    save_model(store, model, channels, residual_blocks)
    return losses
