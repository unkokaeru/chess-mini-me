"""The Mini-Me neural network and the agent that plays in its style.

The Mini-Me is trained by imitation (behavioural cloning): it watches the moves
a particular player makes and learns to predict the same move in the same
position. At play time it samples from that learned distribution, restricted to
the legal moves, so it tends to reach for the same openings, the same tactics
and the same decisions as the player it has copied.

The network is a small convolutional residual network with three outputs, in
the style of modern board-game engines:

* a policy head, predicting which move the player would make;
* a promotion head, predicting which piece a promoting pawn would become;
* a value head, predicting how the game will end, which is a standard auxiliary
  task that helps the shared layers learn useful features.

PyTorch is imported lazily so that importing this module (for example to check
whether a saved Mini-Me exists) does not require PyTorch until it is actually
used.
"""

from __future__ import annotations

import pathlib
from typing import TYPE_CHECKING

import numpy as np

from chess_mini_me import constants, encoding

if TYPE_CHECKING:
    import torch
    from torch import nn

    from chess_mini_me.engine import GameState, Move


def _import_torch():
    """Import and return PyTorch, with a helpful error if it is missing.

    Returns:
        The imported ``torch`` module.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch

        return torch
    except ImportError as error:  # pragma: no cover - exercised only without torch
        raise ImportError(
            "The Mini-Me opponent requires PyTorch. Install it with "
            "'pip install torch' or 'pip install -r requirements-mini-me.txt'."
        ) from error


def build_network(channels: int = 64, residual_blocks: int = 4) -> "nn.Module":
    """Build a fresh, untrained Mini-Me network.

    Args:
        channels: The number of convolutional channels in the trunk.
        residual_blocks: How many residual blocks the trunk contains.

    Returns:
        A new :class:`StyleNetwork` instance.
    """
    return StyleNetwork(channels=channels, residual_blocks=residual_blocks)


def _define_network_classes():
    """Define the network classes, deferring the heavy PyTorch import.

    Returning the classes from a function keeps ``import torch`` out of module
    import time while still letting the classes be referenced by name.

    Returns:
        The ``StyleNetwork`` class.
    """
    torch = _import_torch()
    from torch import nn
    import torch.nn.functional as functional

    class ResidualBlock(nn.Module):
        """A pre-activation residual block of two 3x3 convolutions."""

        def __init__(self, channels: int) -> None:
            """Create the block.

            Args:
                channels: The number of input and output channels.
            """
            super().__init__()
            self.first_convolution = nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, bias=False
            )
            self.first_normalisation = nn.BatchNorm2d(channels)
            self.second_convolution = nn.Conv2d(
                channels, channels, kernel_size=3, padding=1, bias=False
            )
            self.second_normalisation = nn.BatchNorm2d(channels)

        def forward(self, features: "torch.Tensor") -> "torch.Tensor":
            """Apply the residual block.

            Args:
                features: The input feature map.

            Returns:
                The output feature map of the same shape.
            """
            residual = features
            output = functional.relu(self.first_normalisation(self.first_convolution(features)))
            output = self.second_normalisation(self.second_convolution(output))
            return functional.relu(output + residual)

    class StyleNetwork(nn.Module):
        """The convolutional network that imitates a player's choices."""

        def __init__(self, channels: int = 64, residual_blocks: int = 4) -> None:
            """Build the network.

            Args:
                channels: The number of channels in the convolutional trunk.
                residual_blocks: The number of residual blocks in the trunk.
            """
            super().__init__()
            self.stem = nn.Sequential(
                nn.Conv2d(
                    encoding.NUMBER_OF_INPUT_PLANES,
                    channels,
                    kernel_size=3,
                    padding=1,
                    bias=False,
                ),
                nn.BatchNorm2d(channels),
                nn.ReLU(inplace=True),
            )
            self.trunk = nn.Sequential(
                *(ResidualBlock(channels) for _ in range(residual_blocks))
            )

            board_squares = encoding.NUMBER_OF_SQUARES
            self.policy_convolution = nn.Sequential(
                nn.Conv2d(channels, 32, kernel_size=1, bias=False),
                nn.BatchNorm2d(32),
                nn.ReLU(inplace=True),
            )
            self.policy_output = nn.Linear(32 * board_squares, encoding.POLICY_SIZE)

            self.promotion_output = nn.Linear(
                channels, encoding.NUMBER_OF_PROMOTION_CLASSES
            )

            self.value_convolution = nn.Sequential(
                nn.Conv2d(channels, 1, kernel_size=1, bias=False),
                nn.BatchNorm2d(1),
                nn.ReLU(inplace=True),
            )
            self.value_hidden = nn.Linear(board_squares, channels)
            self.value_output = nn.Linear(channels, 1)

        def forward(
            self, planes: "torch.Tensor"
        ) -> tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
            """Run the network on a batch of positions.

            Args:
                planes: A batch of encoded positions, shaped
                    ``(batch, planes, 8, 8)``.

            Returns:
                A tuple ``(policy_logits, promotion_logits, value)``.
            """
            features = self.trunk(self.stem(planes))

            policy_features = self.policy_convolution(features)
            policy_logits = self.policy_output(
                policy_features.flatten(start_dim=1)
            )

            pooled = features.mean(dim=(2, 3))
            promotion_logits = self.promotion_output(pooled)

            value_features = self.value_convolution(features).flatten(start_dim=1)
            value = torch.tanh(
                self.value_output(functional.relu(self.value_hidden(value_features)))
            )

            return policy_logits, promotion_logits, value

    return StyleNetwork


# The network class is created on first use and cached here.
_STYLE_NETWORK_CLASS = None


def StyleNetwork(*arguments, **keyword_arguments):  # noqa: N802 - factory mimics a class
    """Create a :class:`StyleNetwork`, building the class on first use.

    Args:
        *arguments: Positional arguments forwarded to the network.
        **keyword_arguments: Keyword arguments forwarded to the network.

    Returns:
        A new network instance.
    """
    global _STYLE_NETWORK_CLASS
    if _STYLE_NETWORK_CLASS is None:
        _STYLE_NETWORK_CLASS = _define_network_classes()
    return _STYLE_NETWORK_CLASS(*arguments, **keyword_arguments)


class MiniMe:
    """An agent that plays moves in the style of the player it has learned."""

    def __init__(
        self,
        model: "nn.Module",
        temperature: float = 0.6,
        device: str = "cpu",
    ) -> None:
        """Wrap a trained network so it can choose moves.

        Args:
            model: The trained network.
            temperature: How adventurous the play is. Values near zero make the
                agent almost always pick the player's most likely move; larger
                values introduce more variety.
            device: The PyTorch device to run inference on.
        """
        self._torch = _import_torch()
        self.model = model.to(device)
        self.model.eval()
        self.temperature = temperature
        self.device = device

    def select_move(
        self, gamestate: "GameState", legal_moves: list["Move"]
    ) -> tuple["Move", str]:
        """Choose a move in the learned style from the legal moves.

        Args:
            gamestate: The current game state.
            legal_moves: The legal moves available to the side to move.

        Returns:
            A tuple ``(move, promotion_piece)``. The promotion piece is the
            queen unless the move is a promotion, in which case it is the piece
            the network predicts the player would choose.
        """
        torch = self._torch
        planes = encoding.encode_gamestate(gamestate)
        mask, index_to_move = encoding.build_legal_move_lookup(legal_moves)

        with torch.no_grad():
            planes_tensor = torch.from_numpy(planes).unsqueeze(0).to(self.device)
            policy_logits, promotion_logits, _ = self.model(planes_tensor)

        probabilities = self._masked_move_probabilities(
            policy_logits.squeeze(0).cpu().numpy(), mask
        )
        chosen_index = self._sample_index(probabilities, list(index_to_move))
        move = index_to_move[chosen_index]

        promotion_piece = constants.QUEEN
        if move.is_pawn_promotion:
            promotion_class = int(promotion_logits.squeeze(0).argmax().item())
            promotion_piece = encoding.PROMOTION_PIECES[promotion_class]

        return move, promotion_piece

    def _masked_move_probabilities(
        self, policy_logits: np.ndarray, mask: np.ndarray
    ) -> np.ndarray:
        """Turn raw policy logits into a probability over legal moves only.

        Args:
            policy_logits: The network's policy logits for every move index.
            mask: A vector with 1 at legal move indices and 0 elsewhere.

        Returns:
            A probability distribution over the policy indices, with all
            probability mass on legal moves.
        """
        temperature = max(self.temperature, 1e-3)
        scaled = policy_logits / temperature
        # Subtract the maximum for numerical stability before exponentiating.
        scaled = scaled - scaled.max()
        weights = np.exp(scaled) * mask
        total = weights.sum()
        if total <= 0.0:
            # The network gave no weight to any legal move; fall back to a
            # uniform choice among them.
            return mask / mask.sum()
        return weights / total

    def _sample_index(
        self, probabilities: np.ndarray, legal_indices: list[int]
    ) -> int:
        """Sample a legal move index from the probability distribution.

        Args:
            probabilities: The probability of every policy index.
            legal_indices: The indices that correspond to legal moves.

        Returns:
            A chosen legal policy index.
        """
        if self.temperature <= 1e-3:
            return int(max(legal_indices, key=lambda index: probabilities[index]))
        legal_probabilities = np.array(
            [probabilities[index] for index in legal_indices]
        )
        legal_probabilities = legal_probabilities / legal_probabilities.sum()
        return int(np.random.choice(legal_indices, p=legal_probabilities))


def load_mini_me(
    checkpoint_path: pathlib.Path,
    temperature: float = 0.6,
    device: str = "cpu",
) -> MiniMe | None:
    """Load a Mini-Me from a checkpoint, if one exists.

    Args:
        checkpoint_path: The path to a saved model checkpoint.
        temperature: The play temperature for the loaded agent.
        device: The PyTorch device to load onto.

    Returns:
        A ready :class:`MiniMe`, or ``None`` when no checkpoint is present.
    """
    if not checkpoint_path.exists():
        return None
    torch = _import_torch()
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model = StyleNetwork(
        channels=checkpoint.get("channels", 64),
        residual_blocks=checkpoint.get("residual_blocks", 4),
    )
    model.load_state_dict(checkpoint["model_state"])
    return MiniMe(model, temperature=temperature, device=device)
