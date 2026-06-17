"""Decide the moves for non-human players, including the Mini-Me.

This sits between the interface and the two ways the program can choose a move:
the classical negamax search (:mod:`chess_mini_me.move_finder`) and the learned
Mini-Me (:mod:`chess_mini_me.cloner`). Keeping the decision here means the
interface does not need to know how either works, and the logic can be tested
without a display.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Callable

from chess_mini_me import cloner, constants, move_finder

if TYPE_CHECKING:
    from chess_mini_me.engine import GameState, Move
    from chess_mini_me.training import StyleRecorder, StyleStore

# The three kinds of player a side can be controlled by.
PLAYER_HUMAN = "human"
PLAYER_COMPUTER = "computer"
PLAYER_MINI_ME = "mini_me"

PLAYER_TYPES = (PLAYER_HUMAN, PLAYER_COMPUTER, PLAYER_MINI_ME)

PLAYER_LABELS = {
    PLAYER_HUMAN: "Human",
    PLAYER_COMPUTER: "Computer",
    PLAYER_MINI_ME: "Mini-Me",
}


def next_player_type(player_type: str) -> str:
    """Return the next player type when cycling through the choices.

    Args:
        player_type: The current player type.

    Returns:
        The next player type in the cycle.
    """
    position = PLAYER_TYPES.index(player_type)
    return PLAYER_TYPES[(position + 1) % len(PLAYER_TYPES)]


class OpponentController:
    """Choose moves for the computer and Mini-Me players in a game."""

    def __init__(self, store: "StyleStore", temperature: float = 0.6) -> None:
        """Load any saved Mini-Me from the store.

        Args:
            store: The data store holding the trained Mini-Me, if any.
            temperature: The Mini-Me play temperature.
        """
        self.store = store
        self.temperature = temperature
        self._mini_me = self._load_mini_me()

    def _load_mini_me(self):
        """Load the Mini-Me agent if a checkpoint exists and PyTorch is present.

        Returns:
            A :class:`cloner.MiniMe`, or ``None`` if there is no checkpoint or
            PyTorch is not installed.
        """
        try:
            return cloner.load_mini_me(
                self.store.model_path, temperature=self.temperature
            )
        except ImportError:
            # PyTorch is unavailable, so the Mini-Me cannot run; the classical
            # search is used instead.
            return None

    def reload_mini_me(self) -> None:
        """Reload the Mini-Me, picking up a newly trained checkpoint."""
        self._mini_me = self._load_mini_me()

    def is_mini_me_ready(self) -> bool:
        """Return whether a trained Mini-Me is available.

        Returns:
            True if a Mini-Me has been loaded.
        """
        return self._mini_me is not None

    def choose_move(
        self, player_type: str, gamestate: "GameState", legal_moves: list["Move"]
    ) -> tuple["Move", str]:
        """Choose a move for a non-human player.

        The Mini-Me is used when it is ready; otherwise the request falls back
        to the classical search so the game is always playable.

        Args:
            player_type: ``PLAYER_COMPUTER`` or ``PLAYER_MINI_ME``.
            gamestate: The current game state.
            legal_moves: The legal moves for the side to move.

        Returns:
            A tuple ``(move, promotion_piece)``.
        """
        if player_type == PLAYER_MINI_ME and self._mini_me is not None:
            return self._mini_me.select_move(gamestate, legal_moves)
        return self._classical_move(gamestate, legal_moves)

    @staticmethod
    def _classical_move(
        gamestate: "GameState", legal_moves: list["Move"]
    ) -> tuple["Move", str]:
        """Choose a move with the negamax search.

        Args:
            gamestate: The current game state.
            legal_moves: The legal moves for the side to move.

        Returns:
            A tuple ``(move, promotion_piece)``; promotions default to a queen.
        """
        move = move_finder.find_best_move(gamestate, legal_moves)
        return move, constants.QUEEN


def learn_from_finished_game(
    store: "StyleStore",
    recorder: "StyleRecorder",
    result_value: float,
    epochs: int = 4,
    progress_callback: Callable[[int, float], None] | None = None,
) -> list[float]:
    """Teach the Mini-Me from the human moves of a finished game.

    Args:
        store: The data store to update.
        recorder: The recorder holding the human's moves from the game.
        result_value: The game result from White's perspective.
        epochs: How many training passes to make.
        progress_callback: An optional per-epoch callback.

    Returns:
        The mean loss for each epoch, or an empty list if nothing was recorded.
    """
    # Imported here so that the interface can import this module without
    # pulling in PyTorch until a game is actually learned from.
    from chess_mini_me import training

    if not recorder.has_moves():
        return []
    examples = recorder.finalise(result_value)
    return training.learn_from_examples(
        store, examples, epochs=epochs, progress_callback=progress_callback
    )
