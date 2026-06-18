"""Command-line tool to train a Mini-Me from a Lichess account.

Example:

    python -m chess_mini_me.train_cli my_lichess_name --max-games 300 --epochs 8

The trained Mini-Me is stored in the data directory (``~/.chess_mini_me`` by
default) and is then available as an opponent in the game's main menu.
"""

from __future__ import annotations

import argparse
import sys

from chess_mini_me import lichess
from chess_mini_me.training import (
    StyleStore,
    learn_from_examples,
    store_for_profile,
)


def _report_epoch(epoch: int, mean_loss: float) -> None:
    """Print the mean loss after a training epoch.

    Args:
        epoch: The zero-based epoch number.
        mean_loss: The mean training loss for the epoch.
    """
    print(f"  epoch {epoch + 1}: loss {mean_loss:.4f}")


def parse_arguments(argument_values: list[str]) -> argparse.Namespace:
    """Parse the command-line arguments.

    Args:
        argument_values: The argument strings, excluding the program name.

    Returns:
        The parsed arguments.
    """
    parser = argparse.ArgumentParser(
        description="Train a Mini-Me to imitate a Lichess player's style."
    )
    parser.add_argument("username", help="The Lichess username to imitate.")
    parser.add_argument(
        "--max-games",
        type=int,
        default=200,
        help="How many recent games to download (default: 200).",
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=8,
        help="How many training passes to make (default: 8).",
    )
    parser.add_argument(
        "--profile",
        default=None,
        help="Name of the Mini-Me profile to create or update "
             "(default: the username). Existing profiles are extended.",
    )
    parser.add_argument(
        "--data-dir",
        default=None,
        help="A custom directory to store the dataset and model in, instead "
             "of a named profile.",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="An optional Lichess API token to raise the download rate limit.",
    )
    parser.add_argument(
        "--include-casual",
        action="store_true",
        help="Include casual games as well as rated ones.",
    )
    return parser.parse_args(argument_values)


def main(argument_values: list[str] | None = None) -> int:
    """Download a player's games, analyse them and train the Mini-Me.

    Args:
        argument_values: The argument strings, or ``None`` to read ``sys.argv``.

    Returns:
        A process exit code: ``0`` on success and ``1`` on failure.
    """
    arguments = parse_arguments(
        sys.argv[1:] if argument_values is None else argument_values
    )
    if arguments.data_dir is not None:
        import pathlib

        store = StyleStore(pathlib.Path(arguments.data_dir))
    else:
        store = store_for_profile(arguments.profile or arguments.username)

    print(f"Downloading up to {arguments.max_games} games for "
          f"'{arguments.username}' from Lichess ...")
    try:
        pgn_text = lichess.download_games(
            arguments.username,
            max_games=arguments.max_games,
            rated_only=not arguments.include_casual,
            token=arguments.token,
        )
    except (ValueError, ImportError) as error:
        print(f"Could not download games: {error}", file=sys.stderr)
        return 1

    games = lichess.analyse_games(pgn_text, arguments.username)
    print()
    print(lichess.summarise_style(games))
    print()

    examples = lichess.extract_examples(pgn_text, arguments.username)
    if len(examples) == 0:
        print("No usable moves were found, so there is nothing to train on.",
              file=sys.stderr)
        return 1

    print(f"Training the Mini-Me on {len(examples)} of the player's moves ...")
    try:
        learn_from_examples(
            store, examples, epochs=arguments.epochs, progress_callback=_report_epoch
        )
    except ImportError as error:
        print(f"Training needs PyTorch: {error}", file=sys.stderr)
        return 1

    profile_name = arguments.profile or arguments.username
    print(f"\nDone. The Mini-Me profile '{profile_name}' is saved in "
          f"{store.directory} and is now selectable in the game menu.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
