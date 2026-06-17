"""Convenience entry point to train a Mini-Me from a Lichess account.

This simply forwards to :mod:`chess_mini_me.train_cli`. Example:

    python train_from_lichess.py my_lichess_name --max-games 300
"""

from chess_mini_me.train_cli import main

if __name__ == "__main__":
    raise SystemExit(main())
