# Chess Mini-Me

A small, readable game of chess written in Python, with a Pygame interface and
a built-in computer opponent. The project is intended to be easy to follow, so
the rules, the search and the interface are kept in separate modules and the
code favours clarity over cleverness.

## Features

- Full legal move generation, including castling, en passant, pawn promotion,
  pins, checks, checkmate and stalemate.
- A computer opponent using negamax search with alpha-beta pruning and a
  material-plus-position evaluation.
- A Pygame interface with a main menu, drag-and-drop or click-to-move input,
  move animation, last-move and legal-move highlighting, an on-board pawn
  promotion chooser, and a game-over screen.
- A test suite covering move generation (via perft), checkmate, stalemate, en
  passant, castling, promotion and the move finder.

## Project layout

    chess_mini_me/
        constants.py     Shared board, direction and evaluation constants.
        engine.py        Game state, move generation, making and undoing moves.
        move_finder.py   The negamax search and board evaluation.
        interface.py     The Pygame interface (the only module needing Pygame).
    tests/               The pytest test suite.
    main.py              The entry point.
    images/              The piece images.

The engine and move finder have no dependency on Pygame, so they can be
imported and tested without a display.

## Installation

You will need Python 3.10 or newer.

    python -m pip install -r requirements.txt

## Running the game

    python main.py

In the main menu, choose whether each colour is played by a human or the
computer, then start the game. During play:

- Drag a piece to a square, or click the piece and then click the destination.
- When a pawn reaches the far rank, click the piece to promote it to.
- Press `Z` to undo the last move.
- Press `R` to return to the menu, or `Q` to quit.

## Running the tests

    python -m pip install -r requirements-dev.txt
    python -m pytest

## Acknowledgements

The design is inspired by Eddie Sharick's chess programming tutorial series.
This rewrite restructures the code into a package, fixes several bugs (most
notably checkmate being reported as stalemate), removes duplication, and adds a
test suite.

## Licence

Released under the MIT Licence. See the `LICENSE` file for details.
