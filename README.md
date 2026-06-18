# Chess Mini-Me

A small, readable game of chess written in Python, with a Pygame interface, a
classical computer opponent, and a learning "Mini-Me" opponent that copies your
playing style. The project is intended to be easy to follow, so the rules, the
search, the interface and the machine learning are kept in separate modules.

## Features

- Full legal move generation, including castling, en passant, pawn promotion,
  pins, checks, checkmate and stalemate.
- A classical computer opponent using negamax search with alpha-beta pruning
  and a material-plus-position evaluation.
- A **Mini-Me** opponent that learns to imitate you. It is a convolutional
  neural network (PyTorch) trained by behavioural cloning on your moves, and it
  improves after every game you play against it.
- An option to **train the Mini-Me from a Lichess account**, learning a player's
  style from their game history.
- A Pygame interface with a main menu, drag-and-drop or click-to-move input,
  move animation, highlighting, an on-board promotion chooser and a game-over
  screen.
- A test suite covering the engine, the move finder, the encoding, the Lichess
  pipeline and (where PyTorch is installed) the network and trainer.

## How the Mini-Me works

The Mini-Me is trained by **imitation learning**, also called behavioural
cloning: it is shown the positions a player faced and the move they chose, and
it learns to predict the same move in the same position.

- **Encoding** (`encoding.py`): each position becomes a stack of 8x8 planes
  (where each piece type is, whose turn it is, castling rights and en passant),
  built with NumPy. A move is an index into a 64x64 origin-and-destination
  policy vector.
- **Network** (`cloner.py`): a small convolutional residual network with three
  heads, in the style of modern board-game engines: a policy head (which move),
  a promotion head (which piece to promote to) and an auxiliary value head
  (how the game will end).
- **Training** (`training.py`): cross-entropy on the policy against the move the
  player made, plus a promotion loss and a value loss, optimised with Adam. The
  dataset of your moves is kept on disk and grows over time, so the Mini-Me
  keeps improving the more you play.
- **Play**: at move time the network's predictions are restricted to the legal
  moves and sampled with a temperature, so the Mini-Me tends to reach for the
  same openings, tactics and decisions as the player it has copied.

Until a Mini-Me has been trained, choosing it as an opponent simply falls back
to the classical search, so the game is always playable.

## Project layout

    chess_mini_me/
        constants.py     Shared board, direction and evaluation constants.
        engine.py        Game state, move generation, making and undoing moves.
        move_finder.py   The classical negamax search and board evaluation.
        interface.py     The Pygame interface (the only module needing Pygame).
        encoding.py      Position and move encoding for the network (NumPy).
        cloner.py        The Mini-Me network and the agent that plays its style.
        training.py      Behavioural-cloning training and on-disk persistence.
        lichess.py       Downloading and analysing games from Lichess.
        opponent.py      Choosing moves for computer and Mini-Me players.
        train_cli.py     A command-line tool to train from Lichess.
    tests/               The pytest test suite.
    main.py              The entry point for the game.
    train_from_lichess.py  A convenience entry point for Lichess training.
    images/              The piece images.

The engine and move finder depend only on the standard library, so they can be
imported and tested without Pygame, NumPy or PyTorch.

## Installation

You will need Python 3.10 or newer.

    python -m pip install -r requirements.txt

To use the Mini-Me and Lichess features, also install the extra libraries
(PyTorch, python-chess, requests and pandas):

    python -m pip install -r requirements-mini-me.txt

## Running the game

    python main.py

In the main menu, click each colour to cycle its player between Human, Computer
and Mini-Me, then start the game. During play:

- Drag a piece to a square, or click the piece and then click the destination.
- When a pawn reaches the far rank, click the piece to promote it to.
- Press `Z` to undo the last move.
- Press `R` to return to the menu, or `Q` to quit.

After each game, the Mini-Me learns from the moves you made.

## Training the Mini-Me from Lichess

From the main menu, choose "Learn from Lichess" and enter a username. Or use the
command line:

    python train_from_lichess.py my_lichess_name --max-games 300 --epochs 8

The trained Mini-Me and its dataset are stored in `~/.chess_mini_me` (override
with the `CHESS_MINI_ME_DATA` environment variable).

## Running the tests

    python -m pip install -r requirements-dev.txt
    python -m pytest

Tests that need PyTorch are skipped automatically when it is not installed.

## Acknowledgements

The chess engine design is inspired by Eddie Sharick's tutorial series. The
Mini-Me follows the well-known approach of imitation learning with a
convolutional policy network.

## Licence

Released under the MIT Licence. See the `LICENSE` file for details.
