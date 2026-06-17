"""Shared constants for the chess engine, the move finder and the interface.

Centralising these values means the board representation, the direction
vectors and the evaluation tables are defined exactly once and reused
everywhere, rather than being repeated across modules.
"""

from __future__ import annotations

from typing import Final

# ---------------------------------------------------------------------------
# Board representation
# ---------------------------------------------------------------------------

# The board is an 8x8 grid. Each square holds a two-character string: the first
# character is the colour ("w" or "b") and the second is the piece type ("K",
# "Q", "R", "B", "N" or "P"). An empty square is represented by EMPTY_SQUARE.

BOARD_DIMENSION: Final[int] = 8
EMPTY_SQUARE: Final[str] = "--"

WHITE: Final[str] = "w"
BLACK: Final[str] = "b"

KING: Final[str] = "K"
QUEEN: Final[str] = "Q"
ROOK: Final[str] = "R"
BISHOP: Final[str] = "B"
KNIGHT: Final[str] = "N"
PAWN: Final[str] = "P"

STARTING_BOARD: Final[tuple[tuple[str, ...], ...]] = (
    ("bR", "bN", "bB", "bQ", "bK", "bB", "bN", "bR"),
    ("bP", "bP", "bP", "bP", "bP", "bP", "bP", "bP"),
    ("--", "--", "--", "--", "--", "--", "--", "--"),
    ("--", "--", "--", "--", "--", "--", "--", "--"),
    ("--", "--", "--", "--", "--", "--", "--", "--"),
    ("--", "--", "--", "--", "--", "--", "--", "--"),
    ("wP", "wP", "wP", "wP", "wP", "wP", "wP", "wP"),
    ("wR", "wN", "wB", "wQ", "wK", "wB", "wN", "wR"),
)

# Every piece type, used when loading images and validating boards.
PIECE_TYPES: Final[tuple[str, ...]] = (KING, QUEEN, ROOK, BISHOP, KNIGHT, PAWN)
PIECE_COLOURS: Final[tuple[str, ...]] = (WHITE, BLACK)

# ---------------------------------------------------------------------------
# Direction vectors, expressed as (row offset, column offset)
# ---------------------------------------------------------------------------

# Rooks (and queens) move along orthogonal directions; bishops (and queens)
# move along diagonal directions. The king and queen reach every direction.
ORTHOGONAL_DIRECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (-1, 0),
    (1, 0),
    (0, -1),
    (0, 1),
)
DIAGONAL_DIRECTIONS: Final[tuple[tuple[int, int], ...]] = (
    (-1, -1),
    (-1, 1),
    (1, -1),
    (1, 1),
)
ALL_DIRECTIONS: Final[tuple[tuple[int, int], ...]] = (
    ORTHOGONAL_DIRECTIONS + DIAGONAL_DIRECTIONS
)
KNIGHT_OFFSETS: Final[tuple[tuple[int, int], ...]] = (
    (-2, -1),
    (-2, 1),
    (-1, -2),
    (-1, 2),
    (1, -2),
    (1, 2),
    (2, -1),
    (2, 1),
)

# Column index to file letter, used when producing chess notation.
COLUMNS_TO_FILES: Final[dict[int, str]] = {
    0: "a",
    1: "b",
    2: "c",
    3: "d",
    4: "e",
    5: "f",
    6: "g",
    7: "h",
}

# ---------------------------------------------------------------------------
# Move-finder evaluation
# ---------------------------------------------------------------------------

# Material value of each piece type, in pawn-equivalent units. The king is
# given a large value so that the search never trades it away during the
# fixed-depth look-ahead.
PIECE_MATERIAL_VALUE: Final[dict[str, int]] = {
    KING: 0,
    QUEEN: 900,
    ROOK: 500,
    BISHOP: 330,
    KNIGHT: 320,
    PAWN: 100,
}

# A checkmate is worth far more than any realistic material balance, so the
# search always prefers delivering mate (or avoiding being mated).
CHECKMATE_SCORE: Final[int] = 1_000_000
STALEMATE_SCORE: Final[int] = 0

# Default look-ahead depth for the move finder, in plies (half-moves).
DEFAULT_SEARCH_DEPTH: Final[int] = 3

# Piece-square tables add positional knowledge to the otherwise purely
# material evaluation. Each table is written from White's point of view with
# row 0 being the eighth rank (the top of the board, matching STARTING_BOARD).
# White reads a table directly as ``table[row][column]``; Black reads the
# vertically mirrored square ``table[mirror][column]``. Values are in
# pawn-equivalent centipawns. These tables are a widely used simplified set.
PAWN_POSITION_VALUE: Final[tuple[tuple[int, ...], ...]] = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (50, 50, 50, 50, 50, 50, 50, 50),
    (10, 10, 20, 30, 30, 20, 10, 10),
    (5, 5, 10, 25, 25, 10, 5, 5),
    (0, 0, 0, 20, 20, 0, 0, 0),
    (5, -5, -10, 0, 0, -10, -5, 5),
    (5, 10, 10, -20, -20, 10, 10, 5),
    (0, 0, 0, 0, 0, 0, 0, 0),
)
KNIGHT_POSITION_VALUE: Final[tuple[tuple[int, ...], ...]] = (
    (-50, -40, -30, -30, -30, -30, -40, -50),
    (-40, -20, 0, 0, 0, 0, -20, -40),
    (-30, 0, 10, 15, 15, 10, 0, -30),
    (-30, 5, 15, 20, 20, 15, 5, -30),
    (-30, 0, 15, 20, 20, 15, 0, -30),
    (-30, 5, 10, 15, 15, 10, 5, -30),
    (-40, -20, 0, 5, 5, 0, -20, -40),
    (-50, -40, -30, -30, -30, -30, -40, -50),
)
BISHOP_POSITION_VALUE: Final[tuple[tuple[int, ...], ...]] = (
    (-20, -10, -10, -10, -10, -10, -10, -20),
    (-10, 0, 0, 0, 0, 0, 0, -10),
    (-10, 0, 5, 10, 10, 5, 0, -10),
    (-10, 5, 5, 10, 10, 5, 5, -10),
    (-10, 0, 10, 10, 10, 10, 0, -10),
    (-10, 10, 10, 10, 10, 10, 10, -10),
    (-10, 5, 0, 0, 0, 0, 5, -10),
    (-20, -10, -10, -10, -10, -10, -10, -20),
)
ROOK_POSITION_VALUE: Final[tuple[tuple[int, ...], ...]] = (
    (0, 0, 0, 0, 0, 0, 0, 0),
    (5, 10, 10, 10, 10, 10, 10, 5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (-5, 0, 0, 0, 0, 0, 0, -5),
    (0, 0, 0, 5, 5, 0, 0, 0),
)
QUEEN_POSITION_VALUE: Final[tuple[tuple[int, ...], ...]] = (
    (-20, -10, -10, -5, -5, -10, -10, -20),
    (-10, 0, 0, 0, 0, 0, 0, -10),
    (-10, 0, 5, 5, 5, 5, 0, -10),
    (-5, 0, 5, 5, 5, 5, 0, -5),
    (0, 0, 5, 5, 5, 5, 0, -5),
    (-10, 5, 5, 5, 5, 5, 0, -10),
    (-10, 0, 5, 0, 0, 0, 0, -10),
    (-20, -10, -10, -5, -5, -10, -10, -20),
)
KING_POSITION_VALUE: Final[tuple[tuple[int, ...], ...]] = (
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-30, -40, -40, -50, -50, -40, -40, -30),
    (-20, -30, -30, -40, -40, -30, -30, -20),
    (-10, -20, -20, -20, -20, -20, -20, -10),
    (20, 20, 0, 0, 0, 0, 20, 20),
    (20, 30, 10, 0, 0, 10, 30, 20),
)
PIECE_POSITION_VALUE: Final[dict[str, tuple[tuple[int, ...], ...]]] = {
    PAWN: PAWN_POSITION_VALUE,
    KNIGHT: KNIGHT_POSITION_VALUE,
    BISHOP: BISHOP_POSITION_VALUE,
    ROOK: ROOK_POSITION_VALUE,
    QUEEN: QUEEN_POSITION_VALUE,
    KING: KING_POSITION_VALUE,
}

# ---------------------------------------------------------------------------
# Interface configuration
# ---------------------------------------------------------------------------

WINDOW_TITLE: Final[str] = "Chess Mini-Me"
MAXIMUM_FRAMES_PER_SECOND: Final[int] = 30

# Number of animation frames spent traversing a single square. A larger value
# produces a slower, smoother glide.
ANIMATION_FRAMES_PER_SQUARE: Final[int] = 6

# Margin, in pixels, left around the board when sizing the window to the
# available display height.
WINDOW_MARGIN_PIXELS: Final[int] = 100

# Fallback board size used when the display size cannot be determined.
FALLBACK_BOARD_SIZE_PIXELS: Final[int] = 640

# Relative directory (from the package root) that holds the piece images.
IMAGE_DIRECTORY_NAME: Final[str] = "images"
IMAGE_FILE_EXTENSION: Final[str] = ".png"
