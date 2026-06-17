"""Chess Mini-Me: a small, readable chess engine with a Pygame interface.

The package is deliberately split so that the game rules (``engine``) and the
artificial-intelligence search (``move_finder``) carry no dependency on
Pygame. Only ``interface`` imports Pygame, which keeps the engine importable in
headless environments such as continuous integration.
"""

from chess_mini_me.engine import CastleRights, GameState, Move

__all__ = ["CastleRights", "GameState", "Move"]
