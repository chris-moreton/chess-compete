"""
EPD (Extended Position Description) data classes for solve testing.
"""

from dataclasses import dataclass, field
from typing import Any


@dataclass
class EpdPosition:
    """Represents a position from an EPD file with all parsed operations."""
    fen: str
    id: str
    best_moves: list[str] = field(default_factory=list)  # From 'bm' operation
    avoid_moves: list[str] = field(default_factory=list)  # From 'am' operation
    centipawn_eval: int | None = None  # From 'ce' operation
    direct_mate: int | None = None  # From 'dm' operation


@dataclass
class SolveResult:
    """Result of attempting to solve a single EPD position."""
    solved: bool
    move_found: str | None
    solve_time: float | None
    final_depth: int | None
    score: Any  # chess.engine.Score - using Any to avoid import dependency
    score_valid: bool | None
    timed_out: bool = False
