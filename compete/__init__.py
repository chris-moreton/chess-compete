"""
Chess engine competition harness package.

Usage:
    python -m compete --help
    python -m compete v1 v2 --games 100 --time 1.0
    python -m compete --random --games 100 --timelow 0.1 --timehigh 1.0
"""

from compete.constants import (
    K_FACTOR_PROVISIONAL,
    K_FACTOR_ESTABLISHED,
    PROVISIONAL_GAMES,
    DEFAULT_ELO,
    DB_MAX_RETRIES,
    DB_RETRY_BASE_DELAY,
    DB_RETRY_MAX_DELAY,
)

__all__ = [
    # Constants
    'K_FACTOR_PROVISIONAL',
    'K_FACTOR_ESTABLISHED',
    'PROVISIONAL_GAMES',
    'DEFAULT_ELO',
    'DB_MAX_RETRIES',
    'DB_RETRY_BASE_DELAY',
    'DB_RETRY_MAX_DELAY',
    # Competition functions (import from compete.competitions when needed)
    # - run_match, run_league, run_gauntlet, run_random, run_epd, print_league_table
]
