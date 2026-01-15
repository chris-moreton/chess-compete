"""
Constants for the chess engine competition harness.
"""

# Default K-factor for Elo updates (higher = faster adjustment)
# Use higher K for provisional ratings (< 30 games)
K_FACTOR_PROVISIONAL = 40
K_FACTOR_ESTABLISHED = 20
PROVISIONAL_GAMES = 30
DEFAULT_ELO = 1500

# Database retry settings - exponential backoff for handling extended outages
DB_MAX_RETRIES = 12
DB_RETRY_BASE_DELAY = 5  # seconds (initial delay)
DB_RETRY_MAX_DELAY = 60  # seconds (cap on delay between retries)
