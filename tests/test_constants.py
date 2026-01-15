"""Tests for compete.constants module."""

import pytest
from compete.constants import (
    K_FACTOR_PROVISIONAL,
    K_FACTOR_ESTABLISHED,
    PROVISIONAL_GAMES,
    DEFAULT_ELO,
    DB_MAX_RETRIES,
    DB_RETRY_BASE_DELAY,
    DB_RETRY_MAX_DELAY,
)


class TestEloConstants:
    """Tests for Elo rating constants."""

    def test_k_factor_provisional_is_positive(self):
        assert K_FACTOR_PROVISIONAL > 0

    def test_k_factor_established_is_positive(self):
        assert K_FACTOR_ESTABLISHED > 0

    def test_provisional_k_factor_higher_than_established(self):
        """Provisional ratings should adjust faster."""
        assert K_FACTOR_PROVISIONAL > K_FACTOR_ESTABLISHED

    def test_provisional_games_threshold_is_positive(self):
        assert PROVISIONAL_GAMES > 0

    def test_default_elo_is_reasonable(self):
        """Default Elo should be in typical rating range."""
        assert 1000 <= DEFAULT_ELO <= 2000


class TestDatabaseConstants:
    """Tests for database retry constants."""

    def test_max_retries_is_positive(self):
        assert DB_MAX_RETRIES > 0

    def test_base_delay_is_positive(self):
        assert DB_RETRY_BASE_DELAY > 0

    def test_max_delay_is_positive(self):
        assert DB_RETRY_MAX_DELAY > 0

    def test_max_delay_greater_than_base(self):
        """Max delay should be >= base delay."""
        assert DB_RETRY_MAX_DELAY >= DB_RETRY_BASE_DELAY


class TestConstantValues:
    """Tests for specific constant values to catch accidental changes."""

    def test_k_factor_provisional_value(self):
        assert K_FACTOR_PROVISIONAL == 40

    def test_k_factor_established_value(self):
        assert K_FACTOR_ESTABLISHED == 20

    def test_provisional_games_value(self):
        assert PROVISIONAL_GAMES == 30

    def test_default_elo_value(self):
        assert DEFAULT_ELO == 1500

    def test_db_max_retries_value(self):
        assert DB_MAX_RETRIES == 12

    def test_db_retry_base_delay_value(self):
        assert DB_RETRY_BASE_DELAY == 5

    def test_db_retry_max_delay_value(self):
        assert DB_RETRY_MAX_DELAY == 60
