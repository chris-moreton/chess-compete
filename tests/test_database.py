"""Tests for compete.database module."""

import pytest
from compete.database import derive_elo_from_name
from compete.constants import DEFAULT_ELO


class TestDeriveEloFromName:
    """Tests for the pure name parsing logic in get_initial_elo."""

    def test_stockfish_with_numeric_suffix(self):
        """sf-XXXX should return XXXX as ELO."""
        assert derive_elo_from_name("sf-1400") == 1400.0
        assert derive_elo_from_name("sf-2400") == 2400.0
        assert derive_elo_from_name("sf-3000") == 3000.0

    def test_stockfish_full_returns_default(self):
        """sf-full should return default (not numeric)."""
        assert derive_elo_from_name("sf-full") == DEFAULT_ELO

    def test_stockfish_non_numeric_returns_default(self):
        """sf-XXX where XXX is not numeric should return default."""
        assert derive_elo_from_name("sf-abc") == DEFAULT_ELO
        assert derive_elo_from_name("sf-") == DEFAULT_ELO

    def test_rusty_rival_versions_return_2600(self):
        """v* engines (Rusty Rival) should return 2600."""
        assert derive_elo_from_name("v1") == 2600.0
        assert derive_elo_from_name("v10") == 2600.0
        assert derive_elo_from_name("v001-baseline") == 2600.0
        assert derive_elo_from_name("v27-fast") == 2600.0

    def test_non_rusty_v_names_return_default(self):
        """v followed by non-digit should return default."""
        assert derive_elo_from_name("very-fast") == DEFAULT_ELO
        assert derive_elo_from_name("vanguard") == DEFAULT_ELO

    def test_java_rival_returns_default(self):
        """java-rival-XX engines should return default."""
        assert derive_elo_from_name("java-rival-37") == DEFAULT_ELO
        assert derive_elo_from_name("java-rival-38") == DEFAULT_ELO

    def test_unknown_engine_returns_default(self):
        """Unknown engines should return default ELO."""
        assert derive_elo_from_name("unknown-engine") == DEFAULT_ELO
        assert derive_elo_from_name("my-engine") == DEFAULT_ELO

    def test_empty_string_returns_default(self):
        """Empty string should return default ELO."""
        assert derive_elo_from_name("") == DEFAULT_ELO

    def test_default_elo_value(self):
        """Verify the default ELO is reasonable."""
        assert DEFAULT_ELO == 1500
