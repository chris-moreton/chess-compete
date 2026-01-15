"""Tests for compete.engine_manager module."""

import pytest
from compete.engine_manager import parse_engine_name


class TestParseEngineName:
    """Tests for parse_engine_name function."""

    def test_rusty_rival_with_v_prefix(self):
        """v* engines should be recognized as rusty type."""
        assert parse_engine_name("v1") == ("rusty", "v1")
        assert parse_engine_name("v10") == ("rusty", "v10")
        assert parse_engine_name("v001-baseline") == ("rusty", "v001-baseline")
        assert parse_engine_name("v1.0.17") == ("rusty", "v1.0.17")

    def test_java_rival_engines(self):
        """java-rival-* should be recognized as java type."""
        result = parse_engine_name("java-rival-36")
        assert result == ("java", "36")

        result = parse_engine_name("java-rival-38")
        assert result == ("java", "38")

        result = parse_engine_name("java-rival-38.0.0")
        assert result == ("java", "38.0.0")

    def test_stockfish_with_elo(self):
        """sf-XXXX should be recognized as stockfish type."""
        result = parse_engine_name("sf-1400")
        assert result == ("stockfish", "1400")

        result = parse_engine_name("sf-2400")
        assert result == ("stockfish", "2400")

        result = parse_engine_name("sf-3000")
        assert result == ("stockfish", "3000")

    def test_stockfish_full(self):
        """sf-full should be recognized as stockfish type."""
        result = parse_engine_name("sf-full")
        assert result == ("stockfish", "full")

    def test_unknown_engine_returns_none(self):
        """Unknown engine names should return None."""
        assert parse_engine_name("unknown-engine") is None
        assert parse_engine_name("my-engine") is None
        assert parse_engine_name("komodo") is None

    def test_v_followed_by_non_digit(self):
        """v followed by non-digit should not match rusty."""
        assert parse_engine_name("vanguard") is None
        assert parse_engine_name("very-fast") is None

    def test_empty_string(self):
        """Empty string should return None."""
        assert parse_engine_name("") is None

    def test_partial_matches_dont_match(self):
        """Partial prefixes should not match."""
        # Just "v" without a digit
        assert parse_engine_name("v") is None
        # sf- without anything after
        assert parse_engine_name("sf-") == ("stockfish", "")
        # java-rival- without version
        assert parse_engine_name("java-rival-") == ("java", "")
