"""Tests for compete.openings module."""

import pytest
import tempfile
from pathlib import Path

from compete.openings import load_epd_positions, OPENING_BOOK


class TestOpeningBook:
    """Tests for OPENING_BOOK constant."""

    def test_opening_book_is_not_empty(self):
        assert len(OPENING_BOOK) > 0

    def test_opening_book_has_many_positions(self):
        """Should have a substantial number of openings."""
        assert len(OPENING_BOOK) >= 200

    def test_opening_book_entries_are_tuples(self):
        """Each entry should be (fen, name) tuple."""
        for entry in OPENING_BOOK:
            assert isinstance(entry, tuple)
            assert len(entry) == 2

    def test_opening_book_fens_are_valid_format(self):
        """FEN strings should have expected format."""
        for fen, name in OPENING_BOOK:
            parts = fen.split()
            assert len(parts) == 6, f"FEN should have 6 parts: {fen}"
            # Board position should have 8 ranks
            board = parts[0]
            ranks = board.split('/')
            assert len(ranks) == 8, f"FEN board should have 8 ranks: {board}"

    def test_opening_book_names_are_strings(self):
        """Opening names should be non-empty strings."""
        for fen, name in OPENING_BOOK:
            assert isinstance(name, str)
            assert len(name) > 0

    def test_opening_book_has_variety(self):
        """Should have different opening types."""
        names = [name for _, name in OPENING_BOOK]
        # Check for some common opening categories
        categories_found = 0
        for category in ['Sicilian', 'Italian', 'Ruy Lopez', "Queen's Gambit", 'French', 'Caro-Kann']:
            if any(category in name for name in names):
                categories_found += 1
        assert categories_found >= 4, "Should have variety of opening types"


class TestLoadEpdPositions:
    """Tests for load_epd_positions function."""

    def test_load_simple_epd(self):
        """Load EPD with simple positions."""
        epd_content = """8/8/p2p3p/3k2p1/PP6/3K1P1P/8/8 b - -
8/8/8/8/8/8/8/8 w - -"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.epd', delete=False) as f:
            f.write(epd_content)
            f.flush()
            positions = load_epd_positions(Path(f.name))

        assert len(positions) == 2
        # First position
        fen1, id1 = positions[0]
        assert "8/8/p2p3p/3k2p1/PP6/3K1P1P/8/8 b - -" in fen1
        # Should add default halfmove/fullmove
        assert fen1.endswith(" 0 1")

    def test_load_epd_with_id(self):
        """Load EPD with id operations."""
        epd_content = '8/8/p2p3p/3k2p1/PP6/3K1P1P/8/8 b - - bm Kc6; id "E_E_T 001";'
        with tempfile.NamedTemporaryFile(mode='w', suffix='.epd', delete=False) as f:
            f.write(epd_content)
            f.flush()
            positions = load_epd_positions(Path(f.name))

        assert len(positions) == 1
        fen, pos_id = positions[0]
        assert pos_id == "E_E_T 001"

    def test_load_epd_without_id_uses_line_number(self):
        """Positions without id should use line number."""
        epd_content = """8/8/8/8/8/8/8/8 w - -
8/8/8/8/8/8/8/8 b - -"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.epd', delete=False) as f:
            f.write(epd_content)
            f.flush()
            positions = load_epd_positions(Path(f.name))

        assert len(positions) == 2
        assert positions[0][1] == "Position 1"
        assert positions[1][1] == "Position 2"

    def test_load_epd_skips_comments(self):
        """Lines starting with # should be skipped."""
        epd_content = """# This is a comment
8/8/8/8/8/8/8/8 w - -
# Another comment
8/8/8/8/8/8/8/8 b - -"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.epd', delete=False) as f:
            f.write(epd_content)
            f.flush()
            positions = load_epd_positions(Path(f.name))

        assert len(positions) == 2

    def test_load_epd_skips_empty_lines(self):
        """Empty lines should be skipped."""
        epd_content = """8/8/8/8/8/8/8/8 w - -

8/8/8/8/8/8/8/8 b - -

"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.epd', delete=False) as f:
            f.write(epd_content)
            f.flush()
            positions = load_epd_positions(Path(f.name))

        assert len(positions) == 2

    def test_load_epd_skips_malformed_lines(self):
        """Lines with fewer than 4 parts should be skipped."""
        epd_content = """8/8/8/8/8/8/8/8 w - -
8/8 w -
8/8/8/8/8/8/8/8 b - -"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.epd', delete=False) as f:
            f.write(epd_content)
            f.flush()
            positions = load_epd_positions(Path(f.name))

        assert len(positions) == 2

    def test_load_epd_returns_list_of_tuples(self):
        """Should return list of (fen, id) tuples."""
        epd_content = "8/8/8/8/8/8/8/8 w - -"
        with tempfile.NamedTemporaryFile(mode='w', suffix='.epd', delete=False) as f:
            f.write(epd_content)
            f.flush()
            positions = load_epd_positions(Path(f.name))

        assert isinstance(positions, list)
        assert len(positions) == 1
        assert isinstance(positions[0], tuple)
        assert len(positions[0]) == 2

    def test_load_empty_file_returns_empty_list(self):
        """Empty file should return empty list."""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.epd', delete=False) as f:
            f.write("")
            f.flush()
            positions = load_epd_positions(Path(f.name))

        assert positions == []
