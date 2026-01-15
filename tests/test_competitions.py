"""Tests for compete.competitions module."""

import io
import sys
import pytest
from unittest.mock import patch, MagicMock

from compete.competitions import print_league_table


class TestPrintLeagueTable:
    """Tests for print_league_table function."""

    @pytest.fixture
    def mock_ratings(self):
        """Mock load_elo_ratings to return predictable data."""
        return {
            "engine1": {"elo": 2500, "games": 50},
            "engine2": {"elo": 2400, "games": 40},
            "engine3": {"elo": 2300, "games": 20},
        }

    def test_competitors_only_mode_shows_only_competitors(self, mock_ratings):
        """When competitors_only=True, only show engines in the competition."""
        with patch('compete.competitions.load_elo_ratings', return_value=mock_ratings):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                print_league_table(
                    competitors={"engine1", "engine2"},
                    games_this_comp={"engine1": 4, "engine2": 4},
                    points_this_comp={"engine1": 3.0, "engine2": 1.0},
                    round_num=2,
                    competitors_only=True,
                    game_num=8,
                    total_games=20
                )

            output = captured.getvalue()
            # Should show both competitors
            assert "engine1" in output
            assert "engine2" in output
            # Should NOT show engine3 (not in competition)
            assert "engine3" not in output

    def test_competitors_only_mode_shows_standings_header(self, mock_ratings):
        """Compact mode should show 'STANDINGS (X/Y games)' header."""
        with patch('compete.competitions.load_elo_ratings', return_value=mock_ratings):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                print_league_table(
                    competitors={"engine1"},
                    games_this_comp={"engine1": 5},
                    points_this_comp={"engine1": 3.5},
                    round_num=1,
                    competitors_only=True,
                    game_num=5,
                    total_games=10
                )

            output = captured.getvalue()
            assert "STANDINGS (5/10 games)" in output

    def test_competitors_only_sorts_by_points_then_elo(self, mock_ratings):
        """Competitors should be sorted by points (desc) then Elo (desc)."""
        with patch('compete.competitions.load_elo_ratings', return_value=mock_ratings):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                print_league_table(
                    competitors={"engine1", "engine2", "engine3"},
                    games_this_comp={"engine1": 6, "engine2": 6, "engine3": 6},
                    # engine3 has most points, engine1 and engine2 are tied
                    points_this_comp={"engine1": 2.0, "engine2": 2.0, "engine3": 5.0},
                    round_num=3,
                    competitors_only=True,
                    game_num=18,
                    total_games=18
                )

            output = captured.getvalue()
            lines = output.strip().split('\n')
            # Find lines with engine names (skip headers)
            engine_lines = [l for l in lines if 'engine' in l and not l.startswith('#')]
            # engine3 should be first (most points)
            # engine1 should be before engine2 (same points but higher Elo)
            assert engine_lines[0].startswith("1") and "engine3" in engine_lines[0]
            assert engine_lines[1].startswith("2") and "engine1" in engine_lines[1]
            assert engine_lines[2].startswith("3") and "engine2" in engine_lines[2]

    def test_competitors_only_shows_elo_change(self, mock_ratings):
        """Should show Elo change when session_start_elo is provided."""
        with patch('compete.competitions.load_elo_ratings', return_value=mock_ratings):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                print_league_table(
                    competitors={"engine1"},
                    games_this_comp={"engine1": 10},
                    points_this_comp={"engine1": 7.0},
                    round_num=5,
                    competitors_only=True,
                    game_num=10,
                    total_games=10,
                    session_start_elo={"engine1": 2450}  # Started at 2450, now at 2500
                )

            output = captured.getvalue()
            # Should show +50 change
            assert "+50" in output

    def test_full_table_mode_shows_all_engines(self, mock_ratings):
        """Full table mode shows all engines with Elo rankings."""
        with patch('compete.competitions.load_elo_ratings', return_value=mock_ratings):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                print_league_table(
                    competitors={"engine1"},
                    games_this_comp={"engine1": 4},
                    points_this_comp={"engine1": 2.0},
                    round_num=2,
                    is_final=False,
                    competitors_only=False
                )

            output = captured.getvalue()
            # Should show all three engines
            assert "engine1" in output
            assert "engine2" in output
            assert "engine3" in output

    def test_full_table_shows_round_header(self, mock_ratings):
        """Non-final full table should show round number in header."""
        with patch('compete.competitions.load_elo_ratings', return_value=mock_ratings):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                print_league_table(
                    competitors={"engine1"},
                    games_this_comp={"engine1": 4},
                    points_this_comp={"engine1": 2.0},
                    round_num=3,
                    is_final=False,
                    competitors_only=False
                )

            output = captured.getvalue()
            assert "STANDINGS AFTER ROUND 3" in output

    def test_full_table_shows_final_header(self, mock_ratings):
        """Final full table should show 'FINAL LEAGUE STANDINGS'."""
        with patch('compete.competitions.load_elo_ratings', return_value=mock_ratings):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                print_league_table(
                    competitors={"engine1"},
                    games_this_comp={"engine1": 10},
                    points_this_comp={"engine1": 6.0},
                    round_num=5,
                    is_final=True,
                    competitors_only=False
                )

            output = captured.getvalue()
            assert "FINAL LEAGUE STANDINGS" in output

    def test_full_table_marks_competitors(self, mock_ratings):
        """Competitors should be marked with * in full table mode."""
        with patch('compete.competitions.load_elo_ratings', return_value=mock_ratings):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                print_league_table(
                    competitors={"engine2"},  # Only engine2 is competing
                    games_this_comp={"engine2": 4},
                    points_this_comp={"engine2": 2.5},
                    round_num=2,
                    is_final=False,
                    competitors_only=False
                )

            output = captured.getvalue()
            # engine2 should have * marker
            assert "*engine2" in output

    def test_provisional_rating_marker(self, mock_ratings):
        """Engines with < 30 games should have ? marker."""
        mock_ratings["engine3"]["games"] = 15  # Under provisional threshold
        with patch('compete.competitions.load_elo_ratings', return_value=mock_ratings):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                print_league_table(
                    competitors={"engine3"},
                    games_this_comp={"engine3": 4},
                    points_this_comp={"engine3": 2.0},
                    round_num=2,
                    competitors_only=True,
                    game_num=4,
                    total_games=10
                )

            output = captured.getvalue()
            # engine3 line should have ? marker (provisional)
            assert "?" in output

    def test_empty_competitors(self, mock_ratings):
        """Should handle empty competitors set gracefully."""
        with patch('compete.competitions.load_elo_ratings', return_value=mock_ratings):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                print_league_table(
                    competitors=set(),
                    games_this_comp={},
                    points_this_comp={},
                    round_num=0,
                    competitors_only=True,
                    game_num=0,
                    total_games=0
                )

            output = captured.getvalue()
            # Should still produce output without errors
            assert "STANDINGS" in output

    def test_column_headers_present(self, mock_ratings):
        """Should include column headers in output."""
        with patch('compete.competitions.load_elo_ratings', return_value=mock_ratings):
            captured = io.StringIO()
            with patch('sys.stdout', captured):
                print_league_table(
                    competitors={"engine1"},
                    games_this_comp={"engine1": 4},
                    points_this_comp={"engine1": 2.0},
                    round_num=2,
                    competitors_only=True,
                    game_num=4,
                    total_games=10
                )

            output = captured.getvalue()
            # Check for column headers
            assert "Engine" in output
            assert "Points" in output
            assert "Elo" in output
