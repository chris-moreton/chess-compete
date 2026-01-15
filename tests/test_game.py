"""Tests for compete.game module."""

import math
import pytest
from compete.game import calculate_elo_difference


class TestCalculateEloDifference:
    """Tests for calculate_elo_difference function."""

    def test_zero_games_returns_zero(self):
        """No games should return (0.0, 0.0)."""
        elo_diff, error = calculate_elo_difference(0, 0, 0)
        assert elo_diff == 0.0
        assert error == 0.0

    def test_all_wins_returns_high_positive(self):
        """100% wins should return high positive Elo diff."""
        elo_diff, error = calculate_elo_difference(10, 0, 0)
        assert elo_diff == 800.0  # Capped at 800
        assert error == 100.0

    def test_all_losses_returns_high_negative(self):
        """0% score should return high negative Elo diff."""
        elo_diff, error = calculate_elo_difference(0, 10, 0)
        assert elo_diff == -800.0  # Capped at -800
        assert error == 100.0

    def test_even_score_returns_zero(self):
        """50% score should return ~0 Elo diff."""
        elo_diff, error = calculate_elo_difference(5, 5, 0)
        assert abs(elo_diff) < 1.0  # Should be close to 0

    def test_draws_count_as_half(self):
        """Draws should count as 0.5 points."""
        # 5 wins, 5 draws = 7.5/10 = 75%
        elo_diff, _ = calculate_elo_difference(5, 0, 5)
        assert elo_diff > 0  # Should be positive

        # 5 losses, 5 draws = 2.5/10 = 25%
        elo_diff, _ = calculate_elo_difference(0, 5, 5)
        assert elo_diff < 0  # Should be negative

    def test_all_draws_returns_zero(self):
        """All draws = 50% score should return ~0 Elo diff."""
        elo_diff, error = calculate_elo_difference(0, 0, 10)
        assert abs(elo_diff) < 1.0  # Should be close to 0

    def test_positive_elo_for_winning_record(self):
        """More wins than losses should give positive Elo diff."""
        elo_diff, _ = calculate_elo_difference(7, 3, 0)
        assert elo_diff > 0

    def test_negative_elo_for_losing_record(self):
        """More losses than wins should give negative Elo diff."""
        elo_diff, _ = calculate_elo_difference(3, 7, 0)
        assert elo_diff < 0

    def test_error_margin_decreases_with_more_games(self):
        """Error margin should be lower with more games."""
        # Same 60% score, different sample sizes
        _, error_10 = calculate_elo_difference(6, 4, 0)
        _, error_100 = calculate_elo_difference(60, 40, 0)

        assert error_100 < error_10  # More games = less error

    def test_approximate_elo_values(self):
        """Test some approximate expected Elo differences."""
        # ~60% score = ~70 Elo advantage
        elo_diff, _ = calculate_elo_difference(6, 4, 0)
        assert 50 < elo_diff < 100

        # ~70% score = ~150 Elo advantage
        elo_diff, _ = calculate_elo_difference(7, 3, 0)
        assert 100 < elo_diff < 200

        # ~80% score = ~240 Elo advantage
        elo_diff, _ = calculate_elo_difference(8, 2, 0)
        assert 200 < elo_diff < 300

    def test_symmetry_of_results(self):
        """Swapped results should give opposite Elo diff."""
        elo_diff_pos, _ = calculate_elo_difference(7, 3, 0)
        elo_diff_neg, _ = calculate_elo_difference(3, 7, 0)

        assert abs(elo_diff_pos + elo_diff_neg) < 1.0  # Should sum to ~0

    def test_error_margin_is_positive(self):
        """Error margin should always be positive."""
        for wins in range(11):
            for losses in range(11 - wins):
                draws = 10 - wins - losses
                _, error = calculate_elo_difference(wins, losses, draws)
                assert error >= 0

    def test_elo_diff_is_capped(self):
        """Elo diff should be capped at reasonable values."""
        # Even with extreme results, shouldn't exceed 800
        elo_diff, _ = calculate_elo_difference(100, 0, 0)
        assert abs(elo_diff) <= 800

        elo_diff, _ = calculate_elo_difference(0, 100, 0)
        assert abs(elo_diff) <= 800
