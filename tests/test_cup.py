"""Tests for cup competition functionality."""

import pytest
from compete.cup import (
    next_power_of_2,
    generate_bracket,
    get_round_name,
)


class TestNextPowerOf2:
    """Tests for next_power_of_2 function."""

    def test_power_of_2_returns_same(self):
        """Power of 2 inputs return the same value."""
        assert next_power_of_2(2) == 2
        assert next_power_of_2(4) == 4
        assert next_power_of_2(8) == 8
        assert next_power_of_2(16) == 16

    def test_non_power_of_2_rounds_up(self):
        """Non-power-of-2 inputs round up to next power."""
        assert next_power_of_2(3) == 4
        assert next_power_of_2(5) == 8
        assert next_power_of_2(6) == 8
        assert next_power_of_2(7) == 8
        assert next_power_of_2(9) == 16
        assert next_power_of_2(12) == 16
        assert next_power_of_2(15) == 16

    def test_one_returns_one(self):
        """Edge case: 1 returns 1."""
        assert next_power_of_2(1) == 1


class TestGenerateBracket:
    """Tests for generate_bracket function."""

    def test_power_of_2_has_no_byes(self):
        """Power of 2 participants have no byes."""
        bracket = generate_bracket(4)
        for s1, s2 in bracket:
            assert s2 is not None

        bracket = generate_bracket(8)
        for s1, s2 in bracket:
            assert s2 is not None

    def test_non_power_of_2_has_byes(self):
        """Non-power-of-2 participants get byes for top seeds."""
        bracket = generate_bracket(3)
        # 3 participants in 4-bracket = 1 bye
        byes = [s1 for s1, s2 in bracket if s2 is None]
        assert len(byes) == 1
        # Bye should go to seed 1
        assert 1 in byes

    def test_12_participants_has_4_byes(self):
        """12 participants = 4 byes (16 - 12)."""
        bracket = generate_bracket(12)
        byes = [s1 for s1, s2 in bracket if s2 is None]
        assert len(byes) == 4
        # Top 4 seeds get byes
        for seed in [1, 2, 3, 4]:
            assert seed in byes

    def test_bracket_size_matches_next_power_of_2(self):
        """Bracket has correct number of matches."""
        # 8 participants = 4 first-round matches
        bracket = generate_bracket(8)
        assert len(bracket) == 4

        # 16 participants = 8 first-round matches
        bracket = generate_bracket(16)
        assert len(bracket) == 8

        # 12 participants = 8 first-round matches (4 byes + 4 real)
        bracket = generate_bracket(12)
        assert len(bracket) == 8

    def test_seeds_1_and_2_are_separated(self):
        """Seeds 1 and 2 should be in opposite halves of bracket."""
        bracket = generate_bracket(8)
        # Find which match has seed 1
        seed_1_match = next(i for i, (s1, s2) in enumerate(bracket) if s1 == 1 or s2 == 1)
        # Find which match has seed 2
        seed_2_match = next(i for i, (s1, s2) in enumerate(bracket) if s1 == 2 or s2 == 2)
        # They should be in different halves (0-1 vs 2-3 for 4 matches)
        half_size = len(bracket) // 2
        assert (seed_1_match < half_size) != (seed_2_match < half_size)

    def test_standard_seeding_16(self):
        """Verify standard 16-team bracket seeding."""
        bracket = generate_bracket(16)
        # Standard seeding: 1v16, 8v9, 4v13, 5v12, 2v15, 7v10, 3v14, 6v11
        expected_matchups = {(1, 16), (8, 9), (4, 13), (5, 12), (2, 15), (7, 10), (3, 14), (6, 11)}
        actual_matchups = {(s1, s2) for s1, s2 in bracket}
        assert actual_matchups == expected_matchups

    def test_all_seeds_present(self):
        """All seed numbers from 1 to n should be present."""
        for n in [4, 8, 12, 16]:
            bracket = generate_bracket(n)
            seeds_seen = set()
            for s1, s2 in bracket:
                seeds_seen.add(s1)
                if s2 is not None:
                    seeds_seen.add(s2)
            # Seeds 1 to n should all be present
            expected = set(range(1, n + 1))
            assert seeds_seen == expected


class TestGetRoundName:
    """Tests for get_round_name function."""

    def test_final(self):
        assert get_round_name(2) == "Final"

    def test_semifinals(self):
        assert get_round_name(4) == "Semifinals"

    def test_quarterfinals(self):
        assert get_round_name(8) == "Quarterfinals"

    def test_round_of_16(self):
        assert get_round_name(16) == "Round of 16"

    def test_round_of_32(self):
        assert get_round_name(32) == "Round of 32"

    def test_unknown_size_returns_generic_name(self):
        assert get_round_name(128) == "Round of 128"
        assert get_round_name(256) == "Round of 256"
