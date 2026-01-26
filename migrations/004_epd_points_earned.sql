-- Migration 004: Add points_earned column for STS-style scoring
-- This allows tracking point scores (0-10) from test suites like STS
-- where moves are rated by quality, not just correct/incorrect

ALTER TABLE epd_test_results ADD COLUMN points_earned INTEGER;

-- Add comment explaining the column
COMMENT ON COLUMN epd_test_results.points_earned IS 'STS-style points (0-10) based on move quality, NULL if not STS format';
