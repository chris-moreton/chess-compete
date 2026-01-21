-- Migration: Add reference game tracking to SPSA iterations
-- Reference games are played against Stockfish to measure actual engine strength

-- Add columns for reference game tracking
ALTER TABLE spsa_iterations
ADD COLUMN IF NOT EXISTS base_engine_path VARCHAR(500),
ADD COLUMN IF NOT EXISTS ref_engine_path VARCHAR(500),
ADD COLUMN IF NOT EXISTS ref_games_played INTEGER NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS ref_wins INTEGER NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS ref_losses INTEGER NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS ref_draws INTEGER NOT NULL DEFAULT 0,
ADD COLUMN IF NOT EXISTS ref_elo_estimate NUMERIC(7, 2);

-- ref_elo_estimate: Calculated Elo rating relative to reference engine (e.g., SF 2600)
