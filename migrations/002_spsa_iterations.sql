-- Migration: Add SPSA tuning iteration table
-- Run this against your PostgreSQL database to enable SPSA parameter tuning

-- Table for SPSA iteration tracking
-- Each iteration represents one gradient estimation step with a plus/minus engine pair
CREATE TABLE IF NOT EXISTS spsa_iterations (
    id SERIAL PRIMARY KEY,
    iteration_number INTEGER NOT NULL,

    -- Engine binaries (paths to shared location, not registered engines)
    plus_engine_path VARCHAR(500) NOT NULL,
    minus_engine_path VARCHAR(500) NOT NULL,

    -- Time control (consistent with other compete modes: timelow/timehigh)
    timelow_ms INTEGER NOT NULL,   -- e.g., 250 (0.25s)
    timehigh_ms INTEGER NOT NULL,  -- e.g., 1000 (1.0s)

    -- Game tracking (workers increment these atomically)
    target_games INTEGER NOT NULL DEFAULT 150,
    games_played INTEGER NOT NULL DEFAULT 0,
    plus_wins INTEGER NOT NULL DEFAULT 0,
    minus_wins INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,

    -- Status: pending (waiting for workers), in_progress, complete
    status VARCHAR(20) NOT NULL DEFAULT 'pending',

    -- Parameter snapshot (JSON for reproducibility and analysis)
    base_parameters JSONB,       -- θ values before perturbation
    plus_parameters JSONB,       -- θ + δ values
    minus_parameters JSONB,      -- θ - δ values
    perturbation_signs JSONB,    -- +1 or -1 for each parameter

    -- Results (filled when complete)
    gradient_estimate JSONB,     -- Calculated gradient per parameter
    elo_diff NUMERIC(7, 2),      -- Plus vs minus Elo difference

    -- Timestamps
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

-- Indexes for efficient worker polling and status queries
CREATE INDEX IF NOT EXISTS idx_spsa_iteration_number ON spsa_iterations(iteration_number);
CREATE INDEX IF NOT EXISTS idx_spsa_status ON spsa_iterations(status);
CREATE INDEX IF NOT EXISTS idx_spsa_status_games ON spsa_iterations(status, games_played, target_games);
