-- Migration: Add EPD test result tables
-- Run this against your PostgreSQL database to add EPD test tracking

-- Table for EPD test run metadata
CREATE TABLE IF NOT EXISTS epd_test_runs (
    id SERIAL PRIMARY KEY,
    epd_file VARCHAR(255) NOT NULL,
    total_positions INTEGER NOT NULL,
    timeout_seconds FLOAT NOT NULL,
    score_tolerance INTEGER NOT NULL,
    hostname VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Table for individual position results
CREATE TABLE IF NOT EXISTS epd_test_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES epd_test_runs(id) ON DELETE CASCADE,
    engine_id INTEGER NOT NULL REFERENCES engines(id) ON DELETE CASCADE,
    position_id VARCHAR(255) NOT NULL,
    position_index INTEGER NOT NULL,
    fen TEXT NOT NULL,
    test_type VARCHAR(10) NOT NULL,
    expected_moves VARCHAR(255) NOT NULL,
    solved BOOLEAN NOT NULL,
    move_found VARCHAR(20),
    solve_time_ms INTEGER,
    final_depth INTEGER,
    score_cp INTEGER,
    score_mate INTEGER,
    score_valid BOOLEAN,
    timed_out BOOLEAN NOT NULL DEFAULT FALSE
);

-- Indexes for efficient querying
CREATE INDEX IF NOT EXISTS idx_epd_results_run ON epd_test_results(run_id);
CREATE INDEX IF NOT EXISTS idx_epd_results_engine ON epd_test_results(engine_id);
CREATE INDEX IF NOT EXISTS idx_epd_results_position ON epd_test_results(position_id);
CREATE INDEX IF NOT EXISTS idx_epd_runs_file ON epd_test_runs(epd_file);
