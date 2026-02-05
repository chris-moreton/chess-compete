-- SPSA Worker Tracking Tables
-- Tracks worker activity and performance for monitoring distributed workers

-- Summary table: current state of each worker
CREATE TABLE IF NOT EXISTS spsa_workers (
    id SERIAL PRIMARY KEY,
    worker_name VARCHAR(100) UNIQUE NOT NULL,
    last_iteration_id INTEGER REFERENCES spsa_iterations(id),
    last_phase VARCHAR(20),  -- 'spsa' or 'ref'
    total_games INTEGER NOT NULL DEFAULT 0,
    total_spsa_games INTEGER NOT NULL DEFAULT 0,
    total_ref_games INTEGER NOT NULL DEFAULT 0,
    avg_nps INTEGER,  -- Rolling average nodes per second
    last_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    first_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- History table: individual heartbeats/reports
CREATE TABLE IF NOT EXISTS spsa_worker_heartbeats (
    id SERIAL PRIMARY KEY,
    worker_id INTEGER NOT NULL REFERENCES spsa_workers(id) ON DELETE CASCADE,
    iteration_id INTEGER REFERENCES spsa_iterations(id),
    phase VARCHAR(20) NOT NULL,  -- 'spsa' or 'ref'
    games_reported INTEGER NOT NULL,
    avg_nps INTEGER,  -- NPS for this batch
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

-- Indexes for efficient queries
CREATE INDEX IF NOT EXISTS idx_spsa_workers_last_seen ON spsa_workers(last_seen_at);
CREATE INDEX IF NOT EXISTS idx_spsa_worker_heartbeats_worker ON spsa_worker_heartbeats(worker_id);
CREATE INDEX IF NOT EXISTS idx_spsa_worker_heartbeats_created ON spsa_worker_heartbeats(created_at);
