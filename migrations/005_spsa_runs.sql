-- Migration: Add SPSA runs table to group iterations into logical tuning runs
-- This allows keeping historical data while starting fresh tuning sessions

-- 1. Create the spsa_runs table
CREATE TABLE IF NOT EXISTS spsa_runs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

-- 2. Add run_id column to spsa_iterations (nullable initially for migration)
ALTER TABLE spsa_iterations
    ADD COLUMN IF NOT EXISTS run_id INTEGER REFERENCES spsa_runs(id);

-- 3. Create a legacy run for all existing iterations
INSERT INTO spsa_runs (name, description, is_active)
VALUES ('Legacy (pre-fix)', 'Iterations from before SPSA scaling fixes. Data may be unreliable due to step^2 bug.', FALSE);

-- 4. Update all existing iterations to point to the legacy run
UPDATE spsa_iterations
SET run_id = (SELECT id FROM spsa_runs WHERE name = 'Legacy (pre-fix)')
WHERE run_id IS NULL;

-- 5. Drop the old unique constraint on iteration_number alone (if it exists)
-- This was added in a previous migration that may or may not have been applied
ALTER TABLE spsa_iterations
    DROP CONSTRAINT IF EXISTS spsa_iterations_iteration_number_unique;

-- 6. Add new unique constraint on (run_id, iteration_number)
-- Iteration numbers are now unique per run, not globally
ALTER TABLE spsa_iterations
    ADD CONSTRAINT uq_spsa_run_iteration UNIQUE (run_id, iteration_number);

-- 7. Add index on run_id for efficient filtering
CREATE INDEX IF NOT EXISTS idx_spsa_run_id ON spsa_iterations(run_id);

-- 8. Create an active run for new tuning
INSERT INTO spsa_runs (name, description, is_active)
VALUES ('Run 2 - Fixed scaling', 'Fresh tuning run with corrected SPSA scaling (step not step^2).', TRUE);
