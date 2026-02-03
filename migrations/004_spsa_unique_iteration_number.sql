-- Migration: Enforce unique SPSA iteration numbers
-- Prevents duplicate iteration rows if multiple masters run concurrently.

ALTER TABLE spsa_iterations
    ADD CONSTRAINT spsa_iterations_iteration_number_unique UNIQUE (iteration_number);
