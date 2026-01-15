#!/usr/bin/env python3
"""
Migration script to add engine_type filter column.

This script:
1. Adds engine_type column to elo_filter_cache table
2. Updates the unique constraint to include engine_type
3. Clears existing cache entries (since the constraint changes)

Run this script once to migrate the database.
"""

import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from web.app import create_app
from web.database import db


def main():
    app = create_app()

    with app.app_context():
        print("Starting migration to add engine_type filter...")
        print()

        # Check if column already exists
        result = db.session.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'elo_filter_cache' AND column_name = 'engine_type'
        """))
        if result.fetchone():
            print("Column 'engine_type' already exists in elo_filter_cache table.")
            print("Migration already completed.")
            return

        print("=== Phase 1: Clear existing cache ===")
        # Clear existing cache entries since we're changing the unique constraint
        result = db.session.execute(text("SELECT COUNT(*) FROM elo_filter_ratings"))
        count_ratings = result.scalar()
        result = db.session.execute(text("SELECT COUNT(*) FROM elo_filter_cache"))
        count_cache = result.scalar()
        print(f"Existing cache entries: {count_cache}")
        print(f"Existing rating entries: {count_ratings}")

        if count_ratings > 0 or count_cache > 0:
            db.session.execute(text("DELETE FROM elo_filter_ratings"))
            db.session.execute(text("DELETE FROM elo_filter_cache"))
            db.session.commit()
            print(f"  -> Deleted {count_ratings} rating entries and {count_cache} cache entries")
        print()

        print("=== Phase 2: Drop existing unique constraint ===")
        # Drop the old unique constraint (name may vary)
        try:
            db.session.execute(text("""
                ALTER TABLE elo_filter_cache
                DROP CONSTRAINT IF EXISTS elo_filter_cache_min_time_ms_max_time_ms_hostname_key
            """))
            db.session.commit()
            print("  -> Dropped old unique constraint")
        except Exception as e:
            print(f"  -> Note: Could not drop old constraint (may not exist): {e}")
            db.session.rollback()
        print()

        print("=== Phase 3: Add engine_type column ===")
        db.session.execute(text("""
            ALTER TABLE elo_filter_cache
            ADD COLUMN engine_type VARCHAR(20)
        """))
        db.session.commit()
        print("  -> Added engine_type column (nullable, NULL = all engines)")
        print()

        print("=== Phase 4: Add new unique constraint ===")
        db.session.execute(text("""
            ALTER TABLE elo_filter_cache
            ADD CONSTRAINT elo_filter_cache_min_time_ms_max_time_ms_hostname_engine_type_key
            UNIQUE (min_time_ms, max_time_ms, hostname, engine_type)
        """))
        db.session.commit()
        print("  -> Added new unique constraint including engine_type")
        print()

        print("=== Migration Complete ===")
        print("The engine_type filter is now available.")
        print("Cache has been cleared - ELO ratings will be recalculated on first use.")


if __name__ == '__main__':
    main()
