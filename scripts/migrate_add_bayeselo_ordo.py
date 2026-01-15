#!/usr/bin/env python3
"""
Migration script to add BayesElo and Ordo rating columns.

This script:
1. Adds bayes_elo column to elo_filter_ratings table
2. Adds ordo column to elo_filter_ratings table

Both columns are nullable - they will be populated when Force Recalculate is clicked.

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
        print("Starting migration to add BayesElo and Ordo columns...")
        print()

        # Check if bayes_elo column already exists
        result = db.session.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'elo_filter_ratings' AND column_name = 'bayes_elo'
        """))
        bayes_exists = result.fetchone() is not None

        # Check if ordo column already exists
        result = db.session.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'elo_filter_ratings' AND column_name = 'ordo'
        """))
        ordo_exists = result.fetchone() is not None

        if bayes_exists and ordo_exists:
            print("Both columns already exist in elo_filter_ratings table.")
            print("Migration already completed.")
            return

        print("=== Adding new columns ===")

        if not bayes_exists:
            db.session.execute(text("""
                ALTER TABLE elo_filter_ratings
                ADD COLUMN bayes_elo NUMERIC(7, 2)
            """))
            print("  -> Added bayes_elo column")
        else:
            print("  -> bayes_elo column already exists")

        if not ordo_exists:
            db.session.execute(text("""
                ALTER TABLE elo_filter_ratings
                ADD COLUMN ordo NUMERIC(7, 2)
            """))
            print("  -> Added ordo column")
        else:
            print("  -> ordo column already exists")

        db.session.commit()
        print()

        print("=== Migration Complete ===")
        print("BayesElo and Ordo columns have been added.")
        print("Click 'Force Recalculate' on the dashboard to populate these ratings.")


if __name__ == '__main__':
    main()
