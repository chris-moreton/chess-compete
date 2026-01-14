#!/usr/bin/env python3
"""
Migration script for filtered ELO ratings feature.

This script:
1. Deletes games without PGN
2. Deletes games for inactive engines
3. Deletes inactive engines
4. Adds new columns to games table (time_per_move_ms, hostname)
5. Creates new filter cache tables
6. Adds indexes
7. Backfills time_per_move_ms and hostname from existing data
8. Drops old elo_ratings table

Run this script once to migrate the database.
"""

import os
import re
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

from sqlalchemy import text
from web.app import create_app
from web.database import db


def parse_time_control(time_control_str):
    """Parse '1.50s/move' -> 1500 (ms)"""
    if not time_control_str:
        return None
    match = re.match(r'([\d.]+)s/move', time_control_str)
    if match:
        return int(float(match.group(1)) * 1000)
    return None


def parse_hostname_from_pgn(pgn_str):
    """Extract Site header from PGN"""
    if not pgn_str:
        return None
    match = re.search(r'\[Site "([^"]+)"\]', pgn_str)
    return match.group(1) if match else None


def main():
    app = create_app()

    with app.app_context():
        print("Starting migration to filtered ELO ratings...")
        print()

        # Phase 1: Cleanup
        print("=== Phase 1: Database Cleanup ===")

        # Count games without PGN
        result = db.session.execute(text("SELECT COUNT(*) FROM games WHERE pgn IS NULL OR pgn = ''"))
        count_no_pgn = result.scalar()
        print(f"Games without PGN: {count_no_pgn}")

        if count_no_pgn > 0:
            db.session.execute(text("DELETE FROM games WHERE pgn IS NULL OR pgn = ''"))
            db.session.commit()
            print(f"  -> Deleted {count_no_pgn} games without PGN")

        # Count inactive engines
        result = db.session.execute(text("SELECT COUNT(*) FROM engines WHERE active = FALSE"))
        count_inactive = result.scalar()
        print(f"Inactive engines: {count_inactive}")

        if count_inactive > 0:
            # Count games for inactive engines
            result = db.session.execute(text("""
                SELECT COUNT(*) FROM games
                WHERE white_engine_id IN (SELECT id FROM engines WHERE active = FALSE)
                   OR black_engine_id IN (SELECT id FROM engines WHERE active = FALSE)
            """))
            count_inactive_games = result.scalar()
            print(f"  Games for inactive engines: {count_inactive_games}")

            # Delete games for inactive engines
            db.session.execute(text("""
                DELETE FROM games
                WHERE white_engine_id IN (SELECT id FROM engines WHERE active = FALSE)
                   OR black_engine_id IN (SELECT id FROM engines WHERE active = FALSE)
            """))
            db.session.commit()
            print(f"  -> Deleted {count_inactive_games} games for inactive engines")

            # Delete elo_ratings for inactive engines
            db.session.execute(text("""
                DELETE FROM elo_ratings
                WHERE engine_id IN (SELECT id FROM engines WHERE active = FALSE)
            """))
            db.session.commit()

            # Delete inactive engines
            db.session.execute(text("DELETE FROM engines WHERE active = FALSE"))
            db.session.commit()
            print(f"  -> Deleted {count_inactive} inactive engines")

        print()

        # Phase 2: Add new columns
        print("=== Phase 2: Schema Changes ===")

        # Check if columns already exist
        result = db.session.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'games' AND column_name = 'time_per_move_ms'
        """))
        if result.fetchone() is None:
            db.session.execute(text("ALTER TABLE games ADD COLUMN time_per_move_ms INTEGER"))
            db.session.commit()
            print("Added column: games.time_per_move_ms")
        else:
            print("Column games.time_per_move_ms already exists")

        result = db.session.execute(text("""
            SELECT column_name FROM information_schema.columns
            WHERE table_name = 'games' AND column_name = 'hostname'
        """))
        if result.fetchone() is None:
            db.session.execute(text("ALTER TABLE games ADD COLUMN hostname VARCHAR(100)"))
            db.session.commit()
            print("Added column: games.hostname")
        else:
            print("Column games.hostname already exists")

        # Create elo_filter_cache table
        result = db.session.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'elo_filter_cache'
        """))
        if result.fetchone() is None:
            db.session.execute(text("""
                CREATE TABLE elo_filter_cache (
                    id SERIAL PRIMARY KEY,
                    min_time_ms INTEGER NOT NULL DEFAULT 0,
                    max_time_ms INTEGER NOT NULL DEFAULT 999999999,
                    hostname VARCHAR(100),
                    last_game_id INTEGER NOT NULL DEFAULT 0,
                    created_at TIMESTAMP DEFAULT NOW(),
                    UNIQUE(min_time_ms, max_time_ms, hostname)
                )
            """))
            db.session.commit()
            print("Created table: elo_filter_cache")
        else:
            print("Table elo_filter_cache already exists")

        # Create elo_filter_ratings table
        result = db.session.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'elo_filter_ratings'
        """))
        if result.fetchone() is None:
            db.session.execute(text("""
                CREATE TABLE elo_filter_ratings (
                    id SERIAL PRIMARY KEY,
                    filter_id INTEGER NOT NULL REFERENCES elo_filter_cache(id) ON DELETE CASCADE,
                    engine_id INTEGER NOT NULL REFERENCES engines(id) ON DELETE CASCADE,
                    elo NUMERIC(7, 2) NOT NULL,
                    games_played INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(filter_id, engine_id)
                )
            """))
            db.session.commit()
            print("Created table: elo_filter_ratings")
        else:
            print("Table elo_filter_ratings already exists")

        # Create indexes
        try:
            db.session.execute(text("CREATE INDEX idx_games_time_per_move ON games(time_per_move_ms)"))
            db.session.commit()
            print("Created index: idx_games_time_per_move")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("Index idx_games_time_per_move already exists")
                db.session.rollback()
            else:
                raise

        try:
            db.session.execute(text("CREATE INDEX idx_games_hostname ON games(hostname)"))
            db.session.commit()
            print("Created index: idx_games_hostname")
        except Exception as e:
            if "already exists" in str(e).lower():
                print("Index idx_games_hostname already exists")
                db.session.rollback()
            else:
                raise

        print()

        # Phase 3: Backfill data
        print("=== Phase 3: Backfill Data ===")

        # Get games that need backfill
        result = db.session.execute(text("""
            SELECT id, time_control, pgn FROM games
            WHERE time_per_move_ms IS NULL OR hostname IS NULL
        """))
        games_to_update = result.fetchall()
        print(f"Games to backfill: {len(games_to_update)}")

        updated_count = 0
        for game_id, time_control, pgn in games_to_update:
            time_ms = parse_time_control(time_control)
            hostname = parse_hostname_from_pgn(pgn)

            if time_ms is not None or hostname is not None:
                db.session.execute(text("""
                    UPDATE games SET time_per_move_ms = :time_ms, hostname = :hostname
                    WHERE id = :game_id
                """), {"time_ms": time_ms, "hostname": hostname, "game_id": game_id})
                updated_count += 1

                if updated_count % 1000 == 0:
                    db.session.commit()
                    print(f"  Updated {updated_count} games...")

        db.session.commit()
        print(f"  -> Backfilled {updated_count} games")

        print()

        # Phase 4: Drop old elo_ratings table
        print("=== Phase 4: Drop Old elo_ratings Table ===")

        result = db.session.execute(text("""
            SELECT table_name FROM information_schema.tables
            WHERE table_name = 'elo_ratings'
        """))
        if result.fetchone() is not None:
            db.session.execute(text("DROP TABLE elo_ratings"))
            db.session.commit()
            print("Dropped table: elo_ratings")
        else:
            print("Table elo_ratings already dropped")

        print()
        print("=== Migration Complete ===")

        # Summary
        result = db.session.execute(text("SELECT COUNT(*) FROM games"))
        total_games = result.scalar()
        result = db.session.execute(text("SELECT COUNT(*) FROM engines"))
        total_engines = result.scalar()
        result = db.session.execute(text("SELECT COUNT(DISTINCT hostname) FROM games WHERE hostname IS NOT NULL"))
        unique_hostnames = result.scalar()

        print(f"Total games: {total_games}")
        print(f"Total engines: {total_engines}")
        print(f"Unique hostnames: {unique_hostnames}")


if __name__ == "__main__":
    main()
