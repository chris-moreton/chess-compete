"""
Database operations and Elo rating management.
"""

import os
import sys
import time
import traceback
from datetime import datetime
from functools import wraps
from pathlib import Path

from dotenv import load_dotenv

from compete.constants import (
    DEFAULT_ELO,
    DB_MAX_RETRIES,
    DB_RETRY_BASE_DELAY,
    DB_RETRY_MAX_DELAY,
)

# Load environment variables
load_dotenv(Path(__file__).parent.parent / '.env')

# Check if database is configured
DB_ENABLED = os.getenv('DATABASE_URL') is not None

# Add project root to path for web module imports
if DB_ENABLED:
    sys.path.insert(0, str(Path(__file__).parent.parent))

# Cache the app instance to avoid recreating it for every DB operation
_app_instance = None


def db_retry(func):
    """Decorator to retry database operations on connection errors with exponential backoff."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        last_error = None
        for attempt in range(DB_MAX_RETRIES):
            try:
                return func(*args, **kwargs)
            except Exception as e:
                last_error = e
                error_str = str(e).lower()
                # Check if it's a connection error worth retrying
                if any(msg in error_str for msg in ['connection', 'closed', 'terminated', 'timeout', 'operationalerror']):
                    if attempt < DB_MAX_RETRIES - 1:
                        # Exponential backoff: 5, 10, 20, 40, 60, 60, 60... (capped at max)
                        wait_time = min(DB_RETRY_BASE_DELAY * (2 ** attempt), DB_RETRY_MAX_DELAY)
                        print(f"Database connection error, retrying in {wait_time}s... (attempt {attempt + 1}/{DB_MAX_RETRIES})")
                        time.sleep(wait_time)
                        # Reset the app instance to force new connection
                        global _app_instance
                        _app_instance = None
                        continue
                # Not a connection error, or out of retries - raise immediately
                raise
        # All retries exhausted
        raise last_error
    return wrapper


def _get_app():
    """Get or create the Flask app instance for DB operations."""
    global _app_instance
    if _app_instance is None and DB_ENABLED:
        from web.app import create_app
        _app_instance = create_app()
    return _app_instance


@db_retry
def load_elo_ratings() -> dict:
    """
    Load Elo ratings calculated from games.
    Uses the filter cache system with default (unfiltered) parameters.
    Returns dict: {engine_name: {"elo": float, "games": int}}
    """
    if not DB_ENABLED:
        print("Error: DATABASE_URL not configured. Set it in .env file.")
        sys.exit(1)

    try:
        from web.queries import recalculate_elos_incremental
        from web.models import Engine
        from web.app import create_app

        app = create_app()
        with app.app_context():
            # Use unfiltered calculation (min=0, max=huge, hostname=None)
            elo_dict = recalculate_elos_incremental(0, 999999999, None)

            # Build engine name -> id mapping
            engines = {e.id: e.name for e in Engine.query.all()}

            ratings = {}
            for engine_id, (elo, games) in elo_dict.items():
                name = engines.get(engine_id)
                if name:
                    ratings[name] = {"elo": float(elo), "games": games}
            return ratings
    except Exception as e:
        print(f"Error: Failed to load ratings from database: {e}")
        traceback.print_exc()
        sys.exit(1)


@db_retry
def _save_game_to_db_impl(white: str, black: str, result: str, time_control: str,
                          white_score: float, black_score: float,
                          opening_name: str = None, opening_fen: str = None,
                          pgn: str = None, is_rated: bool = True,
                          time_per_move_ms: int = None, hostname: str = None):
    """Internal implementation with retry decorator."""
    from web.database import db
    from web.models import Engine, Game

    app = _get_app()
    with app.app_context():
        white_engine = Engine.query.filter_by(name=white).first()
        if not white_engine:
            white_engine = Engine(name=white, active=True)
            db.session.add(white_engine)
            db.session.flush()

        black_engine = Engine.query.filter_by(name=black).first()
        if not black_engine:
            black_engine = Engine(name=black, active=True)
            db.session.add(black_engine)
            db.session.flush()

        game = Game(
            white_engine_id=white_engine.id,
            black_engine_id=black_engine.id,
            result=result,
            white_score=white_score,
            black_score=black_score,
            date_played=datetime.now().date(),
            time_control=time_control,
            time_per_move_ms=time_per_move_ms,
            hostname=hostname,
            opening_name=opening_name,
            opening_fen=opening_fen,
            pgn=pgn,
            is_rated=is_rated
        )
        db.session.add(game)
        db.session.commit()


def save_game_to_db(white: str, black: str, result: str, time_control: str,
                    opening_name: str = None, opening_fen: str = None,
                    pgn: str = None, is_rated: bool = True,
                    time_per_move_ms: int = None, hostname: str = None):
    """
    Save a game result to the database.
    Uses automatic retry with exponential backoff on connection errors.
    If all retries fail, logs the error but doesn't crash the competition.

    Args:
        is_rated: If False, game won't appear in H2H grid (used for EPD test games)
        time_per_move_ms: Time per move in milliseconds
        hostname: Machine that played the game
    """
    # Calculate scores first - incomplete games should return early without DB interaction
    if result == "1-0":
        white_score, black_score = 1.0, 0.0
    elif result == "0-1":
        white_score, black_score = 0.0, 1.0
    elif result == "1/2-1/2":
        white_score, black_score = 0.5, 0.5
    else:
        return  # Incomplete game, don't record

    try:
        _save_game_to_db_impl(white, black, result, time_control,
                              white_score, black_score,
                              opening_name, opening_fen, pgn, is_rated,
                              time_per_move_ms, hostname)
    except Exception as e:
        print(f"Error: Failed to save game to database after {DB_MAX_RETRIES} retries: {e}")
        traceback.print_exc()
        print("Warning: Game result not saved, but competition will continue.")


def derive_elo_from_name(engine_name: str) -> float:
    """
    Derive initial Elo rating from engine name.

    Rules:
    - sf-XXXX (Stockfish with numeric suffix) -> XXXX
    - v* (Rusty Rival, starts with v followed by digit) -> 2600
    - Anything else -> DEFAULT_ELO
    """
    # Stockfish: sf-XXXX where XXXX is numeric
    if engine_name.startswith("sf-") and len(engine_name) > 3 and engine_name[3:].isdigit():
        return float(engine_name[3:])

    # Rusty Rival: starts with v followed by a digit
    if len(engine_name) >= 2 and engine_name.startswith("v") and engine_name[1:2].isdigit():
        return 2600.0

    return DEFAULT_ELO


@db_retry
def get_initial_elo(engine_name: str) -> float:
    """
    Get the initial Elo rating for an engine.

    Priority:
    1. Database (Engine.initial_elo)
    2. Derive from engine name (sf-2400 -> 2400, v* -> 2600)
    3. DEFAULT_ELO
    """
    # Try database first
    try:
        from web.database import db
        from web.models import Engine

        app = _get_app()
        with app.app_context():
            engine = Engine.query.filter_by(name=engine_name).first()
            if engine and engine.initial_elo:
                return float(engine.initial_elo)
    except Exception as e:
        print(f"Error: Failed to get initial Elo from database: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Fall back to name-based derivation
    return derive_elo_from_name(engine_name)


@db_retry
def save_epd_test_run(epd_file: str, total_positions: int, timeout_seconds: float,
                      score_tolerance: int, hostname: str, engine_results: dict):
    """
    Save EPD test run and results to the database.

    Args:
        epd_file: Name of the EPD file tested
        total_positions: Number of positions in the test
        timeout_seconds: Timeout per position
        score_tolerance: Centipawn tolerance for score validation
        hostname: Machine that ran the test
        engine_results: Dict of {engine_name: [(position_data, result), ...]}
            where position_data is (index, position_id, fen, test_type, expected_moves)
            and result is a SolveResult object
    """
    if not DB_ENABLED:
        print("Warning: DATABASE_URL not configured, results not saved.")
        return

    try:
        from web.database import db
        from web.models import Engine, EpdTestRun, EpdTestResult

        app = _get_app()
        with app.app_context():
            # Create the test run record
            run = EpdTestRun(
                epd_file=epd_file,
                total_positions=total_positions,
                timeout_seconds=timeout_seconds,
                score_tolerance=score_tolerance,
                hostname=hostname
            )
            db.session.add(run)
            db.session.flush()  # Get the run ID

            # Process results for each engine
            for engine_name, results_list in engine_results.items():
                # Get or create engine (don't auto-activate if creating)
                engine = Engine.query.filter_by(name=engine_name).first()
                if not engine:
                    engine = Engine(name=engine_name, active=False)
                    db.session.add(engine)
                    db.session.flush()

                # Add results for each position
                for position_data, result in results_list:
                    pos_index, pos_id, fen, test_type, expected_moves = position_data

                    # Extract score components
                    score_cp = None
                    score_mate = None
                    if result.score:
                        if result.score.is_mate():
                            score_mate = result.score.white().mate()
                        else:
                            score_cp = result.score.white().score(mate_score=10000)

                    epd_result = EpdTestResult(
                        run_id=run.id,
                        engine_id=engine.id,
                        position_id=pos_id,
                        position_index=pos_index,
                        fen=fen,
                        test_type=test_type,
                        expected_moves=expected_moves,
                        solved=result.solved,
                        move_found=result.move_found,
                        solve_time_ms=int(result.solve_time * 1000) if result.solve_time else None,
                        final_depth=result.final_depth,
                        score_cp=score_cp,
                        score_mate=score_mate,
                        score_valid=result.score_valid,
                        timed_out=result.timed_out
                    )
                    db.session.add(epd_result)

            db.session.commit()
            print(f"\nResults saved to database (run_id={run.id})")

    except Exception as e:
        print(f"Error: Failed to save EPD test results to database: {e}")
        traceback.print_exc()
        print("Warning: Results not saved, but test completed.")
