"""
SPSA worker implementation.

Polls the database for pending SPSA iterations, builds engines from
database parameters if needed, and runs games between perturbed engine pairs.
Results are aggregated (no individual game saves).
"""

import os
import random
import socket
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from compete.game import play_game, GameConfig, play_game_from_config
from compete.openings import OPENING_BOOK
from compete.spsa.build import build_spsa_engines, get_rusty_rival_path


def load_config() -> dict:
    """Load SPSA configuration."""
    config_file = Path(__file__).parent / 'config.toml'
    with open(config_file, 'rb') as f:
        return tomllib.load(f)


def get_pending_iteration():
    """
    Find the current SPSA iteration that needs games.
    Returns iteration data dict or None if no work available.
    """
    from web.app import create_app
    from web.database import db
    from web.models import SpsaIteration

    app = create_app()
    with app.app_context():
        # Find iteration that needs games (pending or in_progress, not yet complete)
        iteration = SpsaIteration.query.filter(
            SpsaIteration.status.in_(['pending', 'in_progress']),
            SpsaIteration.games_played < SpsaIteration.target_games
        ).order_by(SpsaIteration.iteration_number.desc()).first()

        if not iteration:
            return None

        # Mark as in_progress if pending
        if iteration.status == 'pending':
            iteration.status = 'in_progress'
            db.session.commit()

        # Return data dict (to avoid detached instance issues)
        return {
            'id': iteration.id,
            'iteration_number': iteration.iteration_number,
            'plus_engine_path': iteration.plus_engine_path,
            'minus_engine_path': iteration.minus_engine_path,
            'plus_parameters': iteration.plus_parameters,
            'minus_parameters': iteration.minus_parameters,
            'timelow_ms': iteration.timelow_ms,
            'timehigh_ms': iteration.timehigh_ms,
            'target_games': iteration.target_games,
            'games_played': iteration.games_played,
        }


def update_iteration_results(iteration_id: int, games: int, plus_wins: int, minus_wins: int, draws: int):
    """
    Atomically update iteration results.
    Uses SQL increment to avoid race conditions between workers.
    """
    from web.app import create_app
    from web.database import db
    from web.models import SpsaIteration

    app = create_app()
    with app.app_context():
        # Atomic increment using SQL
        db.session.execute(
            db.text("""
                UPDATE spsa_iterations
                SET games_played = games_played + :games,
                    plus_wins = plus_wins + :plus_wins,
                    minus_wins = minus_wins + :minus_wins,
                    draws = draws + :draws
                WHERE id = :id
            """),
            {
                'games': games,
                'plus_wins': plus_wins,
                'minus_wins': minus_wins,
                'draws': draws,
                'id': iteration_id
            }
        )
        db.session.commit()


def ensure_engines_built(iteration: dict, config: dict, force_rebuild: bool = False) -> tuple[str, str]:
    """
    Ensure engine binaries exist, building from database parameters if needed.

    Args:
        iteration: Iteration data from database
        config: SPSA configuration
        force_rebuild: If True, rebuild even if binaries exist (for new iterations)

    Returns:
        (plus_path, minus_path) - paths to engine binaries
    """
    # Compute LOCAL paths (don't use database paths which may be from another OS)
    output_base = Path(config['build']['engines_output_path'])
    if not output_base.is_absolute():
        chess_compete_dir = Path(__file__).parent.parent.parent
        output_base = chess_compete_dir / output_base

    binary_name = 'rusty-rival.exe' if os.name == 'nt' else 'rusty-rival'
    plus_dir = output_base / config['build']['plus_engine_name']
    minus_dir = output_base / config['build']['minus_engine_name']
    plus_path = plus_dir / binary_name
    minus_path = minus_dir / binary_name

    # Check if both binaries exist locally (skip if force_rebuild)
    if not force_rebuild and plus_path.exists() and minus_path.exists():
        return str(plus_path), str(minus_path)

    # Need to build engines from parameters
    print(f"\n  Building engines for iteration {iteration['iteration_number']}...")

    src_path = get_rusty_rival_path(config)

    # Build with parameters from database
    plus_params = iteration['plus_parameters']
    minus_params = iteration['minus_parameters']

    try:
        new_plus_path, new_minus_path = build_spsa_engines(
            src_path, output_base,
            plus_params, minus_params,
            config['build']['plus_engine_name'],
            config['build']['minus_engine_name']
        )
        print(f"  Built engines: {new_plus_path}, {new_minus_path}")
        return str(new_plus_path), str(new_minus_path)
    except Exception as e:
        raise RuntimeError(f"Failed to build engines: {e}")


def play_spsa_batch(plus_path: str, minus_path: str, timelow_ms: int, timehigh_ms: int,
                    batch_size: int, concurrency: int) -> tuple[int, int, int, int, float, float]:
    """
    Play a batch of games between plus and minus engines.

    Args:
        plus_path: Path to the plus-perturbed engine binary
        minus_path: Path to the minus-perturbed engine binary
        timelow_ms: Minimum time per move in milliseconds
        timehigh_ms: Maximum time per move in milliseconds
        batch_size: Number of games to play in this batch
        concurrency: Number of parallel games

    Returns:
        (plus_wins, minus_wins, draws, errors, plus_avg_nps, minus_avg_nps)
        - errors are NOT counted as games
        - NPS values are averages across games (0 if no data)
    """
    plus_wins = 0
    minus_wins = 0
    draws = 0
    errors = 0
    plus_nps_total = 0
    minus_nps_total = 0
    plus_nps_count = 0
    minus_nps_count = 0

    if concurrency > 1:
        # Parallel execution
        configs = []
        for i in range(batch_size):
            opening_fen, opening_name = random.choice(OPENING_BOOK)
            time_ms = random.uniform(timelow_ms, timehigh_ms)
            time_per_move = time_ms / 1000.0

            # Alternate colors: even games plus is white, odd games minus is white
            if i % 2 == 0:
                config = GameConfig(
                    game_index=i,
                    white_name="spsa-plus",
                    black_name="spsa-minus",
                    white_path=plus_path,
                    black_path=minus_path,
                    white_uci_options=None,
                    black_uci_options=None,
                    time_per_move=time_per_move,
                    opening_fen=opening_fen,
                    opening_name=opening_name,
                    is_engine1_white=True  # plus is white
                )
            else:
                config = GameConfig(
                    game_index=i,
                    white_name="spsa-minus",
                    black_name="spsa-plus",
                    white_path=minus_path,
                    black_path=plus_path,
                    white_uci_options=None,
                    black_uci_options=None,
                    time_per_move=time_per_move,
                    opening_fen=opening_fen,
                    opening_name=opening_name,
                    is_engine1_white=False  # plus is black
                )
            configs.append(config)

        # Run games in parallel
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(play_game_from_config, c): c for c in configs}

            for future in as_completed(futures):
                config = futures[future]
                try:
                    result = future.result()

                    # Determine winner from plus engine's perspective
                    if config.is_engine1_white:
                        # Plus was white
                        if result.result == "1-0":
                            plus_wins += 1
                        elif result.result == "0-1":
                            minus_wins += 1
                        else:
                            draws += 1
                        # Collect NPS (plus=white, minus=black)
                        if result.white_nps:
                            plus_nps_total += result.white_nps
                            plus_nps_count += 1
                        if result.black_nps:
                            minus_nps_total += result.black_nps
                            minus_nps_count += 1
                    else:
                        # Plus was black
                        if result.result == "0-1":
                            plus_wins += 1
                        elif result.result == "1-0":
                            minus_wins += 1
                        else:
                            draws += 1
                        # Collect NPS (minus=white, plus=black)
                        if result.white_nps:
                            minus_nps_total += result.white_nps
                            minus_nps_count += 1
                        if result.black_nps:
                            plus_nps_total += result.black_nps
                            plus_nps_count += 1
                except Exception as e:
                    print(f"  Error in game: {e}")
                    errors += 1

    else:
        # Sequential execution
        for i in range(batch_size):
            opening_fen, opening_name = random.choice(OPENING_BOOK)
            time_ms = random.uniform(timelow_ms, timehigh_ms)
            time_per_move = time_ms / 1000.0

            try:
                # Alternate colors
                if i % 2 == 0:
                    # Plus is white
                    result, _ = play_game(
                        plus_path, minus_path,
                        "spsa-plus", "spsa-minus",
                        time_per_move, opening_fen, opening_name
                    )
                    if result == "1-0":
                        plus_wins += 1
                    elif result == "0-1":
                        minus_wins += 1
                    else:
                        draws += 1
                else:
                    # Plus is black
                    result, _ = play_game(
                        minus_path, plus_path,
                        "spsa-minus", "spsa-plus",
                        time_per_move, opening_fen, opening_name
                    )
                    if result == "0-1":
                        plus_wins += 1
                    elif result == "1-0":
                        minus_wins += 1
                    else:
                        draws += 1
            except Exception as e:
                print(f"  Error in game: {e}")
                errors += 1

    # Calculate average NPS
    plus_avg_nps = plus_nps_total / plus_nps_count if plus_nps_count > 0 else 0
    minus_avg_nps = minus_nps_total / minus_nps_count if minus_nps_count > 0 else 0

    return plus_wins, minus_wins, draws, errors, plus_avg_nps, minus_avg_nps


def run_spsa_worker(concurrency: int = 1, batch_size: int = 10, poll_interval: int = 10):
    """
    Run the SPSA worker loop.

    Continuously polls for pending SPSA iterations, builds engines if needed,
    and runs games. Results are aggregated and saved to the database.

    Args:
        concurrency: Number of games to run in parallel
        batch_size: Number of games per batch before updating database
        poll_interval: Seconds to wait when no work is available
    """
    hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())

    # Load config for build settings
    config = load_config()

    print(f"\n{'='*60}")
    print("SPSA WORKER MODE")
    print(f"{'='*60}")
    print(f"Host: {hostname}")
    print(f"Concurrency: {concurrency} games in parallel")
    print(f"Batch size: {batch_size} games per update")
    print(f"Poll interval: {poll_interval}s when idle")
    print(f"Opening book: {len(OPENING_BOOK)} positions")
    print(f"Rusty-rival source: {get_rusty_rival_path(config)}")
    print(f"{'='*60}")
    print("\nWaiting for work...")

    games_total = 0
    current_iteration = None
    engines_built_for_iteration = None  # Track which iteration we built engines for

    while True:
        try:
            # Get current iteration
            iteration = get_pending_iteration()

            if not iteration:
                print(".", end="", flush=True)
                time.sleep(poll_interval)
                continue

            # Check if we switched to a new iteration
            iteration_changed = current_iteration != iteration['iteration_number']
            if iteration_changed:
                current_iteration = iteration['iteration_number']
                print(f"\n\nIteration {current_iteration}: {iteration['games_played']}/{iteration['target_games']} games")
                print(f"  Time:  {iteration['timelow_ms']}-{iteration['timehigh_ms']}ms/move")

            # Ensure engines are built (force rebuild if iteration changed)
            need_rebuild = iteration_changed or engines_built_for_iteration != current_iteration
            try:
                plus_path, minus_path = ensure_engines_built(iteration, config, force_rebuild=need_rebuild)
                engines_built_for_iteration = current_iteration
            except RuntimeError as e:
                print(f"\n  ERROR: {e}")
                print("  Waiting before retry...")
                time.sleep(poll_interval * 2)
                continue

            print(f"  Plus:  {plus_path}")
            print(f"  Minus: {minus_path}")

            # Check if iteration is complete
            remaining = iteration['target_games'] - iteration['games_played']
            if remaining <= 0:
                print(f"\nIteration {current_iteration} complete!")
                current_iteration = None
                time.sleep(1)  # Brief pause before checking for next iteration
                continue

            # Adjust batch size if near completion
            actual_batch = min(batch_size, remaining)

            # Play batch of games
            print(f"\n  Playing {actual_batch} games...", end=" ", flush=True)
            plus_wins, minus_wins, draws, errors, plus_nps, minus_nps = play_spsa_batch(
                plus_path,
                minus_path,
                iteration['timelow_ms'],
                iteration['timehigh_ms'],
                actual_batch,
                concurrency
            )

            # Only count successful games (errors don't count toward total)
            completed_games = plus_wins + minus_wins + draws

            if completed_games > 0:
                # Update database with actual completed games
                update_iteration_results(iteration['id'], completed_games, plus_wins, minus_wins, draws)
                games_total += completed_games

            # Format NPS for display (in thousands)
            nps_str = ""
            if plus_nps > 0 or minus_nps > 0:
                plus_knps = plus_nps / 1000 if plus_nps > 0 else 0
                minus_knps = minus_nps / 1000 if minus_nps > 0 else 0
                avg_knps = (plus_nps + minus_nps) / 2000 if plus_nps > 0 and minus_nps > 0 else max(plus_knps, minus_knps)
                nps_str = f" NPS: {avg_knps:.0f}k"

            if errors > 0:
                print(f"+{plus_wins} -{minus_wins} ={draws} errors={errors}{nps_str} (total: {games_total})")
            else:
                print(f"+{plus_wins} -{minus_wins} ={draws}{nps_str} (total: {games_total})")

        except KeyboardInterrupt:
            print(f"\n\nWorker stopped. Total games played: {games_total}")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(poll_interval)
