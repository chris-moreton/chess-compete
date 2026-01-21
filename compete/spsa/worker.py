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
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed, wait, FIRST_COMPLETED
from pathlib import Path

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from compete.game import play_game, GameConfig, play_game_from_config
from compete.openings import OPENING_BOOK
from compete.spsa.build import build_spsa_engines, build_engine, get_rusty_rival_path
from compete.spsa.progress import ProgressDisplay


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
            'base_engine_path': iteration.base_engine_path,
            'ref_engine_path': iteration.ref_engine_path,
            'plus_parameters': iteration.plus_parameters,
            'minus_parameters': iteration.minus_parameters,
            'base_parameters': iteration.base_parameters,
            'timelow_ms': iteration.timelow_ms,
            'timehigh_ms': iteration.timehigh_ms,
            'target_games': iteration.target_games,
            'games_played': iteration.games_played,
            'ref_games_played': iteration.ref_games_played,
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


def update_reference_results(iteration_id: int, games: int, wins: int, losses: int, draws: int):
    """
    Atomically update reference game results (base engine vs Stockfish).
    Uses SQL increment to avoid race conditions between workers.
    """
    from web.app import create_app
    from web.database import db

    app = create_app()
    with app.app_context():
        db.session.execute(
            db.text("""
                UPDATE spsa_iterations
                SET ref_games_played = ref_games_played + :games,
                    ref_wins = ref_wins + :wins,
                    ref_losses = ref_losses + :losses,
                    ref_draws = ref_draws + :draws
                WHERE id = :id
            """),
            {
                'games': games,
                'wins': wins,
                'losses': losses,
                'draws': draws,
                'id': iteration_id
            }
        )
        db.session.commit()


def get_cached_iteration(engine_dir: Path) -> int | None:
    """Read the cached iteration number from an engine directory, or None if not cached."""
    cache_file = engine_dir / '.iteration'
    if cache_file.exists():
        try:
            return int(cache_file.read_text().strip())
        except (ValueError, IOError):
            return None
    return None


def write_cached_iteration(engine_dir: Path, iteration_number: int):
    """Write the iteration number to a cache file in the engine directory."""
    cache_file = engine_dir / '.iteration'
    cache_file.write_text(str(iteration_number))


def ensure_engines_built(iteration: dict, config: dict, force_rebuild: bool = False) -> tuple[str, str, str]:
    """
    Ensure engine binaries exist, building from database parameters if needed.

    Uses a cache file (.iteration) to track which iteration the engines were built for,
    allowing reuse across worker restarts.

    Args:
        iteration: Iteration data from database
        config: SPSA configuration
        force_rebuild: If True, rebuild even if binaries exist (for new iterations)

    Returns:
        (plus_path, minus_path, base_path) - paths to engine binaries
    """
    iteration_number = iteration['iteration_number']

    # Compute LOCAL paths (don't use database paths which may be from another OS)
    output_base = Path(config['build']['engines_output_path'])
    if not output_base.is_absolute():
        chess_compete_dir = Path(__file__).parent.parent.parent
        output_base = chess_compete_dir / output_base

    binary_name = 'rusty-rival.exe' if os.name == 'nt' else 'rusty-rival'
    plus_dir = output_base / config['build']['plus_engine_name']
    minus_dir = output_base / config['build']['minus_engine_name']
    base_dir = output_base / config['build']['base_engine_name']
    plus_path = plus_dir / binary_name
    minus_path = minus_dir / binary_name
    base_path = base_dir / binary_name

    # Check if all binaries exist and were built for the current iteration
    if not force_rebuild:
        plus_cached = get_cached_iteration(plus_dir)
        minus_cached = get_cached_iteration(minus_dir)
        base_cached = get_cached_iteration(base_dir)

        all_binaries_exist = plus_path.exists() and minus_path.exists() and base_path.exists()
        all_iterations_match = (plus_cached == iteration_number and
                                minus_cached == iteration_number and
                                base_cached == iteration_number)

        if all_binaries_exist and all_iterations_match:
            print(f"  Using cached engines for iteration {iteration_number}")
            return str(plus_path), str(minus_path), str(base_path)

    # Need to build engines from parameters
    print(f"\n  Building engines for iteration {iteration_number}...")

    src_path = get_rusty_rival_path(config)

    # Build with parameters from database
    plus_params = iteration['plus_parameters']
    minus_params = iteration['minus_parameters']
    base_params = iteration['base_parameters']

    try:
        new_plus_path, new_minus_path = build_spsa_engines(
            src_path, output_base,
            plus_params, minus_params,
            config['build']['plus_engine_name'],
            config['build']['minus_engine_name']
        )
        print(f"  Built plus/minus engines")

        # Write cache files
        write_cached_iteration(plus_dir, iteration_number)
        write_cached_iteration(minus_dir, iteration_number)

        # Build base engine (unperturbed parameters)
        print(f"  Building base engine ({config['build']['base_engine_name']})...")
        if not build_engine(src_path, base_dir, base_params):
            raise RuntimeError("Failed to build base engine")
        print(f"  Built base engine: {base_path}")

        # Write cache file for base
        write_cached_iteration(base_dir, iteration_number)

        return str(new_plus_path), str(new_minus_path), str(base_path)
    except Exception as e:
        raise RuntimeError(f"Failed to build engines: {e}")


def run_spsa_continuous(plus_path: str, minus_path: str, timelow_ms: int, timehigh_ms: int,
                        concurrency: int, batch_size: int,
                        on_batch_complete: callable) -> tuple[int, int, int, int, float, float]:
    """
    Run SPSA games continuously, updating database every batch_size completions.

    Args:
        plus_path: Path to the plus-perturbed engine binary
        minus_path: Path to the minus-perturbed engine binary
        timelow_ms: Minimum time per move in milliseconds
        timehigh_ms: Maximum time per move in milliseconds
        concurrency: Number of parallel games to keep running
        batch_size: How often to update database (every N completed games)
        on_batch_complete: Callback(plus_wins, minus_wins, draws) -> bool
            Called every batch_size games. Returns True to continue, False to stop.

    Returns:
        (total_plus_wins, total_minus_wins, total_draws, total_errors, plus_avg_nps, minus_avg_nps)
    """
    # Totals across all batches
    total_plus_wins = 0
    total_minus_wins = 0
    total_draws = 0
    total_errors = 0
    plus_nps_total = 0
    minus_nps_total = 0
    plus_nps_count = 0
    minus_nps_count = 0

    # Current batch counters (reset after each callback)
    batch_plus_wins = 0
    batch_minus_wins = 0
    batch_draws = 0

    def make_config(game_index: int) -> GameConfig:
        """Create a game config for the given index."""
        opening_fen, opening_name = random.choice(OPENING_BOOK)
        time_ms = random.uniform(timelow_ms, timehigh_ms)
        time_per_move = time_ms / 1000.0

        # Alternate colors: even games plus is white, odd games minus is white
        if game_index % 2 == 0:
            return GameConfig(
                game_index=game_index,
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
            return GameConfig(
                game_index=game_index,
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

    def process_result(config: GameConfig, result):
        """Process a completed game result."""
        nonlocal batch_plus_wins, batch_minus_wins, batch_draws
        nonlocal total_plus_wins, total_minus_wins, total_draws
        nonlocal plus_nps_total, minus_nps_total, plus_nps_count, minus_nps_count

        # Determine winner from plus engine's perspective
        if config.is_engine1_white:
            # Plus was white
            if result.result == "1-0":
                batch_plus_wins += 1
                total_plus_wins += 1
            elif result.result == "0-1":
                batch_minus_wins += 1
                total_minus_wins += 1
            else:
                batch_draws += 1
                total_draws += 1
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
                batch_plus_wins += 1
                total_plus_wins += 1
            elif result.result == "1-0":
                batch_minus_wins += 1
                total_minus_wins += 1
            else:
                batch_draws += 1
                total_draws += 1
            # Collect NPS (minus=white, plus=black)
            if result.white_nps:
                minus_nps_total += result.white_nps
                minus_nps_count += 1
            if result.black_nps:
                plus_nps_total += result.black_nps
                plus_nps_count += 1

    if concurrency > 1:
        # Continuous pipeline: always keep concurrency games running
        # Call on_batch_complete every batch_size games to update database
        progress = ProgressDisplay(label="Game")
        progress.start()

        def make_move_callback(game_idx: int):
            """Create a callback that updates progress for a specific game."""
            return lambda move_count: progress.update_moves(game_idx, move_count)

        keep_adding = True  # Set to False when iteration is complete

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending_futures = {}  # future -> config
            next_game_index = 0
            batch_completed = 0  # Games completed in current batch

            # Submit initial games up to concurrency limit
            while next_game_index < concurrency:
                config = make_config(next_game_index)
                callback = make_move_callback(next_game_index)
                future = executor.submit(play_game_from_config, config, callback)
                pending_futures[future] = config
                progress.start_game(next_game_index)
                next_game_index += 1

            # Process completions and submit new games to keep workers busy
            while pending_futures:
                # Wait for at least one game to complete
                done, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    config = pending_futures.pop(future)
                    try:
                        result = future.result()
                        # Get NPS for display (average of both engines)
                        nps = None
                        if result.white_nps and result.black_nps:
                            nps = (result.white_nps + result.black_nps) // 2
                        elif result.white_nps:
                            nps = result.white_nps
                        elif result.black_nps:
                            nps = result.black_nps

                        progress.finish_game(config.game_index, result.result, nps)
                        process_result(config, result)
                        batch_completed += 1
                    except Exception as e:
                        progress.finish_game(config.game_index, "err")
                        print(f"\n  Error in game: {e}")
                        total_errors += 1
                        batch_completed += 1

                    # Check if we've completed a batch
                    if batch_completed >= batch_size:
                        # Call the callback to update database
                        keep_adding = on_batch_complete(batch_plus_wins, batch_minus_wins, batch_draws)
                        # Reset batch counters
                        batch_plus_wins = 0
                        batch_minus_wins = 0
                        batch_draws = 0
                        batch_completed = 0

                    # Submit a new game if we should keep adding
                    if keep_adding:
                        new_config = make_config(next_game_index)
                        callback = make_move_callback(next_game_index)
                        new_future = executor.submit(play_game_from_config, new_config, callback)
                        pending_futures[new_future] = new_config
                        progress.start_game(next_game_index)
                        next_game_index += 1

            # Final batch update if any games completed since last callback
            if batch_plus_wins + batch_minus_wins + batch_draws > 0:
                on_batch_complete(batch_plus_wins, batch_minus_wins, batch_draws)

        progress.stop()

    else:
        # Sequential execution - run until callback says stop
        game_index = 0
        batch_completed = 0
        keep_going = True

        while keep_going:
            opening_fen, opening_name = random.choice(OPENING_BOOK)
            time_ms = random.uniform(timelow_ms, timehigh_ms)
            time_per_move = time_ms / 1000.0

            try:
                # Alternate colors
                if game_index % 2 == 0:
                    # Plus is white
                    result, _ = play_game(
                        plus_path, minus_path,
                        "spsa-plus", "spsa-minus",
                        time_per_move, opening_fen, opening_name
                    )
                    if result == "1-0":
                        batch_plus_wins += 1
                        total_plus_wins += 1
                    elif result == "0-1":
                        batch_minus_wins += 1
                        total_minus_wins += 1
                    else:
                        batch_draws += 1
                        total_draws += 1
                else:
                    # Plus is black
                    result, _ = play_game(
                        minus_path, plus_path,
                        "spsa-minus", "spsa-plus",
                        time_per_move, opening_fen, opening_name
                    )
                    if result == "0-1":
                        batch_plus_wins += 1
                        total_plus_wins += 1
                    elif result == "1-0":
                        batch_minus_wins += 1
                        total_minus_wins += 1
                    else:
                        batch_draws += 1
                        total_draws += 1
                batch_completed += 1
            except Exception as e:
                print(f"  Error in game: {e}")
                total_errors += 1
                batch_completed += 1

            game_index += 1

            # Check if we've completed a batch
            if batch_completed >= batch_size:
                keep_going = on_batch_complete(batch_plus_wins, batch_minus_wins, batch_draws)
                batch_plus_wins = 0
                batch_minus_wins = 0
                batch_draws = 0
                batch_completed = 0

        # Final batch update if any games completed since last callback
        if batch_plus_wins + batch_minus_wins + batch_draws > 0:
            on_batch_complete(batch_plus_wins, batch_minus_wins, batch_draws)

    # Calculate average NPS
    plus_avg_nps = plus_nps_total / plus_nps_count if plus_nps_count > 0 else 0
    minus_avg_nps = minus_nps_total / minus_nps_count if minus_nps_count > 0 else 0

    return total_plus_wins, total_minus_wins, total_draws, total_errors, plus_avg_nps, minus_avg_nps


def get_reference_engine_path(config: dict) -> str | None:
    """Get the path to the reference engine (Stockfish), or None if disabled."""
    if not config.get('reference', {}).get('enabled', False):
        return None

    ref_path = config['reference'].get('engine_path')
    if not ref_path:
        return None

    ref_path = Path(ref_path)
    if not ref_path.is_absolute():
        chess_compete_dir = Path(__file__).parent.parent.parent
        ref_path = chess_compete_dir / ref_path

    # If path is a directory, search for stockfish binary inside it
    if ref_path.is_dir():
        # Look for stockfish executable in directory
        patterns = ['stockfish*', 'Stockfish*']
        for pattern in patterns:
            matches = list(ref_path.glob(pattern))
            for match in matches:
                if match.is_file():
                    return str(match)
        return None

    # Check for .exe on Windows
    if os.name == 'nt' and not ref_path.suffix:
        ref_path_exe = ref_path.with_suffix('.exe')
        if ref_path_exe.exists():
            return str(ref_path_exe)

    if ref_path.exists() and ref_path.is_file():
        return str(ref_path)

    return None


def play_reference_batch(base_path: str, ref_path: str, timelow_ms: int, timehigh_ms: int,
                         batch_size: int, concurrency: int) -> tuple[int, int, int, int]:
    """
    Play a batch of reference games (base engine vs Stockfish).

    Args:
        base_path: Path to the base (unperturbed) engine binary
        ref_path: Path to the reference engine (Stockfish)
        timelow_ms: Minimum time per move in milliseconds
        timehigh_ms: Maximum time per move in milliseconds
        batch_size: Number of games to play in this batch
        concurrency: Number of parallel games

    Returns:
        (base_wins, base_losses, draws, errors) - from base engine's perspective
    """
    base_wins = 0
    base_losses = 0
    draws = 0
    errors = 0

    def make_config(game_index: int) -> GameConfig:
        """Create a game config for the given index."""
        opening_fen, opening_name = random.choice(OPENING_BOOK)
        time_ms = random.uniform(timelow_ms, timehigh_ms)
        time_per_move = time_ms / 1000.0

        # Alternate colors: even games base is white, odd games base is black
        if game_index % 2 == 0:
            return GameConfig(
                game_index=game_index,
                white_name="spsa-base",
                black_name="stockfish",
                white_path=base_path,
                black_path=ref_path,
                white_uci_options=None,
                black_uci_options=None,
                time_per_move=time_per_move,
                opening_fen=opening_fen,
                opening_name=opening_name,
                is_engine1_white=True  # base is white
            )
        else:
            return GameConfig(
                game_index=game_index,
                white_name="stockfish",
                black_name="spsa-base",
                white_path=ref_path,
                black_path=base_path,
                white_uci_options=None,
                black_uci_options=None,
                time_per_move=time_per_move,
                opening_fen=opening_fen,
                opening_name=opening_name,
                is_engine1_white=False  # base is black
            )

    def process_result(config: GameConfig, result):
        """Process a completed game result."""
        nonlocal base_wins, base_losses, draws

        # Determine result from base engine's perspective
        if config.is_engine1_white:
            # Base was white
            if result.result == "1-0":
                base_wins += 1
            elif result.result == "0-1":
                base_losses += 1
            else:
                draws += 1
        else:
            # Base was black
            if result.result == "0-1":
                base_wins += 1
            elif result.result == "1-0":
                base_losses += 1
            else:
                draws += 1

    if concurrency > 1:
        # Continuous pipeline: keep all workers busy by starting new games as old ones finish
        progress = ProgressDisplay(label="Ref")
        progress.start()

        def make_move_callback(game_idx: int):
            """Create a callback that updates progress for a specific game."""
            return lambda move_count: progress.update_moves(game_idx, move_count)

        with ThreadPoolExecutor(max_workers=concurrency) as executor:
            pending_futures = {}  # future -> config
            next_game_index = 0
            completed = 0

            # Submit initial batch up to concurrency limit
            while next_game_index < min(batch_size, concurrency):
                config = make_config(next_game_index)
                callback = make_move_callback(next_game_index)
                future = executor.submit(play_game_from_config, config, callback)
                pending_futures[future] = config
                progress.start_game(next_game_index)
                next_game_index += 1

            # Process completions and submit new games to keep workers busy
            while pending_futures:
                # Wait for at least one game to complete
                done, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    config = pending_futures.pop(future)
                    try:
                        result = future.result()
                        progress.finish_game(config.game_index, result.result)
                        process_result(config, result)
                        completed += 1
                    except Exception as e:
                        progress.finish_game(config.game_index, "err")
                        print(f"\n  Error in ref game: {e}")
                        errors += 1
                        completed += 1

                    # Submit a new game if there are more to play
                    if next_game_index < batch_size:
                        new_config = make_config(next_game_index)
                        callback = make_move_callback(next_game_index)
                        new_future = executor.submit(play_game_from_config, new_config, callback)
                        pending_futures[new_future] = new_config
                        progress.start_game(next_game_index)
                        next_game_index += 1

        progress.stop()

    else:
        # Sequential execution
        for i in range(batch_size):
            opening_fen, opening_name = random.choice(OPENING_BOOK)
            time_ms = random.uniform(timelow_ms, timehigh_ms)
            time_per_move = time_ms / 1000.0

            try:
                # Alternate colors
                if i % 2 == 0:
                    # Base is white
                    result, _ = play_game(
                        base_path, ref_path,
                        "spsa-base", "stockfish",
                        time_per_move, opening_fen, opening_name
                    )
                    if result == "1-0":
                        base_wins += 1
                    elif result == "0-1":
                        base_losses += 1
                    else:
                        draws += 1
                else:
                    # Base is black
                    result, _ = play_game(
                        ref_path, base_path,
                        "stockfish", "spsa-base",
                        time_per_move, opening_fen, opening_name
                    )
                    if result == "0-1":
                        base_wins += 1
                    elif result == "1-0":
                        base_losses += 1
                    else:
                        draws += 1
            except Exception as e:
                print(f"  Error in ref game: {e}")
                errors += 1

    return base_wins, base_losses, draws, errors


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

    # Get reference engine path (for tracking strength vs Stockfish)
    ref_engine_path = get_reference_engine_path(config)
    ref_enabled = ref_engine_path is not None

    print(f"\n{'='*60}")
    print("SPSA WORKER MODE")
    print(f"{'='*60}")
    print(f"Host: {hostname}")
    print(f"Concurrency: {concurrency} games in parallel")
    print(f"Batch size: {batch_size} games per update")
    print(f"Poll interval: {poll_interval}s when idle")
    print(f"Opening book: {len(OPENING_BOOK)} positions")
    print(f"Rusty-rival source: {get_rusty_rival_path(config)}")
    if ref_enabled:
        print(f"Reference engine: {ref_engine_path}")
    else:
        print(f"Reference games: DISABLED")
    print(f"{'='*60}")
    print("\nWaiting for work...")

    games_total = 0
    ref_games_total = 0
    current_iteration = None
    # Track this worker's contribution to current iteration
    worker_iteration_games = 0
    worker_iteration_ref = 0

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
                worker_iteration_games = 0
                worker_iteration_ref = 0
                print(f"\n\nIteration {current_iteration}: {iteration['games_played']}/{iteration['target_games']} games")
                print(f"  Time:  {iteration['timelow_ms']}-{iteration['timehigh_ms']}ms/move")

            # Ensure engines are built (cache check handles iteration matching)
            try:
                plus_path, minus_path, base_path = ensure_engines_built(iteration, config)
            except RuntimeError as e:
                print(f"\n  ERROR: {e}")
                print("  Waiting before retry...")
                time.sleep(poll_interval * 2)
                continue

            print(f"  Plus:  {plus_path}")
            print(f"  Minus: {minus_path}")
            if ref_enabled:
                print(f"  Base:  {base_path}")

            # Check if iteration is complete
            remaining = iteration['target_games'] - iteration['games_played']
            if remaining <= 0:
                print(f"\nIteration {current_iteration} complete!")
                current_iteration = None
                time.sleep(1)  # Brief pause before checking for next iteration
                continue

            # Create callback for database updates
            iteration_id = iteration['id']

            def on_spsa_batch(plus_wins: int, minus_wins: int, draws: int) -> bool:
                """Called every batch_size games. Updates DB and returns whether to continue."""
                nonlocal games_total, worker_iteration_games

                completed = plus_wins + minus_wins + draws
                if completed > 0:
                    update_iteration_results(iteration_id, completed, plus_wins, minus_wins, draws)
                    games_total += completed
                    worker_iteration_games += completed

                # Check if iteration is complete
                updated = get_pending_iteration()
                if not updated or updated['iteration_number'] != current_iteration:
                    return False  # Iteration changed or complete
                remaining = updated['target_games'] - updated['games_played']
                if remaining <= 0:
                    return False  # Iteration complete

                # Print progress
                print(f"  Batch: +{plus_wins}W-{minus_wins}L-{draws}D | Total: {updated['games_played']}/{updated['target_games']}")
                return True  # Keep going

            # Run SPSA games continuously until iteration is complete
            print(f"\n  SPSA games (concurrency={concurrency}, update every {batch_size}):")
            spsa_start = time.time()
            total_plus, total_minus, total_draws, total_errors, plus_nps, minus_nps = run_spsa_continuous(
                plus_path,
                minus_path,
                iteration['timelow_ms'],
                iteration['timehigh_ms'],
                concurrency,
                batch_size,
                on_spsa_batch
            )
            spsa_time = time.time() - spsa_start

            # Format NPS for display
            nps_str = ""
            if plus_nps > 0 or minus_nps > 0:
                avg_knps = (plus_nps + minus_nps) / 2000 if plus_nps > 0 and minus_nps > 0 else max(plus_nps, minus_nps) / 1000
                nps_str = f" NPS: {avg_knps:.0f}k"

            total_spsa = total_plus + total_minus + total_draws
            print(f"  SPSA done: {total_plus}W-{total_minus}L-{total_draws}D{nps_str} [{spsa_time:.1f}s]")

            # Play reference games (base vs Stockfish) - match SPSA games played
            if ref_enabled and total_spsa > 0:
                print(f"\n  Reference games ({total_spsa}):")
                ref_start = time.time()
                ref_wins, ref_losses, ref_draws, ref_errors = play_reference_batch(
                    base_path,
                    ref_engine_path,
                    iteration['timelow_ms'],
                    iteration['timehigh_ms'],
                    total_spsa,
                    concurrency
                )
                ref_time = time.time() - ref_start

                ref_completed = ref_wins + ref_losses + ref_draws
                if ref_completed > 0:
                    update_reference_results(iteration_id, ref_completed, ref_wins, ref_losses, ref_draws)
                    ref_games_total += ref_completed
                    worker_iteration_ref += ref_completed

                ref_result = f"{ref_wins}W-{ref_losses}L-{ref_draws}D"
                if ref_errors > 0:
                    ref_result += f" err={ref_errors}"
                print(f"  Ref done: {ref_result} [{ref_time:.1f}s]")

            print(f"  This worker: {worker_iteration_games} SPSA, {worker_iteration_ref} ref for iter {current_iteration}")

        except KeyboardInterrupt:
            print(f"\n\nWorker stopped. Total games played: {games_total}")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(poll_interval)
