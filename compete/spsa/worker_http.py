"""
SPSA HTTP worker implementation.

Polls the web API for pending SPSA iterations, builds engines from
parameters, and runs games between perturbed engine pairs.
Results are reported back via HTTP API.

This worker does NOT require direct database access - it communicates
entirely through the web API, making it suitable for remote Docker workers.

Environment variables (can be set in .env file):
    SPSA_API_URL: Base URL of the chess-compete web API (e.g., https://myapp.herokuapp.com)
    SPSA_API_KEY: API key for authentication
    COMPUTER_NAME: Optional hostname identifier for logging
"""

import os
import random
import socket
import sys
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from pathlib import Path

from dotenv import load_dotenv
import requests

# Load environment variables from .env file in chess-compete root
SPSA_DIR = Path(__file__).parent
CHESS_COMPETE_DIR = SPSA_DIR.parent.parent
load_dotenv(CHESS_COMPETE_DIR / '.env')

try:
    import tomllib
except ImportError:
    import tomli as tomllib

from compete.game import play_game, GameConfig, play_game_from_config
from compete.openings import OPENING_BOOK
from compete.spsa.build import build_spsa_engines, build_engine, get_rusty_rival_path
from compete.spsa.progress import ProgressDisplay


class APIClient:
    """HTTP client for SPSA worker API."""

    def __init__(self, base_url: str, api_key: str):
        self.base_url = base_url.rstrip('/')
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers['X-API-Key'] = api_key
        self.session.headers['Content-Type'] = 'application/json'

    def get_work(self, worker_host: str = None) -> dict | None:
        """
        Poll for pending work.

        Returns iteration data dict or None if no work available.
        """
        headers = {}
        if worker_host:
            headers['X-Worker-Host'] = worker_host

        try:
            resp = self.session.get(
                f'{self.base_url}/api/spsa/work',
                headers=headers,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            # Empty dict means no work
            if not data or 'id' not in data:
                return None
            return data
        except requests.RequestException as e:
            print(f"\n  API error (get_work): {e}")
            return None

    def report_spsa_results(self, iteration_id: int, games: int,
                           plus_wins: int, minus_wins: int, draws: int,
                           worker_name: str = None, avg_nps: int = None) -> int | None:
        """
        Report SPSA game results.

        Returns remaining game count from server, or None on error.
        """
        try:
            payload = {
                'games': games,
                'plus_wins': plus_wins,
                'minus_wins': minus_wins,
                'draws': draws
            }
            if worker_name:
                payload['worker_name'] = worker_name
            if avg_nps:
                payload['avg_nps'] = avg_nps

            resp = self.session.post(
                f'{self.base_url}/api/spsa/iterations/{iteration_id}/results',
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get('remaining')
        except requests.RequestException as e:
            print(f"\n  API error (report_spsa_results): {e}")
            return None

    def report_ref_results(self, iteration_id: int, games: int,
                          wins: int, losses: int, draws: int,
                          worker_name: str = None, avg_nps: int = None) -> int | None:
        """
        Report reference game results.

        Returns remaining game count from server, or None on error.
        """
        try:
            payload = {
                'games': games,
                'wins': wins,
                'losses': losses,
                'draws': draws
            }
            if worker_name:
                payload['worker_name'] = worker_name
            if avg_nps:
                payload['avg_nps'] = avg_nps

            resp = self.session.post(
                f'{self.base_url}/api/spsa/iterations/{iteration_id}/ref-results',
                json=payload,
                timeout=30
            )
            resp.raise_for_status()
            data = resp.json()
            return data.get('remaining')
        except requests.RequestException as e:
            print(f"\n  API error (report_ref_results): {e}")
            return None


def load_config() -> dict:
    """Load SPSA configuration."""
    config_file = Path(__file__).parent / 'config.toml'
    with open(config_file, 'rb') as f:
        return tomllib.load(f)


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


def ensure_spsa_engines_built(iteration: dict, config: dict, force_rebuild: bool = False) -> tuple[str, str]:
    """
    Ensure SPSA engine binaries (plus/minus) exist, building from API parameters if needed.

    Args:
        iteration: Iteration data from API (must be SPSA phase)
        config: SPSA configuration
        force_rebuild: If True, rebuild even if binaries exist

    Returns:
        (plus_path, minus_path) - paths to engine binaries
    """
    iteration_number = iteration['iteration_number']

    # Compute LOCAL paths
    output_base = Path(config['build']['engines_output_path'])
    if not output_base.is_absolute():
        chess_compete_dir = Path(__file__).parent.parent.parent
        output_base = chess_compete_dir / output_base

    binary_name = 'rusty-rival.exe' if os.name == 'nt' else 'rusty-rival'
    plus_dir = output_base / config['build']['plus_engine_name']
    minus_dir = output_base / config['build']['minus_engine_name']
    plus_path = plus_dir / binary_name
    minus_path = minus_dir / binary_name

    # Check if binaries exist and were built for the current iteration
    if not force_rebuild:
        plus_cached = get_cached_iteration(plus_dir)
        minus_cached = get_cached_iteration(minus_dir)

        all_binaries_exist = plus_path.exists() and minus_path.exists()
        all_iterations_match = (plus_cached == iteration_number and
                                minus_cached == iteration_number)

        if all_binaries_exist and all_iterations_match:
            print(f"  Using cached plus/minus engines for iteration {iteration_number}")
            return str(plus_path), str(minus_path)

    # Need to build engines from parameters
    print(f"\n  Building plus/minus engines for iteration {iteration_number}...")

    src_path = get_rusty_rival_path(config)

    # Build with parameters from API
    plus_params = iteration['plus_parameters']
    minus_params = iteration['minus_parameters']

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

        return str(new_plus_path), str(new_minus_path)
    except Exception as e:
        raise RuntimeError(f"Failed to build SPSA engines: {e}")


def ensure_base_engine_ready(iteration: dict, config: dict) -> str:
    """
    Get the base engine path for reference games, building if needed.

    Args:
        iteration: Iteration data from API (must be ref phase)
        config: SPSA configuration

    Returns:
        Path to the base engine binary
    """
    iteration_number = iteration['iteration_number']

    # Compute LOCAL path
    output_base = Path(config['build']['engines_output_path'])
    if not output_base.is_absolute():
        chess_compete_dir = Path(__file__).parent.parent.parent
        output_base = chess_compete_dir / output_base

    binary_name = 'rusty-rival.exe' if os.name == 'nt' else 'rusty-rival'
    base_dir = output_base / config['build']['base_engine_name']
    base_path = base_dir / binary_name

    # Check if base engine exists AND was built for this iteration
    cached_iteration = get_cached_iteration(base_dir)
    if base_path.exists() and cached_iteration == iteration_number:
        print(f"  Using cached base engine for iteration {iteration_number}")
        return str(base_path)

    # Need to build base engine with updated parameters
    print(f"\n  Building base engine for iteration {iteration_number}...")
    src_path = get_rusty_rival_path(config)
    base_params = iteration['base_parameters']

    if not build_engine(src_path, base_dir, base_params):
        raise RuntimeError("Failed to build base engine")

    # Write cache file
    write_cached_iteration(base_dir, iteration_number)
    print(f"  Built base engine: {base_path}")

    return str(base_path)


def run_spsa_games(
    plus_path: str, minus_path: str,
    timelow_ms: int, timehigh_ms: int,
    concurrency: int,
    on_game_complete: callable,
    max_games: int = 0
) -> tuple[dict, float, float]:
    """
    Run SPSA games (plus vs minus), calling on_game_complete after every game.

    Args:
        plus_path: Path to the plus-perturbed engine binary
        minus_path: Path to the minus-perturbed engine binary
        timelow_ms: Minimum time per move in milliseconds
        timehigh_ms: Maximum time per move in milliseconds
        concurrency: Number of parallel games to keep running
        on_game_complete: Callback(results) -> int
            Called after each game. Returns remaining games from server (int),
            or negative value to signal stop.
            results = {'plus_wins': N, 'minus_wins': N, 'draws': N}

    Returns:
        (totals, plus_avg_nps, minus_avg_nps)
    """
    # Totals
    total = {'plus_wins': 0, 'minus_wins': 0, 'draws': 0, 'errors': 0}

    # NPS tracking (overall)
    plus_nps_total = 0
    minus_nps_total = 0
    plus_nps_count = 0
    minus_nps_count = 0

    def make_config(game_index: int, game_num: int) -> GameConfig:
        """Create a SPSA game config (plus vs minus)."""
        opening_fen, opening_name = random.choice(OPENING_BOOK)
        time_ms = random.uniform(timelow_ms, timehigh_ms)
        time_per_move = time_ms / 1000.0

        # Alternate colors based on game number
        if game_num % 2 == 0:
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

    def process_result(config: GameConfig, result) -> dict:
        """Process a completed SPSA game result. Returns single-game result dict."""
        nonlocal plus_nps_total, minus_nps_total, plus_nps_count, minus_nps_count

        game_result = {'plus_wins': 0, 'minus_wins': 0, 'draws': 0}

        # Determine winner from plus engine's perspective
        if config.is_engine1_white:
            # Plus was white
            if result.result == "1-0":
                game_result['plus_wins'] = 1
                total['plus_wins'] += 1
            elif result.result == "0-1":
                game_result['minus_wins'] = 1
                total['minus_wins'] += 1
            else:
                game_result['draws'] = 1
                total['draws'] += 1
            # Collect NPS
            if result.white_nps:
                plus_nps_total += result.white_nps
                plus_nps_count += 1
                game_result['avg_nps'] = result.white_nps
            if result.black_nps:
                minus_nps_total += result.black_nps
                minus_nps_count += 1
        else:
            # Plus was black
            if result.result == "0-1":
                game_result['plus_wins'] = 1
                total['plus_wins'] += 1
            elif result.result == "1-0":
                game_result['minus_wins'] = 1
                total['minus_wins'] += 1
            else:
                game_result['draws'] = 1
                total['draws'] += 1
            # Collect NPS
            if result.white_nps:
                minus_nps_total += result.white_nps
                minus_nps_count += 1
            if result.black_nps:
                plus_nps_total += result.black_nps
                plus_nps_count += 1
                game_result['avg_nps'] = result.black_nps

        return game_result

    if concurrency > 1:
        # Continuous pipeline: always keep concurrency games running
        progress = ProgressDisplay(label="SPSA", concurrency=concurrency)
        progress.start()

        def make_move_callback(game_idx: int):
            """Create a callback that updates progress for a specific game."""
            return lambda move_count: progress.update_moves(game_idx, move_count)

        keep_adding = True
        phase_complete = False

        executor = ThreadPoolExecutor(max_workers=concurrency)
        try:
            pending_futures = {}  # future -> config
            next_game_index = 0

            # Submit initial games up to concurrency limit (capped by max_games)
            initial_limit = concurrency if max_games <= 0 else min(concurrency, max_games)
            while next_game_index < initial_limit:
                config = make_config(next_game_index, next_game_index)
                callback = make_move_callback(next_game_index)
                future = executor.submit(play_game_from_config, config, callback)
                pending_futures[future] = config
                progress.start_game(next_game_index, 'spsa')
                next_game_index += 1

            # Process completions and submit new games
            while pending_futures:
                done, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    config = pending_futures.pop(future)
                    try:
                        result = future.result()
                        # Get NPS for display
                        nps = None
                        if result.white_nps and result.black_nps:
                            nps = (result.white_nps + result.black_nps) // 2
                        elif result.white_nps:
                            nps = result.white_nps
                        elif result.black_nps:
                            nps = result.black_nps

                        # Compute interpreted outcome for progress display
                        if config.is_engine1_white:
                            outcome = "plus_win" if result.result == "1-0" else "minus_win" if result.result == "0-1" else "draw"
                        else:
                            outcome = "plus_win" if result.result == "0-1" else "minus_win" if result.result == "1-0" else "draw"

                        progress.finish_game(config.game_index, result.result, nps, outcome)
                        game_result = process_result(config, result)

                        # Report this single game to server, get remaining count
                        remaining = on_game_complete(game_result)

                        # Use remaining to decide whether to submit more games
                        if remaining is not None and remaining <= 0:
                            keep_adding = False
                            # Cancel pending futures that haven't started yet
                            cancelled = 0
                            for f in list(pending_futures.keys()):
                                if f.cancel():
                                    pending_futures.pop(f)
                                    cancelled += 1
                            in_flight = len(pending_futures)
                            if cancelled or in_flight:
                                print(f"  Phase complete — cancelled {cancelled} queued game(s), abandoning {in_flight} in-flight")
                            pending_futures.clear()
                            phase_complete = True
                            break
                        elif remaining is not None and remaining <= len(pending_futures):
                            pass  # Enough in-flight, don't add more
                        elif keep_adding and (max_games <= 0 or next_game_index < max_games):
                            new_config = make_config(next_game_index, next_game_index)
                            callback = make_move_callback(next_game_index)
                            new_future = executor.submit(play_game_from_config, new_config, callback)
                            pending_futures[new_future] = new_config
                            progress.start_game(next_game_index, 'spsa')
                            next_game_index += 1

                    except Exception as e:
                        progress.finish_game(config.game_index, "err", outcome="err")
                        print(f"\n  Error in SPSA game: {e}")
                        total['errors'] += 1

        finally:
            executor.shutdown(wait=not phase_complete)

        progress.stop()

    else:
        # Sequential execution
        game_index = 0
        keep_going = True

        while keep_going:
            try:
                config = make_config(game_index, game_index)
                result, _ = play_game(
                    config.white_path, config.black_path,
                    config.white_name, config.black_name,
                    config.time_per_move, config.opening_fen, config.opening_name
                )
                # Create a minimal result object for processing
                class MinResult:
                    pass
                min_result = MinResult()
                min_result.result = result
                min_result.white_nps = None
                min_result.black_nps = None
                game_result = process_result(config, min_result)

                # Report this single game to server
                remaining = on_game_complete(game_result)
                if remaining is not None and remaining <= 0:
                    keep_going = False

            except Exception as e:
                print(f"  Error in SPSA game: {e}")
                total['errors'] += 1

            game_index += 1

    # Calculate average NPS
    plus_avg_nps = plus_nps_total / plus_nps_count if plus_nps_count > 0 else 0
    minus_avg_nps = minus_nps_total / minus_nps_count if minus_nps_count > 0 else 0

    return total, plus_avg_nps, minus_avg_nps


def run_ref_games(
    base_path: str, ref_path: str, ref_elo: int,
    timelow_ms: int, timehigh_ms: int,
    concurrency: int,
    on_game_complete: callable,
    max_games: int = 0
) -> dict:
    """
    Run reference games (base vs Stockfish), calling on_game_complete after every game.

    Args:
        base_path: Path to the base engine binary (with updated parameters)
        ref_path: Path to the reference engine (Stockfish)
        ref_elo: ELO limit for reference engine
        timelow_ms: Minimum time per move in milliseconds
        timehigh_ms: Maximum time per move in milliseconds
        concurrency: Number of parallel games to keep running
        on_game_complete: Callback(results) -> int
            Called after each game. Returns remaining games from server (int),
            or negative value to signal stop.
            results = {'wins': N, 'losses': N, 'draws': N}

    Returns:
        totals dict with win/loss/draw counts
    """
    # UCI options to limit Stockfish strength
    stockfish_options = {
        "UCI_LimitStrength": True,
        "UCI_Elo": ref_elo
    }

    # Totals
    total = {'wins': 0, 'losses': 0, 'draws': 0, 'errors': 0}

    def make_config(game_index: int, game_num: int) -> GameConfig:
        """Create a reference game config (base vs stockfish)."""
        opening_fen, opening_name = random.choice(OPENING_BOOK)
        time_ms = random.uniform(timelow_ms, timehigh_ms)
        time_per_move = time_ms / 1000.0

        # Alternate colors based on game number
        if game_num % 2 == 0:
            return GameConfig(
                game_index=game_index,
                white_name="spsa-base",
                black_name=f"sf-{ref_elo}",
                white_path=base_path,
                black_path=ref_path,
                white_uci_options=None,
                black_uci_options=stockfish_options,
                time_per_move=time_per_move,
                opening_fen=opening_fen,
                opening_name=opening_name,
                is_engine1_white=True  # base is white
            )
        else:
            return GameConfig(
                game_index=game_index,
                white_name=f"sf-{ref_elo}",
                black_name="spsa-base",
                white_path=ref_path,
                black_path=base_path,
                white_uci_options=stockfish_options,
                black_uci_options=None,
                time_per_move=time_per_move,
                opening_fen=opening_fen,
                opening_name=opening_name,
                is_engine1_white=False  # base is black
            )

    def process_result(config: GameConfig, result) -> dict:
        """Process a completed reference game result. Returns single-game result dict."""
        game_result = {'wins': 0, 'losses': 0, 'draws': 0}

        # Determine result from base engine's perspective
        if config.is_engine1_white:
            # Base was white
            if result.result == "1-0":
                game_result['wins'] = 1
                total['wins'] += 1
            elif result.result == "0-1":
                game_result['losses'] = 1
                total['losses'] += 1
            else:
                game_result['draws'] = 1
                total['draws'] += 1
            # Track base engine NPS (white)
            if result.white_nps:
                game_result['avg_nps'] = result.white_nps
        else:
            # Base was black
            if result.result == "0-1":
                game_result['wins'] = 1
                total['wins'] += 1
            elif result.result == "1-0":
                game_result['losses'] = 1
                total['losses'] += 1
            else:
                game_result['draws'] = 1
                total['draws'] += 1
            # Track base engine NPS (black)
            if result.black_nps:
                game_result['avg_nps'] = result.black_nps

        return game_result

    if concurrency > 1:
        # Continuous pipeline: always keep concurrency games running
        progress = ProgressDisplay(label="Ref", concurrency=concurrency)
        progress.start()

        def make_move_callback(game_idx: int):
            """Create a callback that updates progress for a specific game."""
            return lambda move_count: progress.update_moves(game_idx, move_count)

        keep_adding = True
        phase_complete = False

        executor = ThreadPoolExecutor(max_workers=concurrency)
        try:
            pending_futures = {}  # future -> config
            next_game_index = 0

            # Submit initial games up to concurrency limit (capped by max_games)
            initial_limit = concurrency if max_games <= 0 else min(concurrency, max_games)
            while next_game_index < initial_limit:
                config = make_config(next_game_index, next_game_index)
                callback = make_move_callback(next_game_index)
                future = executor.submit(play_game_from_config, config, callback)
                pending_futures[future] = config
                progress.start_game(next_game_index, 'ref')
                next_game_index += 1

            # Process completions and submit new games
            while pending_futures:
                done, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)

                for future in done:
                    config = pending_futures.pop(future)
                    try:
                        result = future.result()
                        # Get NPS for display
                        nps = None
                        if result.white_nps and result.black_nps:
                            nps = (result.white_nps + result.black_nps) // 2
                        elif result.white_nps:
                            nps = result.white_nps
                        elif result.black_nps:
                            nps = result.black_nps

                        # Compute interpreted outcome for progress display
                        if config.is_engine1_white:
                            outcome = "win" if result.result == "1-0" else "loss" if result.result == "0-1" else "draw"
                        else:
                            outcome = "win" if result.result == "0-1" else "loss" if result.result == "1-0" else "draw"

                        progress.finish_game(config.game_index, result.result, nps, outcome)
                        game_result = process_result(config, result)

                        # Report this single game to server, get remaining count
                        remaining = on_game_complete(game_result)

                        # Use remaining to decide whether to submit more games
                        if remaining is not None and remaining <= 0:
                            keep_adding = False
                            # Cancel pending futures that haven't started yet
                            cancelled = 0
                            for f in list(pending_futures.keys()):
                                if f.cancel():
                                    pending_futures.pop(f)
                                    cancelled += 1
                            in_flight = len(pending_futures)
                            if cancelled or in_flight:
                                print(f"  Phase complete — cancelled {cancelled} queued game(s), abandoning {in_flight} in-flight")
                            pending_futures.clear()
                            phase_complete = True
                            break
                        elif remaining is not None and remaining <= len(pending_futures):
                            pass  # Enough in-flight, don't add more
                        elif keep_adding and (max_games <= 0 or next_game_index < max_games):
                            new_config = make_config(next_game_index, next_game_index)
                            callback = make_move_callback(next_game_index)
                            new_future = executor.submit(play_game_from_config, new_config, callback)
                            pending_futures[new_future] = new_config
                            progress.start_game(next_game_index, 'ref')
                            next_game_index += 1

                    except Exception as e:
                        progress.finish_game(config.game_index, "err", outcome="err")
                        print(f"\n  Error in ref game: {e}")
                        total['errors'] += 1

        finally:
            executor.shutdown(wait=not phase_complete)

        progress.stop()

    else:
        # Sequential execution
        game_index = 0
        keep_going = True

        while keep_going:
            try:
                config = make_config(game_index, game_index)
                result, _ = play_game(
                    config.white_path, config.black_path,
                    config.white_name, config.black_name,
                    config.time_per_move, config.opening_fen, config.opening_name,
                    config.white_uci_options, config.black_uci_options
                )
                class MinResult:
                    pass
                min_result = MinResult()
                min_result.result = result
                min_result.white_nps = None
                min_result.black_nps = None
                game_result = process_result(config, min_result)

                # Report this single game to server
                remaining = on_game_complete(game_result)
                if remaining is not None and remaining <= 0:
                    keep_going = False

            except Exception as e:
                print(f"  Error in ref game: {e}")
                total['errors'] += 1

            game_index += 1

    return total


def get_reference_engine_path(config: dict) -> str | None:
    """Get the path to the reference engine (Stockfish), or None if not configured."""
    ref_path = config.get('reference', {}).get('engine_path')
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


def run_http_worker(api_url: str, api_key: str, concurrency: int = 1,
                    poll_interval: int = 10, idle_timeout: int = 0):
    """
    Run the SPSA HTTP worker loop.

    Args:
        api_url: Base URL of the chess-compete web API
        api_key: API key for authentication
        concurrency: Number of games to run in parallel
        poll_interval: Seconds to wait when no work is available
        idle_timeout: Shutdown after this many minutes of no work (0 = disabled)
    """
    hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())

    # Create API client
    api = APIClient(api_url, api_key)

    # Load config for build settings
    config = load_config()

    # Get reference engine path and ELO
    ref_engine_path = get_reference_engine_path(config)
    ref_elo = config.get('reference', {}).get('engine_elo', 2600)
    ref_enabled = ref_engine_path is not None

    print(f"\n{'='*60}")
    print("SPSA HTTP WORKER MODE")
    print(f"{'='*60}")
    print(f"Host: {hostname}")
    print(f"API URL: {api_url}")
    print(f"Concurrency: {concurrency} games in parallel")
    print(f"Poll interval: {poll_interval}s when idle")
    print(f"Opening book: {len(OPENING_BOOK)} positions")
    print(f"Rusty-rival source: {get_rusty_rival_path(config)}")
    if ref_enabled:
        print(f"Reference engine: {ref_engine_path} (ELO {ref_elo})")
    else:
        print(f"Reference games: DISABLED")
    print(f"{'='*60}")
    print("\nWaiting for work...")

    games_total = 0
    ref_games_total = 0
    current_iteration = None
    current_phase = None
    last_work_time = time.time()

    while True:
        try:
            # Get current iteration via API
            iteration = api.get_work(hostname)

            if not iteration:
                # Check idle timeout
                if idle_timeout > 0:
                    idle_minutes = (time.time() - last_work_time) / 60
                    if idle_minutes >= idle_timeout:
                        print(f"\n\nNo work for {idle_timeout} minutes — shutting down.")
                        import subprocess
                        subprocess.run(['sudo', 'shutdown', '-h', 'now'], check=False)
                        sys.exit(0)
                print(".", end="", flush=True)
                time.sleep(poll_interval)
                continue

            last_work_time = time.time()
            phase = iteration['phase']
            iteration_num = iteration['iteration_number']
            iteration_id = iteration['id']

            # Check if we switched to a new iteration or phase
            if current_iteration != iteration_num or current_phase != phase:
                current_iteration = iteration_num
                current_phase = phase
                phase_name = "SPSA (plus vs minus)" if phase == 'spsa' else "Reference (base vs Stockfish)"
                print(f"\n\nIteration {iteration_num} - Phase: {phase_name}")
                print(f"  Progress: {iteration['games_played']}/{iteration['target_games']} games")
                print(f"  Time: {iteration['timelow_ms']}-{iteration['timehigh_ms']}ms/move")

            # Check if phase is complete
            remaining = iteration['target_games'] - iteration['games_played']
            if remaining <= 0:
                print(f"\nPhase complete!")
                current_iteration = None
                current_phase = None
                time.sleep(1)  # Brief pause before checking for next work
                continue

            if phase == 'spsa':
                # ===== SPSA PHASE =====
                # Build plus/minus engines
                try:
                    plus_path, minus_path = ensure_spsa_engines_built(iteration, config)
                except RuntimeError as e:
                    print(f"\n  ERROR: {e}")
                    print("  Waiting before retry...")
                    time.sleep(poll_interval * 2)
                    continue

                print(f"  Plus:  {plus_path}")
                print(f"  Minus: {minus_path}")

                # Callback for per-game SPSA updates
                def on_spsa_game(results: dict) -> int:
                    nonlocal games_total
                    game_nps = results.get('avg_nps')
                    remaining = api.report_spsa_results(
                        iteration_id, 1,
                        results['plus_wins'], results['minus_wins'], results['draws'],
                        worker_name=hostname, avg_nps=game_nps
                    )
                    if remaining is None:
                        print("  Warning: Failed to report results to API")
                        return -1  # Signal stop on error
                    games_total += 1
                    return remaining

                # Run SPSA games
                print(f"\n  Running SPSA games (concurrency={concurrency}):")
                start_time = time.time()
                total, plus_nps, minus_nps = run_spsa_games(
                    plus_path, minus_path,
                    iteration['timelow_ms'], iteration['timehigh_ms'],
                    concurrency, on_spsa_game,
                    max_games=remaining
                )
                elapsed = time.time() - start_time

                # Format NPS
                nps_str = ""
                if plus_nps > 0 or minus_nps > 0:
                    avg_knps = (plus_nps + minus_nps) / 2000 if plus_nps > 0 and minus_nps > 0 else max(plus_nps, minus_nps) / 1000
                    nps_str = f" NPS: {avg_knps:.0f}k"

                print(f"  Done: {total['plus_wins']}W-{total['minus_wins']}L-{total['draws']}D{nps_str} [{elapsed:.1f}s]")

            elif phase == 'ref':
                # ===== REFERENCE PHASE =====
                if not ref_enabled:
                    print("  WARNING: Ref phase but reference games disabled!")
                    time.sleep(poll_interval)
                    continue

                # Ensure base engine exists
                try:
                    base_path = ensure_base_engine_ready(iteration, config)
                except RuntimeError as e:
                    print(f"\n  ERROR: {e}")
                    print("  Waiting before retry...")
                    time.sleep(poll_interval * 2)
                    continue

                print(f"  Base: {base_path}")
                print(f"  Ref:  {ref_engine_path}")

                # Callback for per-game ref updates
                def on_ref_game(results: dict) -> int:
                    nonlocal ref_games_total
                    game_nps = results.get('avg_nps')
                    remaining = api.report_ref_results(
                        iteration_id, 1,
                        results['wins'], results['losses'], results['draws'],
                        worker_name=hostname, avg_nps=game_nps
                    )
                    if remaining is None:
                        print("  Warning: Failed to report results to API")
                        return -1  # Signal stop on error
                    ref_games_total += 1
                    return remaining

                # Run reference games
                print(f"\n  Running reference games vs sf-{ref_elo} (concurrency={concurrency}):")
                start_time = time.time()
                total = run_ref_games(
                    base_path, ref_engine_path, ref_elo,
                    iteration['timelow_ms'], iteration['timehigh_ms'],
                    concurrency, on_ref_game,
                    max_games=remaining
                )
                elapsed = time.time() - start_time

                print(f"  Done: {total['wins']}W-{total['losses']}L-{total['draws']}D [{elapsed:.1f}s]")

        except KeyboardInterrupt:
            print(f"\n\nWorker stopped. Total: {games_total} SPSA, {ref_games_total} ref games")
            break
        except Exception as e:
            print(f"\nError: {e}")
            import traceback
            traceback.print_exc()
            time.sleep(poll_interval)


def main():
    """Entry point for HTTP worker."""
    import argparse

    parser = argparse.ArgumentParser(description='SPSA HTTP Worker')
    parser.add_argument('--api-url', type=str,
                        default=os.environ.get('SPSA_API_URL'),
                        help='Base URL of chess-compete API (or set SPSA_API_URL env var)')
    parser.add_argument('--api-key', type=str,
                        default=os.environ.get('SPSA_API_KEY'),
                        help='API key for authentication (or set SPSA_API_KEY env var)')
    parser.add_argument('--concurrency', '-c', type=int, default=1,
                        help='Number of games to run in parallel (default: 1)')
    parser.add_argument('--poll-interval', '-p', type=int, default=10,
                        help='Seconds to wait when no work available (default: 10)')
    parser.add_argument('--idle-timeout', type=int, default=0,
                        help='Shutdown after N minutes of no work (0 = disabled, default: 0)')

    args = parser.parse_args()

    if not args.api_url:
        print("Error: --api-url or SPSA_API_URL environment variable required")
        sys.exit(1)

    if not args.api_key:
        print("Error: --api-key or SPSA_API_KEY environment variable required")
        sys.exit(1)

    run_http_worker(
        api_url=args.api_url,
        api_key=args.api_key,
        concurrency=args.concurrency,
        poll_interval=args.poll_interval,
        idle_timeout=args.idle_timeout
    )


if __name__ == '__main__':
    main()
