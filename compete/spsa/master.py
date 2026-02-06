"""
SPSA Master Controller

Orchestrates the SPSA parameter tuning process with two-phase iteration:

Phase 1 (SPSA Games):
1. Reads current parameters from params.toml
2. Generates perturbed parameter sets (plus/minus)
3. Creates iteration record in database (workers build plus/minus engines)
4. Waits for workers to complete SPSA games (plus vs minus)
5. Calculates gradient and updates parameters
6. Saves updated params to params.toml

Phase 2 (Reference Games):
7. Builds NEW base engine with updated parameters
8. Sets iteration status to 'ref_pending'
9. Waits for workers to complete reference games (new base vs Stockfish)
10. Calculates Elo estimate and marks iteration complete

This two-phase approach ensures reference games measure the actual strength
of the parameter update, not the pre-update baseline.

Usage:
    cd chess-compete
    python -m compete.spsa.master
"""

import math
import os
import random
import sys
import time
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv

# Load environment from chess-compete root
SPSA_DIR = Path(__file__).parent
CHESS_COMPETE_DIR = SPSA_DIR.parent.parent
load_dotenv(CHESS_COMPETE_DIR / '.env')

try:
    import tomllib
except ImportError:
    import tomli as tomllib


def with_db_retry(func, max_retries=5, retry_delay=10):
    """Execute a database operation with retry logic."""
    last_error = None
    for attempt in range(max_retries):
        try:
            return func()
        except Exception as e:
            last_error = e
            if attempt < max_retries - 1:
                print(f"\n  DB error (attempt {attempt + 1}/{max_retries}): {e}")
                print(f"  Retrying in {retry_delay}s...")
                time.sleep(retry_delay)
            else:
                print(f"\n  DB error (attempt {attempt + 1}/{max_retries}): {e}")
    raise RuntimeError(f"Database operation failed after {max_retries} attempts: {last_error}")


def get_active_run() -> tuple[int, str]:
    """
    Get the active SPSA run from the database.

    Returns (run_id, run_name). Raises RuntimeError if no active run exists.
    """
    from web.app import create_app
    from web.models import SpsaRun

    def _get():
        app = create_app()
        with app.app_context():
            run = SpsaRun.query.filter_by(is_active=True).first()
            if not run:
                raise RuntimeError(
                    "No active SPSA run found. Create one in the database first:\n"
                    "  INSERT INTO spsa_runs (name, description, is_active) "
                    "VALUES ('Run N', 'Description', TRUE);"
                )
            return run.id, run.name

    return with_db_retry(_get)


def load_params() -> dict:
    """
    Load parameters from params.toml.
    Returns dict of {param_name: {'value': X, 'min': Y, 'max': Z, 'step': S}}
    """
    params_file = SPSA_DIR / 'params.toml'
    with open(params_file, 'rb') as f:
        return tomllib.load(f)


def save_params(params: dict):
    """
    Save parameters to params.toml.
    Preserves comments by reading original and only updating values.
    """
    params_file = SPSA_DIR / 'params.toml'

    # Read original file
    content = params_file.read_text()

    # Update each parameter's value
    import re
    for name, cfg in params.items():
        # Find and replace the value line for this parameter
        # Pattern: value = <number>
        pattern = rf'(\[{re.escape(name)}\][^\[]*value\s*=\s*)-?[\d.]+'
        replacement = rf'\g<1>{cfg["value"]}'
        content = re.sub(pattern, replacement, content, flags=re.DOTALL)

    params_file.write_text(content)


def load_config() -> dict:
    """Load configuration from config.toml."""
    config_file = SPSA_DIR / 'config.toml'
    with open(config_file, 'rb') as f:
        return tomllib.load(f)


def generate_perturbations(params: dict, c_k: float, iteration: int) -> tuple[dict, dict, dict]:
    """
    Generate perturbed parameter sets for SPSA.

    Args:
        params: Current parameter values with min/max/step/active_from_iteration
        c_k: Perturbation coefficient for this iteration
        iteration: Current iteration number (for phased activation)

    Returns:
        (plus_params, minus_params, signs)
        - plus_params: dict of {param_name: value} for θ + c_k * Δ * step (active) or θ (inactive)
        - minus_params: dict of {param_name: value} for θ - c_k * Δ * step (active) or θ (inactive)
        - signs: dict of {param_name: +1 or -1} for ACTIVE params only
    """
    plus_params = {}
    minus_params = {}
    signs = {}

    for name, cfg in params.items():
        value = cfg['value']
        min_val = cfg['min']
        max_val = cfg['max']

        # Check if this param is active at the current iteration
        active_from = cfg.get('active_from_iteration', 1)
        if iteration < active_from:
            # Inactive param: use current value without perturbation
            plus_params[name] = value
            minus_params[name] = value
            # Don't add to signs - will be skipped in gradient/update
            continue

        # Active param: apply perturbation
        step = cfg['step']

        # Random sign: +1 or -1 (Bernoulli ±1)
        sign = random.choice([-1, 1])
        signs[name] = sign

        # Perturbation amount
        delta = c_k * step * sign

        # Apply with bounds
        plus_val = max(min_val, min(max_val, value + delta))
        minus_val = max(min_val, min(max_val, value - delta))

        plus_params[name] = plus_val
        minus_params[name] = minus_val

    # Enforce inter-parameter constraints (#5)
    # Some parameter pairs have logical relationships that must hold for
    # the engine to function correctly (e.g., depth_reduction < min_depth)
    _enforce_parameter_constraints(plus_params)
    _enforce_parameter_constraints(minus_params)

    return plus_params, minus_params, signs


# Parameter pairs where the first must be less than the second.
# Violating these can produce negative search depths, breaking the engine.
PARAMETER_CONSTRAINTS = [
    ('multicut_depth_reduction', 'multicut_min_depth'),
    ('singular_extension_depth_reduction', 'singular_extension_min_depth'),
]


def _enforce_parameter_constraints(params: dict) -> None:
    """Clamp dependent parameters so logical constraints hold."""
    for less_param, greater_param in PARAMETER_CONSTRAINTS:
        if less_param in params and greater_param in params:
            if params[less_param] >= params[greater_param]:
                clamped = params[greater_param] - 1
                print(f"  Constraint: {less_param} ({params[less_param]:.2f}) "
                      f">= {greater_param} ({params[greater_param]:.2f}), "
                      f"clamping to {clamped:.2f}")
                params[less_param] = clamped


def create_iteration(run_id: int, iteration_number: int, effective_iteration: int, plus_path: str, minus_path: str,
                     ref_path: str | None, ref_target_games: int,
                     config: dict, base_params: dict, plus_params: dict,
                     minus_params: dict, signs: dict) -> int:
    """
    Create a new SPSA iteration record in the database.

    The base_engine_path is NOT set initially - it will be set after
    SPSA games complete and the gradient is calculated. This ensures
    reference games measure the actual updated parameters.

    Returns the iteration ID.
    """
    from web.app import create_app
    from web.database import db
    from web.models import SpsaIteration

    def _create():
        app = create_app()
        with app.app_context():
            # Check for duplicate iteration number within this run
            existing = SpsaIteration.query.filter(
                SpsaIteration.run_id == run_id,
                SpsaIteration.iteration_number == iteration_number
            ).first()
            if existing:
                raise RuntimeError(
                    f"Iteration {iteration_number} already exists in run {run_id} "
                    f"(id={existing.id}, status={existing.status}). "
                    "Another master may be running."
                )
            iteration = SpsaIteration(
                run_id=run_id,
                iteration_number=iteration_number,
                effective_iteration=effective_iteration,
                plus_engine_path=plus_path,
                minus_engine_path=minus_path,
                base_engine_path=None,  # Set after SPSA phase
                ref_engine_path=ref_path,
                timelow_ms=int(config['time_control']['timelow'] * 1000),
                timehigh_ms=int(config['time_control']['timehigh'] * 1000),
                target_games=config['games']['games_per_iteration'],
                ref_target_games=ref_target_games,
                status='pending',
                base_parameters={k: v['value'] for k, v in base_params.items()},
                plus_parameters=plus_params,
                minus_parameters=minus_params,
                perturbation_signs=signs,
            )
            db.session.add(iteration)
            db.session.commit()
            return iteration.id

    return with_db_retry(_create)


def wait_for_spsa_completion(iteration_id: int, poll_interval: int = 30) -> dict:
    """
    Wait for workers to complete SPSA games (Phase 1).

    Returns the completed iteration data once target_games is reached.
    """
    from web.app import create_app
    from web.database import db
    from web.models import SpsaIteration

    db_errors = 0
    max_db_errors = 10

    while True:
        try:
            app = create_app()
            with app.app_context():
                # Expire all cached objects to force fresh read from database
                db.session.expire_all()
                iteration = db.session.get(SpsaIteration, iteration_id)
                if not iteration:
                    raise RuntimeError(f"Iteration {iteration_id} not found!")

                progress = f"{iteration.games_played}/{iteration.target_games}"
                plus_score = iteration.plus_wins + iteration.draws * 0.5
                total = iteration.games_played or 1
                pct = plus_score / total * 100

                print(f"\r  SPSA Progress: {progress} games | Plus: {iteration.plus_wins}W-{iteration.minus_wins}L-{iteration.draws}D ({pct:.1f}%)", end='', flush=True)

                if iteration.games_played >= iteration.target_games:
                    print()  # Newline after progress
                    return {
                        'id': iteration.id,
                        'games_played': iteration.games_played,
                        'plus_wins': iteration.plus_wins,
                        'minus_wins': iteration.minus_wins,
                        'draws': iteration.draws,
                        'base_parameters': iteration.base_parameters,
                        'perturbation_signs': iteration.perturbation_signs,
                    }

                # Reset error counter on success
                db_errors = 0

        except Exception as e:
            db_errors += 1
            print(f"\n  DB error ({db_errors}/{max_db_errors}): {e}")
            if db_errors >= max_db_errors:
                raise RuntimeError(f"Too many database errors, giving up")
            # Wait a bit longer before retry on error
            time.sleep(poll_interval * 2)
            continue

        time.sleep(poll_interval)


def set_ref_phase(iteration_id: int, base_engine_path: str, updated_params: dict):
    """
    Transition iteration to reference game phase.

    Sets the base_engine_path (newly built with updated params) and
    changes status to 'ref_pending' so workers start playing ref games.

    Args:
        iteration_id: The iteration to update
        base_engine_path: Path to the newly built base engine
        updated_params: The updated parameter values (for record keeping)
    """
    from web.app import create_app
    from web.database import db
    from web.models import SpsaIteration

    def _set():
        app = create_app()
        with app.app_context():
            iteration = db.session.get(SpsaIteration, iteration_id)
            if not iteration:
                raise RuntimeError(f"Iteration {iteration_id} not found!")
            iteration.base_engine_path = base_engine_path
            iteration.status = 'ref_pending'
            # Store the updated params (these are the params used for base engine)
            iteration.base_parameters = updated_params
            db.session.commit()

    with_db_retry(_set)


def wait_for_ref_completion(iteration_id: int, poll_interval: int = 30) -> dict:
    """
    Wait for workers to complete reference games (Phase 2).

    Returns the completed iteration data once ref_target_games is reached.
    """
    from web.app import create_app
    from web.database import db
    from web.models import SpsaIteration

    db_errors = 0
    max_db_errors = 10

    while True:
        try:
            app = create_app()
            with app.app_context():
                # Expire all cached objects to force fresh read from database
                db.session.expire_all()
                iteration = db.session.get(SpsaIteration, iteration_id)
                if not iteration:
                    raise RuntimeError(f"Iteration {iteration_id} not found!")

                ref_target = iteration.ref_target_games or 0
                progress = f"{iteration.ref_games_played}/{ref_target}"
                ref_score = iteration.ref_wins + iteration.ref_draws * 0.5
                ref_total = iteration.ref_games_played or 1
                ref_pct = ref_score / ref_total * 100

                print(f"\r  Ref Progress: {progress} games | Base: {iteration.ref_wins}W-{iteration.ref_losses}L-{iteration.ref_draws}D ({ref_pct:.1f}%)", end='', flush=True)

                if iteration.ref_games_played >= ref_target:
                    print()  # Newline after progress
                    return {
                        'id': iteration.id,
                        'ref_games_played': iteration.ref_games_played,
                        'ref_wins': iteration.ref_wins,
                        'ref_losses': iteration.ref_losses,
                        'ref_draws': iteration.ref_draws,
                    }

                # Reset error counter on success
                db_errors = 0

        except Exception as e:
            db_errors += 1
            print(f"\n  DB error ({db_errors}/{max_db_errors}): {e}")
            if db_errors >= max_db_errors:
                raise RuntimeError(f"Too many database errors, giving up")
            # Wait a bit longer before retry on error
            time.sleep(poll_interval * 2)
            continue

        time.sleep(poll_interval)


def calculate_gradient(results: dict, params: dict, c_k: float,
                       max_elo_diff: float = 800.0,
                       max_gradient_factor: float = 0.0) -> tuple[dict, float]:
    """
    Calculate gradient estimate from game results.

    Modified SPSA gradient formula that normalizes by parameter range:
        gradient = elo_diff * sign * step / (2 * c_k)

    This ensures that:
    - Larger step = larger update (intuitive)
    - Parameters update at similar % of their range regardless of scale
    - step represents perturbation size AND controls update magnitude proportionally

    The standard SPSA formula (dividing by step) causes:
    - Small step → huge updates (parameters with small ranges go wild)
    - Large step → tiny updates (parameters with large ranges never move)

    Args:
        max_elo_diff: Cap on |elo_diff| used for gradient calculation.
            Outlier SPSA results (e.g., 427-80) often indicate a broken engine
            rather than meaningful parameter signal. Default 800 (no practical cap).
        max_gradient_factor: If > 0, clip each gradient to ±(factor * step).
            Prevents a single iteration from pushing parameters to bounds.

    Returns:
        (gradient, elo_diff)
    """
    total_games = results['games_played']
    plus_score = results['plus_wins'] + results['draws'] * 0.5

    # Score from plus engine's perspective (0 to 1)
    score = plus_score / total_games

    # Convert to Elo difference
    if score <= 0.001:
        elo_diff = -800.0
    elif score >= 0.999:
        elo_diff = 800.0
    else:
        elo_diff = -400 * math.log10(1 / score - 1)

    # Cap ELO diff to limit influence of outlier results (#6)
    raw_elo_diff = elo_diff
    elo_diff = max(-max_elo_diff, min(max_elo_diff, elo_diff))
    if raw_elo_diff != elo_diff:
        print(f"  ELO diff capped: {raw_elo_diff:.1f} -> {elo_diff:.1f}")

    # Calculate gradient for each parameter
    signs = results['perturbation_signs']
    gradient = {}

    for name, cfg in params.items():
        if name in signs:
            sign = signs[name]
            step = params[name]['step']
            # Gradient proportional to step (not inversely proportional)
            # This makes updates ~proportional to step/range for all parameters
            grad = elo_diff * sign * step / (2 * c_k)

            # Clip gradient to prevent single-iteration blowout (#4)
            if max_gradient_factor > 0:
                max_grad = max_gradient_factor * step
                grad = max(-max_grad, min(max_grad, grad))

            gradient[name] = grad

    return gradient, elo_diff


def update_parameters(params: dict, gradient: dict, a_k: float) -> dict:
    """
    Update parameters using gradient estimate.

    θ_new = θ_old + a_k * gradient

    Respects min/max bounds for each parameter.
    """
    for name, cfg in params.items():
        if name in gradient:
            old_value = cfg['value']
            new_value = old_value + a_k * gradient[name]

            # Clamp to bounds
            new_value = max(cfg['min'], min(cfg['max'], new_value))

            cfg['value'] = new_value

    return params


def calculate_elo_from_score(wins: int, losses: int, draws: int, opponent_elo: float) -> float | None:
    """
    Calculate Elo rating from game results against a known-strength opponent.

    Uses standard Elo expected score formula:
        E = 1 / (1 + 10^((R_opponent - R_player) / 400))

    Solving for R_player given score percentage:
        R_player = R_opponent - 400 * log10(1/S - 1)

    Where S is the score percentage (0 to 1).

    Returns None if no games played or score is 0% or 100% (infinite Elo).
    """
    import math

    total = wins + losses + draws
    if total == 0:
        return None

    score = (wins + draws * 0.5) / total

    # Avoid log(0) for extreme scores - clamp to reasonable range
    if score <= 0.001:
        score = 0.001  # Very weak
    elif score >= 0.999:
        score = 0.999  # Very strong

    # R_player = R_opponent - 400 * log10(1/S - 1)
    elo = opponent_elo - 400 * math.log10(1 / score - 1)

    return elo


def mark_iteration_complete(iteration_id: int, gradient: dict, elo_diff: float, ref_elo: float | None):
    """Mark iteration as complete and save results."""
    from web.app import create_app
    from web.database import db
    from web.models import SpsaIteration

    def _mark():
        app = create_app()
        with app.app_context():
            iteration = db.session.get(SpsaIteration, iteration_id)
            iteration.status = 'complete'
            iteration.gradient_estimate = gradient
            iteration.elo_diff = elo_diff
            iteration.ref_elo_estimate = ref_elo
            iteration.completed_at = datetime.utcnow()
            db.session.commit()

    with_db_retry(_mark)


def migrate_database():
    """Ensure database schema is up to date with new columns."""
    from web.app import create_app
    from web.database import db

    app = create_app()
    with app.app_context():
        # Check if ref_target_games column exists
        try:
            db.session.execute(db.text("SELECT ref_target_games FROM spsa_iterations LIMIT 1"))
            db.session.commit()
        except Exception:
            # Rollback the failed query before trying ALTER
            db.session.rollback()
            # Column doesn't exist, add it
            print("  Adding ref_target_games column...")
            db.session.execute(db.text(
                "ALTER TABLE spsa_iterations ADD COLUMN ref_target_games INTEGER NOT NULL DEFAULT 100"
            ))
            db.session.commit()
            print("  Migration complete.")


def get_last_complete_iteration_number(run_id: int) -> int:
    """Get the last COMPLETED iteration number for the given run, or 0 if none."""
    from web.app import create_app
    from web.models import SpsaIteration

    def _get():
        app = create_app()
        with app.app_context():
            last = SpsaIteration.query.filter(
                SpsaIteration.run_id == run_id,
                SpsaIteration.status == 'complete'
            ).order_by(SpsaIteration.iteration_number.desc()).first()
            if last:
                return last.iteration_number
            return 0

    return with_db_retry(_get)


def get_incomplete_iteration(run_id: int) -> dict | None:
    """Get an incomplete iteration for the given run that needs to be resumed, or None."""
    from web.app import create_app
    from web.models import SpsaIteration

    def _get():
        app = create_app()
        with app.app_context():
            # Find any iteration in this run that's not marked complete
            # Statuses: pending, in_progress (SPSA phase), building, ref_pending (ref phase)
            iteration = SpsaIteration.query.filter(
                SpsaIteration.run_id == run_id,
                SpsaIteration.status.in_(['pending', 'in_progress', 'building', 'ref_pending'])
            ).order_by(SpsaIteration.iteration_number.desc()).first()

            if not iteration:
                return None

            return {
                'id': iteration.id,
                'iteration_number': iteration.iteration_number,
                'status': iteration.status,
                'games_played': iteration.games_played,
                'target_games': iteration.target_games,
                'ref_games_played': iteration.ref_games_played,
                'ref_target_games': iteration.ref_target_games,
                'plus_wins': iteration.plus_wins,
                'minus_wins': iteration.minus_wins,
                'draws': iteration.draws,
                'ref_wins': iteration.ref_wins,
                'ref_losses': iteration.ref_losses,
                'ref_draws': iteration.ref_draws,
                'base_parameters': iteration.base_parameters,
                'plus_parameters': iteration.plus_parameters,
                'minus_parameters': iteration.minus_parameters,
                'perturbation_signs': iteration.perturbation_signs,
                'base_engine_path': iteration.base_engine_path,
            }

    return with_db_retry(_get)


def get_iteration_results(iteration_id: int) -> dict:
    """Fetch stored SPSA results for a completed/ref_pending iteration."""
    from web.app import create_app
    from web.database import db
    from web.models import SpsaIteration

    def _get():
        app = create_app()
        with app.app_context():
            iteration = db.session.get(SpsaIteration, iteration_id)
            if not iteration:
                raise RuntimeError(f"Iteration {iteration_id} not found!")
            return {
                'gradient_estimate': iteration.gradient_estimate or {},
                'elo_diff': float(iteration.elo_diff) if iteration.elo_diff is not None else None,
            }

    return with_db_retry(_get)


def run_master():
    """Main SPSA master loop with two-phase iteration."""
    print(f"\n{'='*60}")
    print("SPSA MASTER CONTROLLER (Two-Phase)")
    print(f"{'='*60}")

    # Run any pending migrations
    migrate_database()

    # Get the active run
    run_id, run_name = get_active_run()
    print(f"Active run: {run_name} (id={run_id})")

    # Load configuration
    config = load_config()
    params = load_params()

    print(f"Parameters: {len(params)}")
    print(f"SPSA games per iteration: {config['games']['games_per_iteration']}")
    print(f"Time control: {config['time_control']['timelow']}-{config['time_control']['timehigh']}s/move")
    print(f"Max iterations: {config['spsa']['max_iterations']}")

    # SPSA hyperparameters
    a = config['spsa']['a']
    c = config['spsa']['c']
    A = config['spsa']['A']
    alpha = config['spsa']['alpha']
    gamma = config['spsa']['gamma']
    max_iterations = config['spsa']['max_iterations']
    effective_iteration_offset = config['spsa'].get('effective_iteration_offset', 0)
    max_elo_diff = config['spsa'].get('max_elo_diff', 800.0)
    max_gradient_factor = config['spsa'].get('max_gradient_factor', 0.0)
    poll_interval = config['database']['poll_interval_seconds']

    # Reference engine settings
    ref_enabled = config.get('reference', {}).get('enabled', False)
    ref_elo = config.get('reference', {}).get('engine_elo', 2600)
    ref_ratio = config.get('reference', {}).get('ratio', 0.25)
    ref_path_config = config.get('reference', {}).get('engine_path')

    # Calculate ref_target_games based on ratio
    spsa_games = config['games']['games_per_iteration']
    ref_target_games = int(spsa_games * ref_ratio) if ref_enabled else 0

    # Resolve reference engine path
    ref_path = None
    if ref_enabled and ref_path_config:
        ref_path = Path(ref_path_config)
        if not ref_path.is_absolute():
            ref_path = CHESS_COMPETE_DIR / ref_path
        # Check for .exe on Windows
        if os.name == 'nt' and not ref_path.suffix:
            ref_path_exe = ref_path.with_suffix('.exe')
            if ref_path_exe.exists():
                ref_path = ref_path_exe
        ref_path = str(ref_path) if ref_path.exists() else None

    if ref_enabled and ref_path:
        print(f"Reference games: {ref_target_games} per iteration (ratio: {ref_ratio})")
        print(f"Reference engine: {ref_path} (Elo: {ref_elo})")
    else:
        print("Reference games: DISABLED")
        ref_enabled = False

    print(f"{'='*60}")

    # Paths
    output_base = Path(config['build']['engines_output_path'])
    if not output_base.is_absolute():
        output_base = CHESS_COMPETE_DIR / output_base

    print("\nCurrent parameter values:")
    for name, cfg in params.items():
        print(f"  {name}: {cfg['value']} (range: {cfg['min']}-{cfg['max']}, step: {cfg['step']})")

    # Check for incomplete iteration to resume (within this run)
    incomplete = get_incomplete_iteration(run_id)
    if incomplete:
        status = incomplete.get('status', 'pending')
        print(f"\nResuming incomplete iteration {incomplete['iteration_number']} (status: {status})")
        start_iteration = incomplete['iteration_number']
        resume_info = incomplete
    else:
        start_iteration = get_last_complete_iteration_number(run_id) + 1
        resume_info = None
        print(f"\nStarting from iteration {start_iteration}")

    # Main loop
    for k in range(start_iteration, max_iterations + 1):
        print(f"\n{'='*60}")
        print(f"ITERATION {k}/{max_iterations}")
        print(f"{'='*60}")

        # Reload params each iteration to pick up any bound changes
        params = load_params()

        # Calculate effective iteration (for learning rate) and SPSA coefficients
        effective_k = max(1, k - effective_iteration_offset)
        a_k = a / ((effective_k + A) ** alpha)
        c_k = c / (effective_k ** gamma)
        print(f"Iteration {k} (effective: {effective_k})")
        print(f"Coefficients: a_k={a_k:.4f}, c_k={c_k:.4f}")

        # Variables to track where we are in the iteration
        iteration_id = None
        gradient = None
        elo_diff = None

        # Check if we're resuming this iteration
        if resume_info is not None and k == start_iteration:
            iteration_id = resume_info['id']
            status = resume_info.get('status', 'pending')

            if status == 'ref_pending':
                # Already past SPSA phase, just need to finish ref games
                print(f"\nResuming ref phase: {resume_info['ref_games_played']}/{resume_info['ref_target_games']} ref games")
                # Load stored SPSA results so we don't overwrite them on completion
                stored = get_iteration_results(iteration_id)
                gradient = stored.get('gradient_estimate', {}) or {}
                elo_diff = stored.get('elo_diff', 0.0) or 0.0

            elif status in ('pending', 'in_progress'):
                # SPSA phase - check if games are complete
                if resume_info['games_played'] >= resume_info['target_games']:
                    print(f"\nSPSA games complete ({resume_info['games_played']}/{resume_info['target_games']}), processing...")
                    # Need to calculate gradient and continue to ref phase
                    results = {
                        'games_played': resume_info['games_played'],
                        'plus_wins': resume_info['plus_wins'],
                        'minus_wins': resume_info['minus_wins'],
                        'draws': resume_info['draws'],
                        'perturbation_signs': resume_info['perturbation_signs'],
                    }
                    gradient, elo_diff = calculate_gradient(results, params, c_k,
                                                           max_elo_diff=max_elo_diff,
                                                           max_gradient_factor=max_gradient_factor)
                    status = 'spsa_complete'  # Internal marker to continue to ref phase
                else:
                    print(f"\nResuming SPSA phase: {resume_info['games_played']}/{resume_info['target_games']} games")
                    status = 'pending'

            resume_info = None  # Clear so next iteration creates new

        else:
            # New iteration - create it
            plus_params, minus_params, signs = generate_perturbations(params, c_k, k)

            # Count active vs inactive params
            active_count = len(signs)
            inactive_count = len(params) - active_count
            print(f"\nActive parameters: {active_count}, Inactive (frozen): {inactive_count}")

            print("\nPerturbations (active params only):")
            for name in params:
                if name not in signs:
                    continue  # Skip inactive params
                base = params[name]['value']
                plus = plus_params[name]
                minus = minus_params[name]
                sign = '+' if signs[name] > 0 else '-'
                print(f"  {name}: {base:.2f} -> {sign} [{minus:.2f}, {plus:.2f}]")

            # Compute expected engine paths (workers will build from params)
            binary_name = 'rusty-rival.exe' if os.name == 'nt' else 'rusty-rival'
            plus_dir = output_base / config['build']['plus_engine_name']
            minus_dir = output_base / config['build']['minus_engine_name']
            plus_path = plus_dir / binary_name
            minus_path = minus_dir / binary_name

            # Create iteration record (no base_engine_path yet)
            print("\nCreating iteration record...")
            iteration_id = create_iteration(
                run_id, k, effective_k, str(plus_path), str(minus_path), ref_path, ref_target_games,
                config, params, plus_params, minus_params, signs
            )
            print(f"  Iteration ID: {iteration_id}")
            status = 'pending'

        # ===== PHASE 1: SPSA Games =====
        if status in ('pending', 'in_progress'):
            print("\n--- PHASE 1: SPSA Games (plus vs minus) ---")
            print("Waiting for workers to complete SPSA games...")
            results = wait_for_spsa_completion(iteration_id, poll_interval)

            # Calculate gradient
            gradient, elo_diff = calculate_gradient(results, params, c_k,
                                                    max_elo_diff=max_elo_diff,
                                                    max_gradient_factor=max_gradient_factor)
            status = 'spsa_complete'

        # ===== Process SPSA Results =====
        if status == 'spsa_complete':
            print(f"\nSPSA Results: Elo diff = {elo_diff:+.1f}")

            print("\nGradient estimates:")
            for name, g in gradient.items():
                print(f"  {name}: {g:+.4f}")

            # Update parameters
            print("\nUpdating parameters:")
            old_values = {name: cfg['value'] for name, cfg in params.items()}
            params = update_parameters(params, gradient, a_k)

            for name, cfg in params.items():
                old = old_values[name]
                new = cfg['value']
                delta = new - old
                print(f"  {name}: {old:.2f} -> {new:.2f} ({delta:+.4f})")

            # Save updated parameters
            save_params(params)
            print("\nSaved updated parameters to params.toml")

            # ===== Transition to ref phase (workers will build base engine) =====
            if ref_enabled:
                # Compute expected base engine path (workers build it locally)
                binary_name = 'rusty-rival.exe' if os.name == 'nt' else 'rusty-rival'
                base_engine_path = str(output_base / config['build']['base_engine_name'] / binary_name)

                # Transition to ref phase
                updated_param_values = {name: cfg['value'] for name, cfg in params.items()}
                set_ref_phase(iteration_id, base_engine_path, updated_param_values)
                print(f"\nTransitioned to ref phase (status='ref_pending')")
                print("  Workers will build base engine with updated parameters")
                status = 'ref_pending'

        # ===== PHASE 2: Reference Games =====
        ref_elo_estimate = None
        if ref_enabled and status == 'ref_pending':
            print("\n--- PHASE 2: Reference Games (new base vs Stockfish) ---")
            print("Waiting for workers to complete reference games...")
            ref_results = wait_for_ref_completion(iteration_id, poll_interval)

            # Calculate reference Elo
            ref_elo_estimate = calculate_elo_from_score(
                ref_results['ref_wins'], ref_results['ref_losses'],
                ref_results['ref_draws'], ref_elo
            )
            if ref_elo_estimate:
                print(f"\nReference Elo estimate: {ref_elo_estimate:.0f}")

        # ===== Mark iteration complete =====
        mark_iteration_complete(iteration_id, gradient or {}, elo_diff or 0.0, ref_elo_estimate)
        print(f"\nIteration {k} complete!")

    print(f"\n{'='*60}")
    print("SPSA TUNING COMPLETE")
    print(f"{'='*60}")
    print("\nFinal parameter values:")
    for name, cfg in params.items():
        print(f"  {name}: {cfg['value']:.2f}")


if __name__ == '__main__':
    try:
        run_master()
    except KeyboardInterrupt:
        print("\n\nMaster stopped by user.")
        sys.exit(0)
