"""
SPSA Master Controller

Orchestrates the SPSA parameter tuning process with two-phase iteration:

Phase 1 (SPSA Games):
1. Loads current parameters from the database (spsa_params table, keyed by run_id)
2. Generates perturbed parameter sets (plus/minus)
3. Creates iteration record in database (workers build plus/minus engines)
4. Waits for workers to complete SPSA games (plus vs minus)
5. Calculates gradient and updates parameters
6. Saves updated params to the database

Phase 2 (Reference Games):
7. Builds NEW base engine with updated parameters
8. Sets iteration status to 'ref_pending'
9. Waits for workers to complete reference games (new base vs Stockfish)
10. Calculates Elo estimate and marks iteration complete

This two-phase approach ensures reference games measure the actual strength
of the parameter update, not the pre-update baseline.

Usage:
    cd chess-compete
    python -m compete.spsa.master              # interactive run selection
    python -m compete.spsa.master --run "Run 3" # activate or create "Run 3"
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


def _activate_run(db, SpsaRun, run):
    """Deactivate all runs and activate the given one."""
    SpsaRun.query.update({SpsaRun.is_active: False})
    run.is_active = True
    db.session.commit()


def load_params(run_id: int) -> dict:
    """
    Load parameters from the database for a given run.
    Returns dict of {param_name: {'value': X, 'min': Y, 'max': Z, 'step': S, 'group': G}}
    """
    from web.app import create_app
    from web.models import SpsaParam

    def _load():
        app = create_app()
        with app.app_context():
            rows = SpsaParam.query.filter_by(run_id=run_id).all()
            if not rows:
                raise RuntimeError(
                    f"No parameters found for run {run_id}. "
                    "Run migration or use --run to create a new run with params."
                )
            params = {}
            for row in rows:
                params[row.name] = {
                    'value': row.value,
                    'min': row.min_value,
                    'max': row.max_value,
                    'step': row.step,
                    'group': row.group,
                }
            return params

    return with_db_retry(_load)


def save_params(run_id: int, params: dict):
    """
    Save parameter values to the database for a given run.
    Only updates the 'value' column for existing rows.
    """
    from web.app import create_app
    from web.database import db
    from web.models import SpsaParam

    def _save():
        app = create_app()
        with app.app_context():
            for name, cfg in params.items():
                db.session.execute(
                    db.text(
                        "UPDATE spsa_params SET value = :value "
                        "WHERE run_id = :run_id AND name = :name"
                    ),
                    {'value': cfg['value'], 'run_id': run_id, 'name': name}
                )
            db.session.commit()

    with_db_retry(_save)


def load_config() -> dict:
    """Load configuration from config.toml."""
    try:
        import tomllib
    except ImportError:
        import tomli as tomllib
    config_file = SPSA_DIR / 'config.toml'
    with open(config_file, 'rb') as f:
        return tomllib.load(f)


def generate_perturbations(params: dict, c_k: float, active_groups: set | None) -> tuple[dict, dict, dict]:
    """
    Generate perturbed parameter sets for SPSA.

    Args:
        params: Current parameter values with min/max/step/group
        c_k: Perturbation coefficient for this iteration
        active_groups: Set of group names to tune, or None for all groups

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

        # Check if this param's group is active
        if active_groups is not None and cfg.get('group') not in active_groups:
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


def create_iteration(run_id: int, iteration_number: int, effective_iteration: int,
                     ref_path: str | None, ref_target_games: int,
                     timelow: float, timehigh: float, games_per_iteration: int,
                     base_params: dict, plus_params: dict,
                     minus_params: dict, signs: dict) -> int:
    """
    Create a new SPSA iteration record in the database.

    Workers build engines locally from the parameter JSON stored in the
    iteration record — no engine paths are needed.

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
                ref_engine_path=ref_path,
                timelow_ms=int(timelow * 1000),
                timehigh_ms=int(timehigh * 1000),
                target_games=games_per_iteration,
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


def set_ref_phase(iteration_id: int, updated_params: dict):
    """
    Transition iteration to reference game phase.

    Changes status to 'ref_pending' so workers start playing ref games.
    Workers build the base engine locally from the updated parameters.

    Args:
        iteration_id: The iteration to update
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


def save_llm_report(iteration_id: int, report: str):
    """Save an LLM-generated report to a completed iteration."""
    from web.app import create_app
    from web.database import db

    def _save():
        app = create_app()
        with app.app_context():
            db.session.execute(
                db.text("UPDATE spsa_iterations SET llm_report = :report WHERE id = :id"),
                {'report': report, 'id': iteration_id}
            )
            db.session.commit()

    with_db_retry(_save)


def generate_llm_report(run_id: int, iteration_number: int) -> str | None:
    """
    Generate an LLM analysis report for the current state of an SPSA run.

    Queries the last 10 completed iterations, builds a prompt with parameter
    and Elo data, and calls OpenAI gpt-4o-mini for a concise analysis.

    Returns the report text, or None if generation fails or is unavailable.
    """
    import os
    api_key = os.environ.get('OPENAI_API_KEY')
    if not api_key:
        print("  LLM report skipped: OPENAI_API_KEY not set")
        return None

    try:
        from openai import OpenAI
    except ImportError:
        print("  LLM report skipped: openai package not installed")
        return None

    from web.app import create_app
    from web.models import SpsaIteration, SpsaRun, SpsaParam

    try:
        app = create_app()
        with app.app_context():
            # Load run info
            run = SpsaRun.query.get(run_id)
            if not run:
                return None

            # Load parameters with bounds
            params = SpsaParam.query.filter_by(run_id=run_id).all()
            active_groups = set(run.active_groups) if run.active_groups else None

            # Load last 10 completed iterations
            recent_iterations = SpsaIteration.query.filter(
                SpsaIteration.run_id == run_id,
                SpsaIteration.status == 'complete'
            ).order_by(SpsaIteration.iteration_number.desc()).limit(10).all()
            recent_iterations.reverse()  # chronological order

            if not recent_iterations:
                return None

            # Build parameter table
            param_lines = []
            for p in sorted(params, key=lambda x: x.name):
                active = '*' if (active_groups is None or p.group in active_groups) else ' '
                current_val = recent_iterations[-1].base_parameters.get(p.name, p.value) if recent_iterations[-1].base_parameters else p.value
                param_lines.append(
                    f"  {active} {p.name}: current={current_val:.4f}, "
                    f"min={p.min_value}, max={p.max_value}, initial={p.value:.4f}, "
                    f"group={p.group}"
                )

            # Build iteration table
            iter_lines = []
            for it in recent_iterations:
                elo_diff = float(it.elo_diff) if it.elo_diff else 0
                ref_elo = float(it.ref_elo_estimate) if it.ref_elo_estimate else None
                ref_str = f"{ref_elo:.0f}" if ref_elo else "n/a"

                # Summarize gradient magnitudes
                grad_mag = "n/a"
                if it.gradient_estimate:
                    mags = [abs(v) for v in it.gradient_estimate.values()]
                    if mags:
                        grad_mag = f"avg={sum(mags)/len(mags):.4f}, max={max(mags):.4f}"

                iter_lines.append(
                    f"  Iter {it.iteration_number}: elo_diff={elo_diff:+.1f}, "
                    f"ref_elo={ref_str}, gradient=[{grad_mag}]"
                )

            # Compute convergence metrics (reuse logic from routes.py)
            convergence_info = ""
            if len(recent_iterations) >= 2:
                updates = []
                for i in range(1, len(recent_iterations)):
                    prev = recent_iterations[i-1].base_parameters or {}
                    curr = recent_iterations[i].base_parameters or {}
                    for name in curr:
                        if name in prev and prev[name] != 0:
                            pct = abs(curr[name] - prev[name]) / abs(prev[name]) * 100
                            updates.append(pct)
                if updates:
                    avg_update = sum(updates) / len(updates)
                    convergence_info = f"Recent avg parameter update: {avg_update:.3f}%/iter"

            prompt = f"""You are analyzing an SPSA chess engine parameter tuning run. Provide a concise report (3-5 paragraphs).

Run: {run.name} — {run.description or 'No description'}
Active groups: {', '.join(sorted(active_groups)) if active_groups else 'ALL'}
Current iteration: {iteration_number}
Games per iteration: {run.games_per_iteration}
Time control: {run.timelow}-{run.timehigh}s/move

Parameters (current value, min, max, initial) — * = active:
{chr(10).join(param_lines)}

Recent iterations (last {len(recent_iterations)}):
{chr(10).join(iter_lines)}

{convergence_info}

Please analyze:
1. Overall trend — is the run improving, plateauing, or degrading?
2. Parameter convergence — which params have stabilized vs still moving?
3. Any params near or at bounds?
4. Recommendation: continue, stop, or adjust settings?
Keep it concise and actionable."""

            # Call OpenAI
            client = OpenAI(api_key=api_key)
            response = client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[{"role": "user", "content": prompt}],
                max_tokens=500,
                temperature=0.3,
            )
            report = response.choices[0].message.content
            return report.strip() if report else None

    except Exception as e:
        print(f"  LLM report generation failed: {e}")
        return None


def migrate_database():
    """Ensure database schema is up to date with new columns/tables."""
    from web.app import create_app
    from web.database import db

    app = create_app()
    with app.app_context():
        # Check if ref_target_games column exists
        try:
            db.session.execute(db.text("SELECT ref_target_games FROM spsa_iterations LIMIT 1"))
            db.session.commit()
        except Exception:
            db.session.rollback()
            print("  Adding ref_target_games column...")
            db.session.execute(db.text(
                "ALTER TABLE spsa_iterations ADD COLUMN ref_target_games INTEGER NOT NULL DEFAULT 100"
            ))
            db.session.commit()
            print("  Migration complete.")

        # Drop unused engine path columns (idempotent)
        for col in ('plus_engine_path', 'minus_engine_path', 'base_engine_path'):
            try:
                db.session.execute(db.text(
                    f"ALTER TABLE spsa_iterations DROP COLUMN IF EXISTS {col}"
                ))
                db.session.commit()
            except Exception:
                db.session.rollback()

        # Create spsa_params table if it doesn't exist
        try:
            db.session.execute(db.text("SELECT 1 FROM spsa_params LIMIT 1"))
            db.session.commit()
        except Exception:
            db.session.rollback()
            print("  Creating spsa_params table...")
            db.session.execute(db.text("""
                CREATE TABLE spsa_params (
                    id SERIAL PRIMARY KEY,
                    run_id INTEGER NOT NULL REFERENCES spsa_runs(id),
                    name VARCHAR(100) NOT NULL,
                    value DOUBLE PRECISION NOT NULL,
                    min_value DOUBLE PRECISION NOT NULL,
                    max_value DOUBLE PRECISION NOT NULL,
                    step DOUBLE PRECISION NOT NULL,
                    "group" VARCHAR(50) NOT NULL DEFAULT 'ungrouped',
                    CONSTRAINT uq_spsa_param_run_name UNIQUE (run_id, name)
                )
            """))
            db.session.commit()
            print("  spsa_params table created.")

        # Add effective_iteration_offset column to spsa_runs (per-run offset)
        try:
            db.session.execute(db.text("SELECT effective_iteration_offset FROM spsa_runs LIMIT 1"))
            db.session.commit()
        except Exception:
            db.session.rollback()
            print("  Adding effective_iteration_offset column to spsa_runs...")
            db.session.execute(db.text(
                "ALTER TABLE spsa_runs ADD COLUMN effective_iteration_offset INTEGER NOT NULL DEFAULT 0"
            ))
            # Seed Run 2 with the legacy offset (118); all others default to 0
            db.session.execute(db.text(
                "UPDATE spsa_runs SET effective_iteration_offset = 118 WHERE id = 2"
            ))
            db.session.commit()
            print("  Migration complete (seeded Run 2 with offset=118).")

        # Add timelow/timehigh columns to spsa_runs (per-run time control)
        try:
            db.session.execute(db.text("SELECT timelow FROM spsa_runs LIMIT 1"))
            db.session.commit()
        except Exception:
            db.session.rollback()
            print("  Adding timelow/timehigh columns to spsa_runs...")
            db.session.execute(db.text(
                "ALTER TABLE spsa_runs ADD COLUMN timelow DOUBLE PRECISION NOT NULL DEFAULT 0.1"
            ))
            db.session.execute(db.text(
                "ALTER TABLE spsa_runs ADD COLUMN timehigh DOUBLE PRECISION NOT NULL DEFAULT 0.2"
            ))
            # Run 3 keeps defaults (0.1-0.2); Runs 1 & 2 used 0.25-5.0
            db.session.execute(db.text(
                "UPDATE spsa_runs SET timelow = 0.25, timehigh = 5.0 WHERE id IN (1, 2)"
            ))
            db.session.commit()
            print("  Migration complete (Runs 1&2: 0.25-5.0s, Run 3: 0.1-0.2s).")

        # Add SPSA hyperparameter and reference columns to spsa_runs
        try:
            db.session.execute(db.text("SELECT games_per_iteration FROM spsa_runs LIMIT 1"))
            db.session.commit()
        except Exception:
            db.session.rollback()
            print("  Adding SPSA/reference columns to spsa_runs...")
            columns = [
                ("games_per_iteration", "INTEGER NOT NULL DEFAULT 500"),
                ("max_iterations", "INTEGER NOT NULL DEFAULT 1500"),
                ("spsa_a", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
                ("spsa_c", "DOUBLE PRECISION NOT NULL DEFAULT 1.0"),
                ("spsa_big_a", "DOUBLE PRECISION NOT NULL DEFAULT 50"),
                ("spsa_alpha", "DOUBLE PRECISION NOT NULL DEFAULT 0.602"),
                ("spsa_gamma", "DOUBLE PRECISION NOT NULL DEFAULT 0.101"),
                ("max_elo_diff", "DOUBLE PRECISION NOT NULL DEFAULT 100.0"),
                ("max_gradient_factor", "DOUBLE PRECISION NOT NULL DEFAULT 3.0"),
                ("ref_enabled", "BOOLEAN NOT NULL DEFAULT TRUE"),
                ("ref_ratio", "DOUBLE PRECISION NOT NULL DEFAULT 0.25"),
            ]
            for col_name, col_type in columns:
                db.session.execute(db.text(
                    f"ALTER TABLE spsa_runs ADD COLUMN {col_name} {col_type}"
                ))
            db.session.commit()
            print("  All columns added (defaults match current config values).")

        # Add active_groups column to spsa_runs
        try:
            db.session.execute(db.text("SELECT active_groups FROM spsa_runs LIMIT 1"))
            db.session.commit()
        except Exception:
            db.session.rollback()
            print("  Adding active_groups column to spsa_runs...")
            db.session.execute(db.text(
                "ALTER TABLE spsa_runs ADD COLUMN active_groups JSON"
            ))
            db.session.commit()
            print("  Migration complete.")

        # Add group column to spsa_params and migrate from active_from_iteration
        try:
            db.session.execute(db.text('SELECT "group" FROM spsa_params LIMIT 1'))
            db.session.commit()
        except Exception:
            db.session.rollback()
            print("  Adding group column to spsa_params...")
            db.session.execute(db.text(
                "ALTER TABLE spsa_params ADD COLUMN \"group\" VARCHAR(50) NOT NULL DEFAULT 'ungrouped'"
            ))
            db.session.commit()

            # Migrate existing params: map by name prefix to groups
            print("  Migrating parameters to groups by name prefix...")
            group_mapping = {
                # search-pruning
                'beta_prune_': 'search-pruning',
                'null_move_': 'search-pruning',
                'see_prune_': 'search-pruning',
                'alpha_prune_': 'search-pruning',
                'lmp_threshold_': 'search-pruning',
                # reductions-extensions
                'lmr_history_': 'reductions-extensions',
                'lmr_continuation_': 'reductions-extensions',
                'probcut_': 'reductions-extensions',
                'multicut_': 'reductions-extensions',
                'singular_extension_': 'reductions-extensions',
                # move-ordering
                'move_score_': 'move-ordering',
                'countermove_history_divisor': 'move-ordering',
                'followup_history_divisor': 'move-ordering',
                'capture_history_divisor': 'move-ordering',
                # evaluation
                'rook_open_file_bonus': 'evaluation',
                'rook_semi_open_file_bonus': 'evaluation',
                'value_knight_outpost': 'evaluation',
                'value_rook_behind_passed_pawn': 'evaluation',
                'value_guarded_passed_pawn': 'evaluation',
                'doubled_pawn_penalty': 'evaluation',
                'isolated_pawn_penalty': 'evaluation',
                'value_backward_pawn_penalty': 'evaluation',
                'value_bishop_pair': 'evaluation',
                'space_bonus_': 'evaluation',
                # piece-values
                'pawn_value_': 'piece-values',
                'knight_value_': 'piece-values',
                'bishop_value_': 'piece-values',
                'rook_value_': 'piece-values',
                'queen_value_': 'piece-values',
            }

            rows = db.session.execute(db.text("SELECT id, name FROM spsa_params")).fetchall()
            for row in rows:
                param_id, param_name = row[0], row[1]
                group = 'ungrouped'
                for prefix, grp in group_mapping.items():
                    if param_name == prefix or param_name.startswith(prefix):
                        group = grp
                        break
                db.session.execute(db.text(
                    'UPDATE spsa_params SET "group" = :group WHERE id = :id'
                ), {'group': group, 'id': param_id})
            db.session.commit()
            print("  Group migration complete.")

            # Drop old active_from_iteration column
            try:
                db.session.execute(db.text(
                    "ALTER TABLE spsa_params DROP COLUMN IF EXISTS active_from_iteration"
                ))
                db.session.commit()
                print("  Dropped active_from_iteration column.")
            except Exception:
                db.session.rollback()

        # Add llm_report column to spsa_iterations
        try:
            db.session.execute(db.text("SELECT llm_report FROM spsa_iterations LIMIT 1"))
            db.session.commit()
        except Exception:
            db.session.rollback()
            print("  Adding llm_report column to spsa_iterations...")
            db.session.execute(db.text(
                "ALTER TABLE spsa_iterations ADD COLUMN llm_report TEXT"
            ))
            db.session.commit()
            print("  Migration complete.")

        # Ensure a "Default" template run exists
        default_exists = db.session.execute(
            db.text("SELECT 1 FROM spsa_runs WHERE name = 'Default' LIMIT 1")
        ).fetchone()
        if not default_exists:
            print("  Creating 'Default' template run...")
            db.session.execute(db.text(
                "INSERT INTO spsa_runs (name, is_active) VALUES ('Default', FALSE)"
            ))
            db.session.commit()
            print("  Default run created.")


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


def _create_new_run(db, SpsaRun, SpsaParam) -> tuple[int, str]:
    """Interactively create a new run, seeding params from defaults or an existing run."""
    run_name = input("\nEnter name for new run: ").strip()
    if not run_name:
        raise RuntimeError("Run name cannot be empty.")

    # Get existing runs with params for "copy" option (exclude Default template)
    all_runs = SpsaRun.query.filter(SpsaRun.name != 'Default').order_by(SpsaRun.id).all()
    runs_with_params = []
    for r in all_runs:
        count = SpsaParam.query.filter_by(run_id=r.id).count()
        if count > 0:
            runs_with_params.append(r)

    if not runs_with_params:
        raise RuntimeError(
            "Cannot create a new run: no existing runs with parameters to copy bounds from."
        )

    print("\nHow should parameters be initialized?")
    print("  0. defaults — use midpoint of min/max bounds")
    for r in runs_with_params:
        print(f"  {r.id}. copy from {r.name}")
    print()

    valid_ids = {r.id for r in runs_with_params}
    while True:
        choice = input("Enter choice (0 for defaults, or run ID to copy): ").strip()
        if choice == '0':
            seed_mode = 'defaults'
            source_run_id = None
            break
        try:
            source_run_id = int(choice)
            if source_run_id in valid_ids:
                seed_mode = 'copy'
                break
        except ValueError:
            pass
        print(f"Invalid choice. Enter 0 or a run ID ({', '.join(str(i) for i in sorted(valid_ids))}).")

    # Copy run-level settings from the Default template run
    default_run = SpsaRun.query.filter_by(name='Default').first()
    if not default_run:
        raise RuntimeError("No 'Default' template run found. Run migration first.")

    # If copying from an existing run, use its settings as defaults for prompts
    settings_source = default_run
    if seed_mode == 'copy':
        source_run = db.session.get(SpsaRun, source_run_id)
        if source_run:
            settings_source = source_run

    # Ask about key run settings (press Enter to accept defaults)
    def _prompt_float(label: str, default: float) -> float:
        val = input(f"  {label} [{default}]: ").strip()
        if not val:
            return default
        try:
            return float(val)
        except ValueError:
            print(f"    Invalid, using default: {default}")
            return default

    def _prompt_int(label: str, default: int) -> int:
        val = input(f"  {label} [{default}]: ").strip()
        if not val:
            return default
        try:
            return int(val)
        except ValueError:
            print(f"    Invalid, using default: {default}")
            return default

    print(f"\nRun settings (press Enter for defaults from '{settings_source.name}'):")
    timelow = _prompt_float("Time low (s/move)", settings_source.timelow)
    timehigh = _prompt_float("Time high (s/move)", settings_source.timehigh)
    games_per_iteration = _prompt_int("Games per iteration", settings_source.games_per_iteration)
    max_iterations = _prompt_int("Max iterations", settings_source.max_iterations)

    # Determine available groups from source params
    if seed_mode == 'copy':
        source_params = SpsaParam.query.filter_by(run_id=source_run_id).all()
    else:
        source_params = SpsaParam.query.filter_by(run_id=runs_with_params[-1].id).all()

    available_groups = {}
    for p in source_params:
        grp = p.group or 'ungrouped'
        available_groups[grp] = available_groups.get(grp, 0) + 1

    print("\nAvailable parameter groups:")
    sorted_groups = sorted(available_groups.items())
    for i, (grp, count) in enumerate(sorted_groups, 1):
        print(f"  {i}. {grp} ({count} params)")
    print(f"  A. All groups (tune everything)")

    while True:
        choice = input("\nSelect groups to activate (comma-separated numbers, or A for all): ").strip()
        if choice.upper() == 'A':
            selected_groups = None  # None = all groups active
            break
        try:
            indices = [int(x.strip()) for x in choice.split(',')]
            if all(1 <= i <= len(sorted_groups) for i in indices):
                selected_groups = [sorted_groups[i - 1][0] for i in indices]
                break
        except ValueError:
            pass
        print(f"Invalid choice. Enter comma-separated numbers (1-{len(sorted_groups)}) or 'A'.")

    if selected_groups:
        active_count = sum(available_groups[g] for g in selected_groups)
        print(f"  Active groups: {', '.join(selected_groups)} ({active_count} params)")
    else:
        print(f"  All groups active ({sum(available_groups.values())} params)")

    # Deactivate all runs, create the new one
    SpsaRun.query.update({SpsaRun.is_active: False})
    new_run = SpsaRun(
        name=run_name, is_active=True,
        timelow=timelow, timehigh=timehigh,
        games_per_iteration=games_per_iteration,
        max_iterations=max_iterations,
        spsa_a=settings_source.spsa_a, spsa_c=settings_source.spsa_c,
        spsa_big_a=settings_source.spsa_big_a,
        spsa_alpha=settings_source.spsa_alpha, spsa_gamma=settings_source.spsa_gamma,
        max_elo_diff=settings_source.max_elo_diff,
        max_gradient_factor=settings_source.max_gradient_factor,
        ref_enabled=settings_source.ref_enabled, ref_ratio=settings_source.ref_ratio,
        active_groups=selected_groups,
    )
    db.session.add(new_run)
    db.session.flush()  # get ID

    if seed_mode == 'defaults':
        print(f"  Seeding with midpoint defaults (bounds from run {runs_with_params[-1].id})...")
        for p in source_params:
            midpoint = (p.min_value + p.max_value) / 2.0
            db.session.add(SpsaParam(
                run_id=new_run.id, name=p.name,
                value=midpoint, min_value=p.min_value,
                max_value=p.max_value, step=p.step,
                group=p.group,
            ))
    elif seed_mode == 'copy':
        print(f"  Copying params from run {source_run_id}...")
        for p in source_params:
            db.session.add(SpsaParam(
                run_id=new_run.id, name=p.name,
                value=p.value, min_value=p.min_value,
                max_value=p.max_value, step=p.step,
                group=p.group,
            ))

    db.session.commit()
    param_count = SpsaParam.query.filter_by(run_id=new_run.id).count()
    print(f"  Created run '{run_name}' (id={new_run.id}) with {param_count} params")
    return new_run.id, new_run.name


def select_run() -> tuple[int, str]:
    """
    Select which SPSA run to use.

    Non-interactive mode (INTERACTIVE_MODE_DISABLED=true):
        Uses RUN_ID env var to select run by ID. Raises error if not set or not found.

    Interactive mode:
        If --run <name> is provided, activates that run (or creates it interactively).
        Otherwise, lists all runs and prompts the user to select one or create a new one.

    Returns (run_id, run_name).
    """
    from web.app import create_app
    from web.database import db
    from web.models import SpsaRun, SpsaParam

    # Non-interactive mode: use RUN_ID env var
    if os.environ.get('INTERACTIVE_MODE_DISABLED', '').lower() == 'true':
        run_id_str = os.environ.get('RUN_ID')
        if not run_id_str:
            raise RuntimeError(
                "INTERACTIVE_MODE_DISABLED=true but RUN_ID env var is not set. "
                "Set RUN_ID to the numeric ID of the run to use."
            )
        try:
            run_id = int(run_id_str)
        except ValueError:
            raise RuntimeError(f"RUN_ID must be a numeric ID, got: '{run_id_str}'")

        app = create_app()
        with app.app_context():
            selected = db.session.get(SpsaRun, run_id)
            if not selected:
                raise RuntimeError(f"Run with ID {run_id} not found in database.")
            _activate_run(db, SpsaRun, selected)
            print(f"Non-interactive mode: activated run '{selected.name}' (id={selected.id})")
            return selected.id, selected.name

    import argparse

    parser = argparse.ArgumentParser(description='SPSA Master Controller')
    parser.add_argument('--run', type=str, help='Run name to use or create')
    args = parser.parse_args()

    app = create_app()
    with app.app_context():
        # If --run provided, find or create that specific run
        if args.run is not None:
            existing = SpsaRun.query.filter_by(name=args.run).first()
            if existing:
                _activate_run(db, SpsaRun, existing)
                print(f"Activated run: {existing.name} (id={existing.id})")
                return existing.id, existing.name
            else:
                print(f"Run '{args.run}' does not exist.")
                return _create_new_run(db, SpsaRun, SpsaParam)

        # No --run argument: show interactive menu (exclude Default template)
        all_runs = SpsaRun.query.filter(SpsaRun.name != 'Default').order_by(SpsaRun.id).all()

        if not all_runs:
            raise RuntimeError(
                "No SPSA runs exist. Create one first:\n"
                "  INSERT INTO spsa_runs (name, description, is_active) "
                "VALUES ('Run 1', 'Description', TRUE);"
            )

        active_run = SpsaRun.query.filter_by(is_active=True).first()

        print("\nAvailable runs:")
        for r in all_runs:
            # Get iteration count and param count
            from web.models import SpsaIteration
            iter_count = SpsaIteration.query.filter_by(
                run_id=r.id, status='complete'
            ).count()
            param_count = SpsaParam.query.filter_by(run_id=r.id).count()
            active_marker = " *" if r.is_active else ""
            print(f"  {r.id}. {r.name} ({iter_count} iterations, {param_count} params){active_marker}")

        print(f"  N. Create new run")
        if active_run:
            print(f"\n  * = currently active")
            default_hint = f" [default: {active_run.id}]"
        else:
            default_hint = ""

        while True:
            choice = input(f"\nSelect run{default_hint}: ").strip()

            # Default to active run if user just presses Enter
            if not choice and active_run:
                _activate_run(db, SpsaRun, active_run)
                return active_run.id, active_run.name

            if choice.upper() == 'N':
                return _create_new_run(db, SpsaRun, SpsaParam)

            try:
                run_id = int(choice)
                selected = SpsaRun.query.get(run_id)
                if selected:
                    _activate_run(db, SpsaRun, selected)
                    print(f"Activated run: {selected.name} (id={selected.id})")
                    return selected.id, selected.name
            except ValueError:
                pass

            print("Invalid choice. Enter a run number or 'N' for new.")


def get_run_settings(run_id: int) -> dict:
    """Load all per-run settings from the database."""
    from web.app import create_app
    from web.database import db
    from web.models import SpsaRun

    def _get():
        app = create_app()
        with app.app_context():
            run = db.session.get(SpsaRun, run_id)
            if not run:
                raise RuntimeError(f"Run {run_id} not found!")
            return {
                'effective_iteration_offset': run.effective_iteration_offset,
                'timelow': run.timelow,
                'timehigh': run.timehigh,
                'games_per_iteration': run.games_per_iteration,
                'max_iterations': run.max_iterations,
                'a': run.spsa_a,
                'c': run.spsa_c,
                'A': run.spsa_big_a,
                'alpha': run.spsa_alpha,
                'gamma': run.spsa_gamma,
                'max_elo_diff': run.max_elo_diff,
                'max_gradient_factor': run.max_gradient_factor,
                'ref_enabled': run.ref_enabled,
                'ref_ratio': run.ref_ratio,
                'active_groups': set(run.active_groups) if run.active_groups else None,
            }

    return with_db_retry(_get)


def run_master():
    """Main SPSA master loop with two-phase iteration."""
    print(f"\n{'='*60}")
    print("SPSA MASTER CONTROLLER (Two-Phase)")
    print(f"{'='*60}")

    # Run any pending migrations
    migrate_database()

    # Select or create a run (interactive or via --run flag)
    run_id, run_name = select_run()
    print(f"Active run: {run_name} (id={run_id})")

    # Load configuration (infrastructure settings only)
    config = load_config()
    poll_interval = config['database']['poll_interval_seconds']
    ref_elo = config.get('reference', {}).get('engine_elo', 2600)
    ref_path_config = config.get('reference', {}).get('engine_path')

    # Resolve reference engine path (informational only — workers have their own)
    ref_path = None
    if ref_path_config:
        ref_path = Path(ref_path_config)
        if not ref_path.is_absolute():
            ref_path = CHESS_COMPETE_DIR / ref_path
        if os.name == 'nt' and not ref_path.suffix:
            ref_path_exe = ref_path.with_suffix('.exe')
            if ref_path_exe.exists():
                ref_path = ref_path_exe
        if not ref_path.exists():
            print(f"  Note: reference engine not found locally ({ref_path}), workers will use their own")
        ref_path = str(ref_path)

    # Print initial settings
    run_settings = get_run_settings(run_id)
    params = load_params(run_id)
    print(f"Parameters: {len(params)}")
    active_groups = run_settings['active_groups']
    if active_groups:
        active_count = sum(1 for p in params.values() if p.get('group') in active_groups)
        print(f"Active groups: {', '.join(sorted(active_groups))} ({active_count} params)")
    else:
        print(f"Active groups: ALL ({len(params)} params)")
    print(f"SPSA games per iteration: {run_settings['games_per_iteration']}")
    print(f"Time control: {run_settings['timelow']}-{run_settings['timehigh']}s/move")
    print(f"Max iterations: {run_settings['max_iterations']}")

    if run_settings['ref_enabled']:
        ref_target = int(run_settings['games_per_iteration'] * run_settings['ref_ratio'])
        print(f"Reference games: {ref_target} per iteration (ratio: {run_settings['ref_ratio']})")
        if ref_path:
            print(f"Reference engine: {ref_path} (Elo: {ref_elo})")
    else:
        print("Reference games: DISABLED")

    print(f"{'='*60}")

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
    k = start_iteration
    while True:
        # Reload run settings and params from DB each iteration
        run_settings = get_run_settings(run_id)
        max_iterations = run_settings['max_iterations']

        if k > max_iterations:
            break

        effective_iteration_offset = run_settings['effective_iteration_offset']
        timelow = run_settings['timelow']
        timehigh = run_settings['timehigh']
        games_per_iteration = run_settings['games_per_iteration']
        a = run_settings['a']
        c = run_settings['c']
        A = run_settings['A']
        alpha = run_settings['alpha']
        gamma = run_settings['gamma']
        max_elo_diff = run_settings['max_elo_diff']
        max_gradient_factor = run_settings['max_gradient_factor']
        ref_enabled = run_settings['ref_enabled']
        ref_ratio = run_settings['ref_ratio']
        ref_target_games = int(games_per_iteration * ref_ratio) if ref_enabled else 0

        print(f"\n{'='*60}")
        print(f"ITERATION {k}/{max_iterations}")
        print(f"{'='*60}")

        params = load_params(run_id)

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
            active_groups = run_settings['active_groups']
            plus_params, minus_params, signs = generate_perturbations(params, c_k, active_groups)

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

            # Create iteration record (workers build engines locally from params)
            print("\nCreating iteration record...")
            iteration_id = create_iteration(
                run_id, k, effective_k, ref_path, ref_target_games,
                timelow, timehigh, games_per_iteration,
                params, plus_params, minus_params, signs
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

            # Save updated parameters to DB
            save_params(run_id, params)
            print("\nSaved updated parameters to database")

            # ===== Transition to ref phase (workers will build base engine) =====
            if ref_enabled:
                # Transition to ref phase
                updated_param_values = {name: cfg['value'] for name, cfg in params.items()}
                set_ref_phase(iteration_id, updated_param_values)
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

        # Generate LLM report every 5 iterations
        if k % 5 == 0:
            try:
                report = generate_llm_report(run_id, k)
                if report:
                    save_llm_report(iteration_id, report)
                    print(f"  LLM report saved for iteration {k}")
            except Exception as e:
                print(f"  LLM report failed: {e}")

        k += 1

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
