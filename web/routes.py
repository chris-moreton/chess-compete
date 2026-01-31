"""
Flask routes for the competition dashboard.
"""

from flask import render_template, request, redirect, url_for
from web.queries import (
    get_dashboard_data, get_unique_hostnames, get_time_range,
    clear_elo_cache, recalculate_all_and_store
)
from web.models import Cup, CupRound, CupMatch, Engine, Game, EpdTestRun, EpdTestResult, SpsaIteration
from web.database import db
from sqlalchemy import func, case, and_, or_


def register_routes(app):
    """Register all routes with the Flask app."""

    @app.route('/')
    def dashboard():
        """Main dashboard showing H2H grid with optional filters."""
        # Check for active_only parameter (default True)
        active_only = request.args.get('all') != '1'

        # Get filter parameters
        min_time = request.args.get('min_time', type=int)
        max_time = request.args.get('max_time', type=int)
        hostname = request.args.get('hostname') or None  # Empty string -> None
        engine_type = request.args.get('engine_type') or None  # Empty string -> None

        # Get sort parameters (for preserving across refreshes)
        sort_column = request.args.get('sort') or None
        sort_direction = request.args.get('sort_dir') or None

        # Get actual time range from database for slider defaults
        db_min_time, db_max_time = get_time_range()

        # Use database range as defaults if no filter specified
        if min_time is None:
            min_time = db_min_time
        if max_time is None:
            max_time = db_max_time

        # Ensure min <= max
        if min_time > max_time:
            min_time, max_time = max_time, min_time

        engines, grid, column_headers, last_played = get_dashboard_data(
            active_only=active_only,
            min_time_ms=min_time,
            max_time_ms=max_time,
            hostname=hostname,
            engine_type=engine_type,
            sort_column=sort_column,
            sort_direction=sort_direction
        )

        # Get list of unique hostnames for dropdown
        hostnames = get_unique_hostnames()

        return render_template(
            'dashboard.html',
            grid=grid,
            column_headers=column_headers,
            active_only=active_only,
            total_engines=len(engines),
            last_played=last_played,
            # Filter values
            min_time=min_time,
            max_time=max_time,
            db_min_time=db_min_time,
            db_max_time=db_max_time,
            hostname=hostname,
            hostnames=hostnames,
            engine_type=engine_type,
            # Sort values (for preserving across refreshes)
            sort_column=sort_column,
            sort_direction=sort_direction
        )

    @app.route('/force-recalculate', methods=['POST'])
    def force_recalculate():
        """Clear cache and recalculate all rating systems (Elo, BayesElo, Ordo)."""
        # Get filter parameters from form
        min_time = request.form.get('min_time', type=int)
        max_time = request.form.get('max_time', type=int)
        hostname = request.form.get('hostname') or None
        engine_type = request.form.get('engine_type') or None
        show_all = request.form.get('all') == '1'
        sort_column = request.form.get('sort') or None
        sort_direction = request.form.get('sort_dir') or None

        # Get defaults from database
        db_min_time, db_max_time = get_time_range()
        if min_time is None:
            min_time = db_min_time
        if max_time is None:
            max_time = db_max_time

        # Recalculate all ratings for this filter
        recalculate_all_and_store(min_time, max_time, hostname, engine_type)

        # Build redirect URL with filter params
        params = []
        if show_all:
            params.append('all=1')
        if min_time != db_min_time:
            params.append(f'min_time={min_time}')
        if max_time != db_max_time:
            params.append(f'max_time={max_time}')
        if hostname:
            params.append(f'hostname={hostname}')
        if engine_type:
            params.append(f'engine_type={engine_type}')
        if sort_column:
            params.append(f'sort={sort_column}')
        if sort_direction:
            params.append(f'sort_dir={sort_direction}')

        redirect_url = '/?' + '&'.join(params) if params else '/'
        return redirect(redirect_url)

    @app.route('/clear-cache', methods=['POST'])
    def clear_cache():
        """Clear all ELO filter caches."""
        clear_elo_cache()
        return redirect(url_for('dashboard'))

    @app.route('/cups')
    def cups_list():
        """List all cups."""
        cups = Cup.query.order_by(Cup.created_at.desc()).all()

        # Enrich cups with winner name
        cups_data = []
        for cup in cups:
            winner_name = None
            if cup.winner_engine_id:
                winner = Engine.query.get(cup.winner_engine_id)
                if winner:
                    winner_name = winner.name

            cups_data.append({
                'id': cup.id,
                'name': cup.name,
                'status': cup.status,
                'num_participants': cup.num_participants,
                'games_per_match': cup.games_per_match,
                'time_per_move_ms': cup.time_per_move_ms,
                'time_low_ms': cup.time_low_ms,
                'time_high_ms': cup.time_high_ms,
                'winner_name': winner_name,
                'hostname': cup.hostname,
                'created_at': cup.created_at,
                'completed_at': cup.completed_at,
            })

        return render_template('cups_list.html', cups=cups_data)

    @app.route('/cup/<int:cup_id>')
    def cup_detail(cup_id):
        """Show cup bracket and results."""
        cup = Cup.query.get_or_404(cup_id)

        # Get winner name
        winner_name = None
        if cup.winner_engine_id:
            winner = Engine.query.get(cup.winner_engine_id)
            if winner:
                winner_name = winner.name

        # Get rounds with matches
        rounds_data = []
        for cup_round in cup.rounds:
            matches_data = []
            for match in cup_round.matches:
                engine1 = Engine.query.get(match.engine1_id)
                engine2 = Engine.query.get(match.engine2_id) if match.engine2_id else None
                winner = Engine.query.get(match.winner_engine_id) if match.winner_engine_id else None

                matches_data.append({
                    'id': match.id,
                    'match_order': match.match_order,
                    'engine1_name': engine1.name if engine1 else 'Unknown',
                    'engine2_name': engine2.name if engine2 else 'BYE',
                    'engine1_seed': match.engine1_seed,
                    'engine2_seed': match.engine2_seed,
                    'engine1_points': float(match.engine1_points) if match.engine1_points else 0,
                    'engine2_points': float(match.engine2_points) if match.engine2_points else 0,
                    'games_played': match.games_played,
                    'winner_name': winner.name if winner else None,
                    'status': match.status,
                    'is_bye': match.status == 'bye',
                    'is_tiebreaker': match.is_tiebreaker,
                    'decided_by_coin_flip': match.decided_by_coin_flip,
                })

            rounds_data.append({
                'round_number': cup_round.round_number,
                'round_name': cup_round.round_name,
                'status': cup_round.status,
                'matches': matches_data,
            })

        # Format time control
        if cup.time_per_move_ms:
            time_control = f"{cup.time_per_move_ms / 1000:.1f}s/move"
        elif cup.time_low_ms and cup.time_high_ms:
            time_control = f"{cup.time_low_ms / 1000:.1f}-{cup.time_high_ms / 1000:.1f}s/move"
        else:
            time_control = "Unknown"

        return render_template(
            'cup.html',
            cup=cup,
            winner_name=winner_name,
            rounds=rounds_data,
            time_control=time_control,
        )

    @app.route('/engines')
    def engines_list():
        """Engine management page."""
        active_only = request.args.get('all') != '1'
        engine_type = request.args.get('engine_type') or None

        # Build query
        query = Engine.query

        if active_only:
            query = query.filter(Engine.active == True)

        if engine_type == 'rusty':
            query = query.filter(Engine.name.like('v%'))
        elif engine_type == 'stockfish':
            query = query.filter(Engine.name.like('sf-%'))

        engines = query.order_by(Engine.name).all()

        # Get game counts for each engine
        engine_data = []
        for engine in engines:
            games_count = Game.query.filter(
                or_(Game.white_engine_id == engine.id, Game.black_engine_id == engine.id),
                Game.is_rated == True
            ).count()

            engine_data.append({
                'id': engine.id,
                'name': engine.name,
                'active': engine.active,
                'initial_elo': engine.initial_elo,
                'games_count': games_count,
                'created_at': engine.created_at,
            })

        return render_template(
            'engines_list.html',
            engines=engine_data,
            active_only=active_only,
            engine_type=engine_type,
        )

    @app.route('/engine/<name>/toggle', methods=['POST'])
    def engine_toggle(name):
        """Toggle engine active status."""
        engine = Engine.query.filter_by(name=name).first_or_404()
        engine.active = not engine.active
        db.session.commit()

        # Redirect back to engines list with current filters
        return redirect(request.referrer or url_for('engines_list'))

    @app.route('/engine/<name>')
    def engine_detail(name):
        """Engine detail page with H2H stats."""
        engine = Engine.query.filter_by(name=name).first_or_404()
        active_only = request.args.get('all') != '1'

        # Get all opponents
        opponent_query = Engine.query.filter(Engine.id != engine.id)
        if active_only:
            opponent_query = opponent_query.filter(Engine.active == True)
        opponents = {e.id: e for e in opponent_query.all()}

        # Time control buckets (in ms)
        time_buckets = [
            (0, 100, '0-100ms'),
            (101, 500, '101-500ms'),
            (501, 1000, '501ms-1s'),
            (1001, 2000, '1-2s'),
            (2001, 999999999, '2s+'),
        ]

        # Get H2H stats for each opponent
        h2h_stats = []
        for opp_id, opponent in opponents.items():
            # Games where engine was white against this opponent
            white_games = Game.query.filter(
                Game.white_engine_id == engine.id,
                Game.black_engine_id == opp_id,
                Game.is_rated == True
            ).all()

            # Games where engine was black against this opponent
            black_games = Game.query.filter(
                Game.white_engine_id == opp_id,
                Game.black_engine_id == engine.id,
                Game.is_rated == True
            ).all()

            if not white_games and not black_games:
                continue

            # Calculate stats as white
            white_wins = sum(1 for g in white_games if g.result == '1-0')
            white_losses = sum(1 for g in white_games if g.result == '0-1')
            white_draws = sum(1 for g in white_games if g.result == '1/2-1/2')

            # Calculate stats as black
            black_wins = sum(1 for g in black_games if g.result == '0-1')
            black_losses = sum(1 for g in black_games if g.result == '1-0')
            black_draws = sum(1 for g in black_games if g.result == '1/2-1/2')

            # Time control breakdown
            time_breakdown = []
            all_games = white_games + black_games
            for min_t, max_t, label in time_buckets:
                bucket_games = [g for g in all_games if g.time_per_move_ms and min_t <= g.time_per_move_ms <= max_t]
                if bucket_games:
                    wins = sum(1 for g in bucket_games if
                               (g.white_engine_id == engine.id and g.result == '1-0') or
                               (g.black_engine_id == engine.id and g.result == '0-1'))
                    losses = sum(1 for g in bucket_games if
                                 (g.white_engine_id == engine.id and g.result == '0-1') or
                                 (g.black_engine_id == engine.id and g.result == '1-0'))
                    draws = sum(1 for g in bucket_games if g.result == '1/2-1/2')
                    time_breakdown.append({
                        'label': label,
                        'wins': wins,
                        'losses': losses,
                        'draws': draws,
                        'total': len(bucket_games),
                    })

            total_games = len(white_games) + len(black_games)
            total_wins = white_wins + black_wins
            total_losses = white_losses + black_losses
            total_draws = white_draws + black_draws
            score = total_wins + total_draws * 0.5
            score_pct = (score / total_games * 100) if total_games > 0 else 0

            h2h_stats.append({
                'opponent_name': opponent.name,
                'opponent_active': opponent.active,
                'white_wins': white_wins,
                'white_losses': white_losses,
                'white_draws': white_draws,
                'white_total': len(white_games),
                'black_wins': black_wins,
                'black_losses': black_losses,
                'black_draws': black_draws,
                'black_total': len(black_games),
                'total_wins': total_wins,
                'total_losses': total_losses,
                'total_draws': total_draws,
                'total_games': total_games,
                'score_pct': score_pct,
                'time_breakdown': time_breakdown,
            })

        # Sort by total games descending
        h2h_stats.sort(key=lambda x: -x['total_games'])

        # Overall stats
        total_games = sum(s['total_games'] for s in h2h_stats)
        total_wins = sum(s['total_wins'] for s in h2h_stats)
        total_losses = sum(s['total_losses'] for s in h2h_stats)
        total_draws = sum(s['total_draws'] for s in h2h_stats)

        return render_template(
            'engine_detail.html',
            engine=engine,
            h2h_stats=h2h_stats,
            active_only=active_only,
            total_games=total_games,
            total_wins=total_wins,
            total_losses=total_losses,
            total_draws=total_draws,
        )

    @app.route('/health')
    def health():
        """Health check endpoint."""
        return {'status': 'ok'}

    @app.route('/epd-tests')
    def epd_tests_list():
        """List all EPD test runs with engine performance summary."""
        # Get all test runs, ordered by most recent first
        runs = EpdTestRun.query.order_by(EpdTestRun.created_at.desc()).all()

        if not runs:
            return render_template('epd_tests_list.html', runs=[], engines=[])

        # Get all engines that have EPD test results
        engine_ids = db.session.query(EpdTestResult.engine_id).distinct().all()
        engine_ids = [e[0] for e in engine_ids]
        engines = Engine.query.filter(Engine.id.in_(engine_ids)).order_by(Engine.name).all()

        # Build data for each run
        runs_data = []
        for run in runs:
            run_data = {
                'id': run.id,
                'epd_file': run.epd_file,
                'total_positions': run.total_positions,
                'timeout_seconds': run.timeout_seconds,
                'hostname': run.hostname,
                'created_at': run.created_at,
                'engine_results': {}
            }

            # Get results for each engine from this run
            for engine in engines:
                results = EpdTestResult.query.filter(
                    EpdTestResult.run_id == run.id,
                    EpdTestResult.engine_id == engine.id
                ).all()

                if results:
                    solved = sum(1 for r in results if r.solved)
                    total = len(results)
                    pct = 100 * solved / total if total > 0 else 0
                    run_data['engine_results'][engine.name] = {
                        'solved': solved,
                        'total': total,
                        'pct': pct
                    }

            runs_data.append(run_data)

        return render_template(
            'epd_tests_list.html',
            runs=runs_data,
            engines=engines
        )

    @app.route('/epd-tests/<path:epd_file>')
    def epd_test_detail(epd_file):
        """Show detailed results for a specific EPD file."""
        # Get specific run by ID, or latest run for this file
        run_id = request.args.get('run_id', type=int)
        if run_id:
            run = EpdTestRun.query.get(run_id)
        else:
            run = EpdTestRun.query.filter(
                EpdTestRun.epd_file == epd_file
            ).order_by(EpdTestRun.created_at.desc()).first()

        if not run:
            return render_template('epd_test_detail.html', epd_file=epd_file, run=None, positions=[], engines=[],
                                   page=1, total_pages=1, per_page=50, total_positions=0)

        # Pagination parameters
        page = request.args.get('page', 1, type=int)
        per_page = request.args.get('per_page', 50, type=int)
        per_page = min(per_page, 200)  # Cap at 200

        # Get all engines that have results for this run
        engine_ids = db.session.query(EpdTestResult.engine_id).filter(
            EpdTestResult.run_id == run.id
        ).distinct().all()
        engine_ids = [e[0] for e in engine_ids]
        engines = Engine.query.filter(Engine.id.in_(engine_ids)).order_by(Engine.name).all()
        engine_map = {e.id: e.name for e in engines}

        # Fetch ALL results for this run in a single query
        all_results = EpdTestResult.query.filter(
            EpdTestResult.run_id == run.id
        ).all()

        # Group results by position_index for efficient lookup
        results_by_position = {}
        for result in all_results:
            if result.position_index not in results_by_position:
                results_by_position[result.position_index] = {
                    'position_id': result.position_id,
                    'fen': result.fen,
                    'test_type': result.test_type,
                    'expected_moves': result.expected_moves,
                    'engine_results': {}
                }
            engine_name = engine_map.get(result.engine_id)
            if engine_name:
                results_by_position[result.position_index]['engine_results'][engine_name] = {
                    'solved': result.solved,
                    'move_found': result.move_found,
                    'solve_time_ms': result.solve_time_ms,
                    'final_depth': result.final_depth,
                    'timed_out': result.timed_out,
                    'score_cp': result.score_cp,
                    'score_mate': result.score_mate,
                    'points_earned': result.points_earned
                }

        # Calculate engine stats from the already-fetched results
        engine_stats = {e.name: {'solved': 0, 'total': 0, 'solve_times': [], 'points_earned': 0, 'points_max': 0} for e in engines}
        for result in all_results:
            engine_name = engine_map.get(result.engine_id)
            if engine_name:
                engine_stats[engine_name]['total'] += 1
                if result.points_earned is not None:
                    engine_stats[engine_name]['points_earned'] += result.points_earned
                    engine_stats[engine_name]['points_max'] += 10  # STS max is always 10
                if result.solved:
                    engine_stats[engine_name]['solved'] += 1
                    if result.solve_time_ms:
                        engine_stats[engine_name]['solve_times'].append(result.solve_time_ms)

        # Finalize engine stats
        for name in engine_stats:
            stats = engine_stats[name]
            solve_times = stats.pop('solve_times')
            stats['pct'] = 100 * stats['solved'] / stats['total'] if stats['total'] > 0 else 0
            stats['avg_time'] = sum(solve_times) / len(solve_times) / 1000 if solve_times else 0
            stats['points_pct'] = 100 * stats['points_earned'] / stats['points_max'] if stats['points_max'] > 0 else 0

        # Get sorted position indices and apply pagination
        sorted_indices = sorted(results_by_position.keys())
        total_positions = len(sorted_indices)
        total_pages = (total_positions + per_page - 1) // per_page
        page = max(1, min(page, total_pages)) if total_pages > 0 else 1

        start_idx = (page - 1) * per_page
        end_idx = start_idx + per_page
        paginated_indices = sorted_indices[start_idx:end_idx]

        # Build position data for the current page only
        positions = []
        for pos_index in paginated_indices:
            pos_info = results_by_position[pos_index]
            fen = pos_info['fen']
            pos_data = {
                'index': pos_index,
                'id': pos_info['position_id'],
                'fen': fen,
                'fen_short': fen[:40] + '...' if len(fen) > 40 else fen,
                'test_type': pos_info['test_type'],
                'expected_moves': pos_info['expected_moves'],
                'engine_results': pos_info['engine_results'],
                'failure_count': sum(1 for r in pos_info['engine_results'].values() if not r['solved'])
            }
            positions.append(pos_data)

        return render_template(
            'epd_test_detail.html',
            epd_file=epd_file,
            run=run,
            positions=positions,
            engines=engines,
            engine_stats=engine_stats,
            page=page,
            total_pages=total_pages,
            per_page=per_page,
            total_positions=total_positions
        )

    @app.route('/spsa')
    def spsa_dashboard():
        """SPSA parameter tuning dashboard with progression graphs."""
        import tomllib
        from pathlib import Path

        # Load SPSA config for ref_ratio and effective_iteration_offset
        config_path = Path(__file__).parent.parent / 'compete' / 'spsa' / 'config.toml'
        ref_ratio = 1.0  # Default
        effective_iteration_offset = 0  # Default
        try:
            with open(config_path, 'rb') as f:
                spsa_config = tomllib.load(f)
                ref_ratio = spsa_config.get('reference', {}).get('ratio', 1.0)
                effective_iteration_offset = spsa_config.get('spsa', {}).get('effective_iteration_offset', 0)
        except Exception:
            pass  # Use default if config not found

        # Get all completed iterations ordered by iteration number
        iterations = SpsaIteration.query.filter(
            SpsaIteration.status == 'complete'
        ).order_by(SpsaIteration.iteration_number.asc()).all()

        if not iterations:
            return render_template('spsa.html', iterations=[], params_data={}, elo_data=[], ref_ratio=ref_ratio)

        # Get parameter names from ALL iterations (union of all params seen)
        # This handles cases where new params are added mid-tuning
        all_param_names = set()
        for it in iterations:
            if it.base_parameters:
                all_param_names.update(it.base_parameters.keys())
        param_names = sorted(all_param_names)

        # Build parameter progression data for Chart.js
        params_data = {name: [] for name in param_names}
        iteration_numbers = []
        elo_data = []
        ref_elo_data = []

        for it in iterations:
            iteration_numbers.append(it.iteration_number)
            elo_data.append(float(it.elo_diff) if it.elo_diff else 0)
            ref_elo_data.append(float(it.ref_elo_estimate) if it.ref_elo_estimate else None)
            if it.base_parameters:
                for name in param_names:
                    # Use None for params that don't exist in this iteration
                    # Chart.js will skip null points, showing gaps for new params
                    val = it.base_parameters.get(name)
                    params_data[name].append(val)

        # Get current/latest values for display
        latest = iterations[-1] if iterations else None
        current_params = {}
        if latest and latest.base_parameters:
            current_params = latest.base_parameters

        # Calculate total change from first appearance to last iteration
        # For params added mid-tuning, find their first iteration
        param_changes = {}
        if len(iterations) >= 1 and iterations[-1].base_parameters:
            last_params = iterations[-1].base_parameters
            for name in param_names:
                # Find first iteration where this param exists
                first_val = None
                for it in iterations:
                    if it.base_parameters and name in it.base_parameters:
                        first_val = it.base_parameters[name]
                        break
                if first_val is None:
                    first_val = 0

                last_val = last_params.get(name, 0)
                if first_val != 0:
                    pct_change = ((last_val - first_val) / abs(first_val)) * 100
                else:
                    pct_change = 0
                param_changes[name] = {
                    'first': first_val,
                    'last': last_val,
                    'change': last_val - first_val,
                    'pct_change': pct_change
                }

        # Get in-progress iteration if any (two-phase: pending, in_progress, building, ref_pending)
        in_progress = SpsaIteration.query.filter(
            SpsaIteration.status.in_(['pending', 'in_progress', 'building', 'ref_pending'])
        ).order_by(SpsaIteration.iteration_number.desc()).first()

        # Calculate rolling stability (standard deviation over window) for convergence chart
        # Sample every Nth point to reduce computation - chart doesn't need every point
        window_size = 10
        sample_rate = max(1, len(iterations) // 100)  # At most ~100 points
        stability_data = {name: [] for name in param_names}
        stability_iterations = []

        for name in param_names:
            values = params_data[name]
            param_stability = []
            for i in range(0, len(values), sample_rate):
                if i < window_size - 1:
                    window = [v for v in values[:i+1] if v is not None]
                else:
                    window = [v for v in values[i-window_size+1:i+1] if v is not None]

                if len(window) > 1:
                    mean = sum(window) / len(window)
                    variance = sum((x - mean) ** 2 for x in window) / len(window)
                    std_dev = variance ** 0.5
                    param_range = param_changes[name]['first'] if name in param_changes and param_changes[name]['first'] != 0 else 1
                    param_stability.append(std_dev / abs(param_range) * 100)
                else:
                    param_stability.append(0 if len(window) == 1 else None)
            stability_data[name] = param_stability

        # Build sampled iteration numbers for stability chart x-axis
        stability_iterations = [iteration_numbers[i] for i in range(0, len(iteration_numbers), sample_rate)]

        # Get the latest ref_elo for display (only from iteration 110+)
        ref_min_iteration = 110
        latest_ref_elo = None
        for i in range(len(ref_elo_data) - 1, -1, -1):
            if ref_elo_data[i] is not None and iteration_numbers[i] >= ref_min_iteration:
                latest_ref_elo = ref_elo_data[i]
                break

        # Build filtered data for ref_elo chart (only iterations with reference data)
        # Uses ref_min_iteration defined above to exclude buggy early data
        ref_iteration_numbers = []
        ref_elo_filtered = []
        for i, elo in enumerate(ref_elo_data):
            if elo is not None and iteration_numbers[i] >= ref_min_iteration:
                ref_iteration_numbers.append(iteration_numbers[i])
                ref_elo_filtered.append(elo)

        # Get latest effective iteration from completed iterations
        latest_effective_iteration = None
        if iterations:
            latest_effective_iteration = iterations[-1].effective_iteration

        return render_template(
            'spsa.html',
            iterations=iterations,
            param_names=param_names,
            params_data=params_data,
            iteration_numbers=iteration_numbers,
            elo_data=elo_data,
            ref_elo_data=ref_elo_filtered,
            ref_iteration_numbers=ref_iteration_numbers,
            latest_ref_elo=latest_ref_elo,
            current_params=current_params,
            param_changes=param_changes,
            in_progress=in_progress,
            total_iterations=len(iterations),
            stability_data=stability_data,
            stability_iterations=stability_iterations,
            ref_ratio=ref_ratio,
            effective_iteration_offset=effective_iteration_offset,
            latest_effective_iteration=latest_effective_iteration
        )
