"""
Flask routes for the competition dashboard.
"""

import os
from flask import render_template, request, redirect, url_for, jsonify
from web.queries import (
    get_dashboard_data, get_unique_hostnames, get_time_range,
    clear_elo_cache, recalculate_all_and_store
)
from datetime import datetime, timedelta
from web.models import (
    Cup, CupRound, CupMatch, Engine, Game, EpdTestRun, EpdTestResult,
    DatagenJob, H2hMatch,
    SpsaIteration, SpsaParam, SpsaRun, SpsaWorker, SpsaWorkerHeartbeat
)
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
        sort_column = request.args.get('sort') or 'bayes'
        sort_direction = request.args.get('sort_dir') or 'desc'

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

        # Recalculate all ratings (Elo, BayesElo, Ordo) on every refresh
        recalculate_all_and_store(min_time, max_time, hostname, engine_type)

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

        # Get all runs for the dropdown (exclude Default template)
        all_runs = SpsaRun.query.filter(SpsaRun.name != 'Default').order_by(SpsaRun.id.desc()).all()

        # Get selected run from query param, default to active run
        selected_run_id = request.args.get('run', type=int)
        if selected_run_id:
            selected_run = SpsaRun.query.get(selected_run_id)
        else:
            # Default to active run, or most recent if none active
            selected_run = SpsaRun.query.filter_by(is_active=True).first()
            if not selected_run and all_runs:
                selected_run = all_runs[0]  # Most recent

        # If no runs exist yet, show empty state
        if not selected_run:
            return render_template('spsa.html', iterations=[], params_data={}, elo_data=[],
                                   ref_ratio=1.0, all_runs=[], selected_run=None)

        # Read ref_ratio from selected run
        ref_ratio = selected_run.ref_ratio

        # Calculate effective_iteration_offset from database (for selected run)
        # offset = iteration_number - effective_iteration (from latest iteration with effective_iteration set)
        effective_iteration_offset = 0
        latest_with_eff = SpsaIteration.query.filter(
            SpsaIteration.run_id == selected_run.id,
            SpsaIteration.effective_iteration.isnot(None)
        ).order_by(SpsaIteration.iteration_number.desc()).first()
        if latest_with_eff and latest_with_eff.effective_iteration:
            effective_iteration_offset = latest_with_eff.iteration_number - latest_with_eff.effective_iteration

        # Load parameter bounds from database for the selected run
        param_bounds = {}  # {param_name: {'min': X, 'max': Y}}
        spsa_params = SpsaParam.query.filter_by(run_id=selected_run.id).all()
        active_param_names = set()
        param_group_map = {}  # {param_name: group_name}
        for p in spsa_params:
            param_bounds[p.name] = {'min': p.min_value, 'max': p.max_value}
            param_group_map[p.name] = p.group or 'ungrouped'
            if selected_run.active_groups is None or p.group in selected_run.active_groups:
                active_param_names.add(p.name)

        # Get all completed iterations for selected run, ordered by iteration number
        iterations = SpsaIteration.query.filter(
            SpsaIteration.run_id == selected_run.id,
            SpsaIteration.status == 'complete'
        ).order_by(SpsaIteration.iteration_number.asc()).all()

        if not iterations:
            # Still need to check for in-progress iteration even with no completed ones
            in_progress = SpsaIteration.query.filter(
                SpsaIteration.run_id == selected_run.id,
                SpsaIteration.status.in_(['pending', 'in_progress', 'building', 'ref_pending'])
            ).order_by(SpsaIteration.iteration_number.desc()).first()
            active_cutoff = datetime.utcnow() - timedelta(minutes=5)
            active_worker_count = SpsaWorker.query.filter(
                SpsaWorker.last_seen_at >= active_cutoff
            ).count()
            return render_template('spsa.html', iterations=[], params_data={}, elo_data=[],
                                   ref_ratio=ref_ratio, all_runs=all_runs, selected_run=selected_run,
                                   in_progress=in_progress, active_param_names=active_param_names,
                                   param_group_map=param_group_map,
                                   active_worker_count=active_worker_count,
                                   effective_concurrency=0,
                                   update_data={}, avg_update_data=[], update_iterations=[],
                                   convergence_score=0, recent_avg_update=0, last_update={},
                                   workers_per_iteration=[],
                                   )

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
        ref_game_results = []  # W/L/D for each iteration's reference games

        for it in iterations:
            iteration_numbers.append(it.iteration_number)
            elo_data.append(float(it.elo_diff) if it.elo_diff else 0)
            ref_elo_data.append(float(it.ref_elo_estimate) if it.ref_elo_estimate else None)
            # Calculate duration in minutes
            duration_mins = None
            if it.created_at and it.completed_at:
                duration_mins = (it.completed_at - it.created_at).total_seconds() / 60
            ref_game_results.append({
                'wins': it.ref_wins or 0,
                'losses': it.ref_losses or 0,
                'draws': it.ref_draws or 0,
                'duration_mins': duration_mins
            })
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

        # Get in-progress iteration for selected run (two-phase: pending, in_progress, building, ref_pending)
        in_progress = SpsaIteration.query.filter(
            SpsaIteration.run_id == selected_run.id,
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

        # Compute per-iteration parameter update magnitude (% of value)
        update_data = {name: [] for name in param_names}
        avg_update_data = []
        update_iterations = []

        for i in range(0, len(iterations), sample_rate):
            if i == 0:
                for name in param_names:
                    update_data[name].append(0)
                avg_update_data.append(0)
                update_iterations.append(iteration_numbers[i])
                continue

            update_iterations.append(iteration_numbers[i])
            iter_updates = []
            for name in param_names:
                cur = params_data[name][i] if i < len(params_data[name]) else None
                prev = params_data[name][i - 1] if (i - 1) < len(params_data[name]) else None
                if cur is not None and prev is not None and prev != 0:
                    pct = abs(cur - prev) / abs(prev) * 100
                else:
                    pct = 0
                update_data[name].append(pct)
                if name in active_param_names:
                    iter_updates.append(pct)
            avg_update_data.append(sum(iter_updates) / len(iter_updates) if iter_updates else 0)

        # Convergence score: compare recent vs early average update magnitude
        convergence_score = 0
        recent_avg_update = 0
        if len(avg_update_data) >= 2:
            early_n = min(10, len(avg_update_data) // 2)
            late_n = min(10, len(avg_update_data) // 2)
            early_avg = sum(avg_update_data[1:1 + early_n]) / early_n if early_n > 0 else 0
            recent_vals = avg_update_data[-late_n:]
            recent_avg = sum(recent_vals) / len(recent_vals) if recent_vals else 0
            recent_avg_update = recent_avg
            if early_avg > 0:
                convergence_score = max(0, min(100, (1 - recent_avg / early_avg) * 100))

        # Last iteration's update for each parameter
        last_update = {}
        if len(iterations) >= 2:
            last_params = iterations[-1].base_parameters or {}
            prev_params = iterations[-2].base_parameters or {}
            for name in param_names:
                cur = last_params.get(name)
                prev = prev_params.get(name)
                if cur is not None and prev is not None:
                    delta = cur - prev
                    pct = abs(delta) / abs(prev) * 100 if prev != 0 else 0
                    last_update[name] = {'abs': delta, 'pct': pct}
                else:
                    last_update[name] = {'abs': 0, 'pct': 0}
        else:
            for name in param_names:
                last_update[name] = {'abs': 0, 'pct': 0}

        # Get the latest ref_elo for display
        # Run 1 had buggy data before iteration 110, so filter those out
        ref_min_iteration = 110 if selected_run.id == 1 else 0
        latest_ref_elo = None
        for i in range(len(ref_elo_data) - 1, -1, -1):
            if ref_elo_data[i] is not None and iteration_numbers[i] >= ref_min_iteration:
                latest_ref_elo = ref_elo_data[i]
                break

        # Calculate rolling average Elo over last ~X reference games
        # n = ceil(target_games / ref_games_per_iteration)
        import math
        rolling_elo_target_games = 5000  # Default: average over last ~5000 ref games
        rolling_elo_avg = None
        rolling_elo_n = 0
        ref_games_per_iter = latest.ref_target_games if latest and latest.ref_target_games else 75
        if ref_games_per_iter > 0:
            rolling_elo_n = math.ceil(rolling_elo_target_games / ref_games_per_iter)
            # Get last n iterations with valid ref_elo
            valid_ref_elos = [(iteration_numbers[i], ref_elo_data[i])
                             for i in range(len(ref_elo_data))
                             if ref_elo_data[i] is not None and iteration_numbers[i] >= ref_min_iteration]
            if valid_ref_elos:
                last_n = valid_ref_elos[-rolling_elo_n:] if len(valid_ref_elos) >= rolling_elo_n else valid_ref_elos
                if last_n:
                    rolling_elo_avg = sum(elo for _, elo in last_n) / len(last_n)
                    rolling_elo_n = len(last_n)  # Actual number used

        # Compute rolling avg changes from different lookback periods
        rolling_elo_changes = {}
        if rolling_elo_avg is not None and valid_ref_elos:
            for lookback in [1, 10, 50]:
                if len(valid_ref_elos) >= rolling_elo_n + lookback:
                    past_elos = valid_ref_elos[-(rolling_elo_n + lookback):-lookback]
                    past_avg = sum(elo for _, elo in past_elos) / len(past_elos)
                    rolling_elo_changes[lookback] = rolling_elo_avg - past_avg
                else:
                    rolling_elo_changes[lookback] = None

        # Build filtered data for ref_elo chart (only iterations with reference data)
        # Uses ref_min_iteration to exclude buggy early data (run 1 only)
        ref_iteration_numbers = []
        ref_elo_filtered = []
        ref_results_filtered = []
        for i, elo in enumerate(ref_elo_data):
            if elo is not None and iteration_numbers[i] >= ref_min_iteration:
                ref_iteration_numbers.append(iteration_numbers[i])
                ref_elo_filtered.append(elo)
                ref_results_filtered.append(ref_game_results[i])

        # Count distinct workers per iteration (for overlay on ref Elo chart)
        iter_id_map = {it.iteration_number: it.id for it in iterations}
        workers_per_iteration = []
        if ref_iteration_numbers:
            ref_iter_ids = [iter_id_map[n] for n in ref_iteration_numbers if n in iter_id_map]
            if ref_iter_ids:
                worker_counts_raw = db.session.execute(db.text("""
                    SELECT iteration_id, COUNT(DISTINCT worker_id) as worker_count
                    FROM spsa_worker_heartbeats
                    WHERE iteration_id = ANY(:ids)
                    GROUP BY iteration_id
                """), {'ids': ref_iter_ids}).fetchall()
                worker_count_map = {row.iteration_id: row.worker_count for row in worker_counts_raw}
                workers_per_iteration = [
                    worker_count_map.get(iter_id_map.get(n, -1), 0)
                    for n in ref_iteration_numbers
                ]

        # Get latest effective iteration from completed iterations
        latest_effective_iteration = None
        if iterations:
            latest_effective_iteration = iterations[-1].effective_iteration

        # Count active workers (seen in last 5 minutes) and effective concurrency
        active_cutoff = datetime.utcnow() - timedelta(minutes=5)
        active_workers_list = SpsaWorker.query.filter(
            SpsaWorker.last_seen_at >= active_cutoff
        ).all()
        active_worker_count = len(active_workers_list)

        # Compute effective concurrent games from active workers
        effective_concurrency = 0.0
        if active_workers_list:
            from sqlalchemy import func as sa_func
            aw_ids = [w.id for w in active_workers_list]
            latest_hb_subq = db.session.query(
                SpsaWorkerHeartbeat.worker_id,
                sa_func.max(SpsaWorkerHeartbeat.id).label('max_id')
            ).filter(
                SpsaWorkerHeartbeat.worker_id.in_(aw_ids)
            ).group_by(SpsaWorkerHeartbeat.worker_id).subquery()

            latest_hbs = db.session.query(
                SpsaWorkerHeartbeat.concurrency,
                SpsaWorkerHeartbeat.timemult
            ).join(
                latest_hb_subq,
                SpsaWorkerHeartbeat.id == latest_hb_subq.c.max_id
            ).all()
            for hb in latest_hbs:
                if hb.concurrency and hb.timemult and hb.timemult > 0:
                    effective_concurrency += hb.concurrency / hb.timemult
                elif hb.concurrency:
                    effective_concurrency += hb.concurrency
        effective_concurrency = round(effective_concurrency, 1)

        return render_template(
            'spsa.html',
            iterations=iterations,
            param_names=param_names,
            params_data=params_data,
            iteration_numbers=iteration_numbers,
            elo_data=elo_data,
            ref_elo_data=ref_elo_filtered,
            ref_iteration_numbers=ref_iteration_numbers,
            ref_game_results=ref_results_filtered,
            workers_per_iteration=workers_per_iteration,
            latest_ref_elo=latest_ref_elo,
            rolling_elo_avg=rolling_elo_avg,
            rolling_elo_target_games=rolling_elo_target_games,
            rolling_elo_n=rolling_elo_n,
            current_params=current_params,
            param_changes=param_changes,
            in_progress=in_progress,
            total_iterations=len(iterations),
            stability_data=stability_data,
            stability_iterations=stability_iterations,
            ref_ratio=ref_ratio,
            effective_iteration_offset=effective_iteration_offset,
            latest_effective_iteration=latest_effective_iteration,
            param_bounds=param_bounds,
            rolling_elo_changes=rolling_elo_changes,
            all_runs=all_runs,
            selected_run=selected_run,
            active_param_names=active_param_names,
            param_group_map=param_group_map,
            active_worker_count=active_worker_count,
            effective_concurrency=effective_concurrency,
            update_data=update_data,
            avg_update_data=avg_update_data,
            update_iterations=update_iterations,
            convergence_score=convergence_score,
            recent_avg_update=recent_avg_update,
            last_update=last_update,
        )

    @app.route('/elo-stats')
    def elo_stats():
        """Elo statistics calculator - probability of detecting Elo differences."""
        return render_template('elo_stats.html')

    @app.route('/api/spsa/status')
    def spsa_status():
        """API endpoint for SPSA iteration status polling.

        Returns JSON with current iteration info for live dashboard updates.
        All database access happens server-side - no credentials exposed to frontend.
        Accepts optional 'run' query parameter to filter by run_id.
        """
        # Active worker count and effective concurrency (seen in last 5 minutes)
        active_cutoff = datetime.utcnow() - timedelta(minutes=5)
        active_workers_list = SpsaWorker.query.filter(
            SpsaWorker.last_seen_at >= active_cutoff
        ).all()
        active_workers = len(active_workers_list)

        effective_concurrency = 0.0
        if active_workers_list:
            from sqlalchemy import func as sa_func
            aw_ids = [w.id for w in active_workers_list]
            latest_hb_subq = db.session.query(
                SpsaWorkerHeartbeat.worker_id,
                sa_func.max(SpsaWorkerHeartbeat.id).label('max_id')
            ).filter(
                SpsaWorkerHeartbeat.worker_id.in_(aw_ids)
            ).group_by(SpsaWorkerHeartbeat.worker_id).subquery()

            latest_hbs = db.session.query(
                SpsaWorkerHeartbeat.concurrency,
                SpsaWorkerHeartbeat.timemult
            ).join(
                latest_hb_subq,
                SpsaWorkerHeartbeat.id == latest_hb_subq.c.max_id
            ).all()
            for hb in latest_hbs:
                if hb.concurrency and hb.timemult and hb.timemult > 0:
                    effective_concurrency += hb.concurrency / hb.timemult
                elif hb.concurrency:
                    effective_concurrency += hb.concurrency
        effective_concurrency = round(effective_concurrency, 1)

        # Get run_id from query param (optional)
        run_id = request.args.get('run', type=int)

        # Build base query filter
        base_filter = []
        if run_id:
            base_filter.append(SpsaIteration.run_id == run_id)

        # Find the current in-progress iteration, or the latest completed one
        in_progress_query = SpsaIteration.query.filter(
            *base_filter,
            SpsaIteration.status.in_(['pending', 'in_progress', 'building', 'ref_pending'])
        ).order_by(SpsaIteration.iteration_number.desc())
        in_progress = in_progress_query.first()

        if in_progress:
            return jsonify({
                'iteration_number': in_progress.iteration_number,
                'status': in_progress.status,
                'games_played': in_progress.games_played,
                'target_games': in_progress.target_games,
                'ref_games_played': in_progress.ref_games_played,
                'ref_target_games': in_progress.ref_target_games,
                'active_workers': active_workers,
                'effective_concurrency': effective_concurrency
            })

        # No in-progress iteration, return the latest completed one
        latest_query = SpsaIteration.query.filter(
            *base_filter,
            SpsaIteration.status == 'complete'
        ).order_by(SpsaIteration.iteration_number.desc())
        latest = latest_query.first()

        if latest:
            return jsonify({
                'iteration_number': latest.iteration_number,
                'status': 'complete',
                'games_played': latest.games_played,
                'target_games': latest.target_games,
                'ref_games_played': latest.ref_games_played,
                'ref_target_games': latest.ref_target_games,
                'active_workers': active_workers,
                'effective_concurrency': effective_concurrency
            })

        # No iterations at all
        return jsonify({
            'iteration_number': None,
            'status': None,
            'games_played': 0,
            'target_games': 0,
            'ref_games_played': 0,
            'ref_target_games': 0,
            'active_workers': active_workers,
            'effective_concurrency': effective_concurrency
        })

    # =========================================================================
    # SPSA Worker API Endpoints (for remote Docker workers)
    # =========================================================================

    def verify_worker_api_key():
        """Verify the API key from request header. Returns True if valid."""
        api_key = request.headers.get('X-API-Key')
        expected_key = os.environ.get('SPSA_WORKER_API_KEY')
        if not expected_key:
            # No key configured = API disabled
            return False
        return api_key == expected_key

    @app.route('/api/spsa/work')
    def spsa_get_work():
        """
        Poll for pending SPSA work.

        Workers call this to get the next iteration that needs games.
        Returns iteration data including parameters (workers build engines locally).

        Two-phase flow:
        - status='ref_pending': Reference phase takes priority (base vs Stockfish)
        - status='pending'/'in_progress': SPSA phase (plus vs minus)

        Returns JSON with iteration data or empty object if no work available.
        """
        if not verify_worker_api_key():
            return jsonify({'error': 'unauthorized'}), 401

        # Get worker hostname for logging
        worker_host = request.headers.get('X-Worker-Host', 'unknown')

        # Only serve work from the active run
        active_run = SpsaRun.query.filter_by(is_active=True).first()
        if not active_run:
            return jsonify({})  # No active run

        # First check for ref_pending iterations (Phase 2 takes priority)
        iteration = SpsaIteration.query.filter(
            SpsaIteration.run_id == active_run.id,
            SpsaIteration.status == 'ref_pending',
            SpsaIteration.ref_games_played < SpsaIteration.ref_target_games
        ).order_by(SpsaIteration.iteration_number.desc()).first()

        if iteration:
            return jsonify({
                'id': iteration.id,
                'run_id': iteration.run_id,
                'iteration_number': iteration.iteration_number,
                'phase': 'ref',
                'base_parameters': iteration.base_parameters,
                'timelow_ms': iteration.timelow_ms,
                'timehigh_ms': iteration.timehigh_ms,
                'target_games': iteration.ref_target_games,
                'games_played': iteration.ref_games_played,
                'tc_moves': iteration.tc_moves,
                'tc_base_seconds': iteration.tc_base_seconds,
                'tc_increment': iteration.tc_increment,
            })

        # Check for SPSA phase iterations (pending or in_progress)
        iteration = SpsaIteration.query.filter(
            SpsaIteration.run_id == active_run.id,
            SpsaIteration.status.in_(['pending', 'in_progress']),
            SpsaIteration.games_played < SpsaIteration.target_games
        ).order_by(SpsaIteration.iteration_number.desc()).first()

        if not iteration:
            return jsonify({})  # No work available

        # Mark as in_progress if pending
        if iteration.status == 'pending':
            iteration.status = 'in_progress'
            db.session.commit()

        return jsonify({
            'id': iteration.id,
            'run_id': iteration.run_id,
            'iteration_number': iteration.iteration_number,
            'phase': 'spsa',
            'plus_parameters': iteration.plus_parameters,
            'minus_parameters': iteration.minus_parameters,
            'base_parameters': iteration.base_parameters,
            'timelow_ms': iteration.timelow_ms,
            'timehigh_ms': iteration.timehigh_ms,
            'target_games': iteration.target_games,
            'games_played': iteration.games_played,
            'tc_moves': iteration.tc_moves,
            'tc_base_seconds': iteration.tc_base_seconds,
            'tc_increment': iteration.tc_increment,
        })

    def record_worker_activity(worker_name: str, iteration_id: int, phase: str,
                                games: int, avg_nps: int = None, timemult: float = None,
                                concurrency: int = None):
        """
        Record worker activity in the tracking tables.

        Updates the worker summary and creates a heartbeat record.
        """
        if not worker_name:
            return  # Skip if no worker name provided

        try:
            # Get or create worker record
            worker = SpsaWorker.query.filter_by(worker_name=worker_name).first()
            if not worker:
                worker = SpsaWorker(
                    worker_name=worker_name,
                    first_seen_at=datetime.utcnow()
                )
                db.session.add(worker)
                db.session.flush()  # Get the ID

            # Update worker summary
            worker.last_iteration_id = iteration_id
            worker.last_phase = phase
            worker.total_games += games
            if phase == 'spsa':
                worker.total_spsa_games += games
            else:
                worker.total_ref_games += games
            worker.last_seen_at = datetime.utcnow()

            # Update rolling average NPS (simple moving average)
            if avg_nps:
                if worker.avg_nps:
                    # Weight new reading at 20%
                    worker.avg_nps = int(worker.avg_nps * 0.8 + avg_nps * 0.2)
                else:
                    worker.avg_nps = avg_nps

            # Create heartbeat record
            heartbeat = SpsaWorkerHeartbeat(
                worker_id=worker.id,
                iteration_id=iteration_id,
                phase=phase,
                games_reported=games,
                avg_nps=avg_nps,
                timemult=timemult,
                concurrency=concurrency
            )
            db.session.add(heartbeat)

            # Prune heartbeats older than 3 days (retain for full run duration)
            cutoff_time = datetime.utcnow() - timedelta(hours=72)
            SpsaWorkerHeartbeat.query.filter(
                SpsaWorkerHeartbeat.created_at < cutoff_time
            ).delete(synchronize_session=False)

            db.session.commit()
        except Exception as e:
            # Don't fail the main request if tracking fails
            db.session.rollback()
            print(f"Warning: Failed to record worker activity: {e}")

    @app.route('/api/spsa/iterations/<int:iteration_id>/results', methods=['POST'])
    def spsa_report_results(iteration_id):
        """
        Report SPSA game results (plus vs minus).

        Atomically increments game counters to avoid race conditions
        when multiple workers report simultaneously.

        Expects JSON body:
        {
            "games": N,
            "plus_wins": N,
            "minus_wins": N,
            "draws": N,
            "worker_name": "optional-hostname",
            "avg_nps": optional-integer
        }
        """
        if not verify_worker_api_key():
            return jsonify({'error': 'unauthorized'}), 401

        data = request.get_json()
        if not data:
            return jsonify({'error': 'missing JSON body'}), 400

        # Validate required fields
        required = ['games', 'plus_wins', 'minus_wins', 'draws']
        for field in required:
            if field not in data:
                return jsonify({'error': f'missing field: {field}'}), 400
            if not isinstance(data[field], int) or data[field] < 0:
                return jsonify({'error': f'invalid value for {field}'}), 400

        # Reject results for completed iterations
        iter_check = db.session.execute(db.text(
            "SELECT status FROM spsa_iterations WHERE id = :id"
        ), {'id': iteration_id}).fetchone()
        if not iter_check:
            return jsonify({'error': 'iteration not found'}), 404
        if iter_check.status == 'complete':
            return jsonify({'status': 'ignored', 'reason': 'iteration already complete', 'remaining': 0})

        # Atomic increment using SQL
        try:
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
                    'games': data['games'],
                    'plus_wins': data['plus_wins'],
                    'minus_wins': data['minus_wins'],
                    'draws': data['draws'],
                    'id': iteration_id
                }
            )
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500

        # Read back current state to return remaining count
        row = db.session.execute(db.text(
            "SELECT games_played, target_games FROM spsa_iterations WHERE id = :id"
        ), {'id': iteration_id}).fetchone()
        remaining = max(0, row.target_games - row.games_played) if row else 0

        # Record worker activity (optional, won't fail the request)
        worker_name = data.get('worker_name') or request.headers.get('X-Worker-Host')
        avg_nps = data.get('avg_nps')
        timemult = data.get('timemult')
        concurrency = data.get('concurrency')
        record_worker_activity(worker_name, iteration_id, 'spsa', data['games'], avg_nps, timemult, concurrency)

        return jsonify({'status': 'ok', 'remaining': remaining})

    @app.route('/api/spsa/iterations/<int:iteration_id>/ref-results', methods=['POST'])
    def spsa_report_ref_results(iteration_id):
        """
        Report reference game results (base vs Stockfish).

        Atomically increments game counters to avoid race conditions
        when multiple workers report simultaneously.

        Expects JSON body:
        {
            "games": N,
            "wins": N,
            "losses": N,
            "draws": N,
            "worker_name": "optional-hostname",
            "avg_nps": optional-integer
        }
        """
        if not verify_worker_api_key():
            return jsonify({'error': 'unauthorized'}), 401

        data = request.get_json()
        if not data:
            return jsonify({'error': 'missing JSON body'}), 400

        # Validate required fields
        required = ['games', 'wins', 'losses', 'draws']
        for field in required:
            if field not in data:
                return jsonify({'error': f'missing field: {field}'}), 400
            if not isinstance(data[field], int) or data[field] < 0:
                return jsonify({'error': f'invalid value for {field}'}), 400

        # Reject results for completed iterations
        iter_check = db.session.execute(db.text(
            "SELECT status FROM spsa_iterations WHERE id = :id"
        ), {'id': iteration_id}).fetchone()
        if not iter_check:
            return jsonify({'error': 'iteration not found'}), 404
        if iter_check.status == 'complete':
            return jsonify({'status': 'ignored', 'reason': 'iteration already complete', 'remaining': 0})

        # Atomic increment using SQL
        try:
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
                    'games': data['games'],
                    'wins': data['wins'],
                    'losses': data['losses'],
                    'draws': data['draws'],
                    'id': iteration_id
                }
            )
            db.session.commit()
        except Exception as e:
            db.session.rollback()
            return jsonify({'error': str(e)}), 500

        # Read back current state to return remaining count
        row = db.session.execute(db.text(
            "SELECT ref_games_played, ref_target_games FROM spsa_iterations WHERE id = :id"
        ), {'id': iteration_id}).fetchone()
        remaining = max(0, row.ref_target_games - row.ref_games_played) if row else 0

        # Record worker activity (optional, won't fail the request)
        worker_name = data.get('worker_name') or request.headers.get('X-Worker-Host')
        avg_nps = data.get('avg_nps')
        timemult = data.get('timemult')
        concurrency = data.get('concurrency')
        record_worker_activity(worker_name, iteration_id, 'ref', data['games'], avg_nps, timemult, concurrency)

        return jsonify({'status': 'ok', 'remaining': remaining})

    # =========================================================================
    # SPSA Workers Dashboard
    # =========================================================================

    @app.route('/spsa/workers')
    def spsa_workers():
        """Dashboard showing SPSA worker activity and statistics."""
        from sqlalchemy.orm import joinedload

        # Get workers seen in the last 4 hours
        cutoff = datetime.utcnow() - timedelta(hours=4)
        workers = SpsaWorker.query.filter(
            SpsaWorker.last_seen_at >= cutoff
        ).order_by(SpsaWorker.last_seen_at.desc()).all()

        # Batch-fetch latest timemult and concurrency for all workers in one query
        worker_ids = [w.id for w in workers]
        latest_hb_data = {}  # worker_id -> {timemult, concurrency}
        if worker_ids:
            from sqlalchemy import func
            latest_hb_subq = db.session.query(
                SpsaWorkerHeartbeat.worker_id,
                func.max(SpsaWorkerHeartbeat.id).label('max_id')
            ).filter(
                SpsaWorkerHeartbeat.worker_id.in_(worker_ids)
            ).group_by(SpsaWorkerHeartbeat.worker_id).subquery()

            latest_hbs = db.session.query(
                SpsaWorkerHeartbeat.worker_id,
                SpsaWorkerHeartbeat.timemult,
                SpsaWorkerHeartbeat.concurrency
            ).join(
                latest_hb_subq,
                SpsaWorkerHeartbeat.id == latest_hb_subq.c.max_id
            ).all()
            latest_hb_data = {
                row.worker_id: {'timemult': row.timemult, 'concurrency': row.concurrency}
                for row in latest_hbs
            }

        # Calculate time since last seen for each worker
        now = datetime.utcnow()
        workers_data = []
        for w in workers:
            time_since = now - w.last_seen_at
            if time_since.total_seconds() < 60:
                last_seen_str = f"{int(time_since.total_seconds())}s ago"
                status = 'active'
            elif time_since.total_seconds() < 3600:
                last_seen_str = f"{int(time_since.total_seconds() / 60)}m ago"
                status = 'active' if time_since.total_seconds() < 300 else 'idle'
            elif time_since.total_seconds() < 86400:
                last_seen_str = f"{int(time_since.total_seconds() / 3600)}h ago"
                status = 'idle'
            else:
                last_seen_str = f"{int(time_since.total_seconds() / 86400)}d ago"
                status = 'offline'

            hb_info = latest_hb_data.get(w.id, {})
            tm = hb_info.get('timemult')
            conc = hb_info.get('concurrency')

            workers_data.append({
                'id': w.id,
                'name': w.worker_name,
                'status': status,
                'last_seen': last_seen_str,
                'last_seen_at': w.last_seen_at,
                'first_seen_at': w.first_seen_at,
                'last_phase': w.last_phase,
                'last_iteration_id': w.last_iteration_id,
                'total_games': w.total_games,
                'total_spsa_games': w.total_spsa_games,
                'total_ref_games': w.total_ref_games,
                'avg_nps': w.avg_nps,
                'last_timemult': tm,
                'concurrency': conc,
                'effective_concurrency': (conc / tm) if conc and tm and tm > 0 else (conc if conc else None),
            })

        # Compute effective concurrent games from workers seen in last 15 minutes
        active_cutoff = timedelta(minutes=15)
        effective_concurrency = 0.0
        for w in workers_data:
            time_since = now - w['last_seen_at']
            if time_since <= active_cutoff:
                conc = w.get('concurrency')
                tm = w.get('last_timemult')
                if conc and tm and tm > 0:
                    effective_concurrency += conc / tm
                elif conc:
                    effective_concurrency += conc

        # Get recent heartbeats for activity table (eager-load relationships)
        heartbeats = SpsaWorkerHeartbeat.query.options(
            joinedload(SpsaWorkerHeartbeat.worker),
            joinedload(SpsaWorkerHeartbeat.iteration),
        ).order_by(
            SpsaWorkerHeartbeat.created_at.desc()
        ).limit(100).all()

        heartbeats_data = []
        for h in heartbeats:
            time_since = now - h.created_at
            if time_since.total_seconds() < 60:
                time_str = f"{int(time_since.total_seconds())}s ago"
            elif time_since.total_seconds() < 3600:
                time_str = f"{int(time_since.total_seconds() / 60)}m ago"
            else:
                time_str = f"{int(time_since.total_seconds() / 3600)}h ago"

            iteration = h.iteration
            heartbeats_data.append({
                'worker_name': h.worker.worker_name,
                'phase': h.phase,
                'games': h.games_reported,
                'nps': h.avg_nps,
                'timemult': h.timemult,
                'run_number': iteration.run_id if iteration else None,
                'iteration_number': iteration.iteration_number if iteration else None,
                'time': time_str,
                'created_at': h.created_at,
            })

        # Calculate summary stats
        total_workers = len(workers)
        active_workers = sum(1 for w in workers_data if w['status'] == 'active')
        total_games = sum(w.total_games for w in workers)
        avg_nps_all = None
        nps_values = [w.avg_nps for w in workers if w.avg_nps]
        if nps_values:
            avg_nps_all = sum(nps_values) // len(nps_values)

        return render_template(
            'spsa_workers.html',
            workers=workers_data,
            heartbeats=heartbeats_data,
            total_workers=total_workers,
            active_workers=active_workers,
            total_games=total_games,
            avg_nps=avg_nps_all,
            effective_concurrency=round(effective_concurrency, 1),
        )

    # =========================================================================
    # Head-to-Head Match Endpoints
    # =========================================================================

    @app.route('/api/h2h/create', methods=['POST'])
    def h2h_create():
        """Create a new H2H match. Returns the match ID."""
        if not verify_worker_api_key():
            return jsonify({'error': 'Invalid API key'}), 401
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing JSON body'}), 400
        required = ['engine1_tag', 'engine2_tag', 'total_games', 'time_control']
        for field in required:
            if field not in data:
                return jsonify({'error': f'Missing field: {field}'}), 400
        match = H2hMatch(
            engine1_tag=data['engine1_tag'],
            engine2_tag=data['engine2_tag'],
            total_games=data['total_games'],
            time_control=data['time_control'],
        )
        db.session.add(match)
        db.session.commit()
        return jsonify({'match_id': match.id})

    @app.route('/api/h2h/result', methods=['POST'])
    def h2h_result():
        """Report a batch of game results for a H2H match."""
        if not verify_worker_api_key():
            return jsonify({'error': 'Invalid API key'}), 401
        data = request.get_json()
        if not data or 'match_id' not in data:
            return jsonify({'error': 'Missing match_id'}), 400

        match_id = data['match_id']
        e1_wins = data.get('engine1_wins', 0)
        e2_wins = data.get('engine2_wins', 0)
        draws = data.get('draws', 0)
        games = e1_wins + e2_wins + draws

        if games == 0:
            return jsonify({'ok': True})

        # Atomic increment
        db.session.execute(
            db.text('''
                UPDATE h2h_matches SET
                    games_played = games_played + :games,
                    engine1_wins = engine1_wins + :e1w,
                    engine2_wins = engine2_wins + :e2w,
                    draws = draws + :d
                WHERE id = :mid
            '''),
            {'games': games, 'e1w': e1_wins, 'e2w': e2_wins, 'd': draws, 'mid': match_id}
        )
        db.session.commit()
        return jsonify({'ok': True})

    @app.route('/api/h2h/calibration', methods=['POST'])
    def h2h_calibration():
        """Report NPS calibration results for a H2H match."""
        if not verify_worker_api_key():
            return jsonify({'error': 'Invalid API key'}), 401
        data = request.get_json()
        if not data or 'match_id' not in data:
            return jsonify({'error': 'Missing match_id'}), 400

        match = H2hMatch.query.get(data['match_id'])
        if not match:
            return jsonify({'error': 'Match not found'}), 404

        if 'avg_nps' in data:
            match.avg_nps = data['avg_nps']
        if 'timemult' in data:
            match.timemult = data['timemult']
        if 'effective_tc' in data:
            match.effective_tc = data['effective_tc']
        db.session.commit()
        return jsonify({'ok': True})

    @app.route('/api/h2h/pgn', methods=['POST'])
    def h2h_pgn():
        """Upload the full PGN for a completed H2H match."""
        if not verify_worker_api_key():
            return jsonify({'error': 'Invalid API key'}), 401
        data = request.get_json()
        if not data or 'match_id' not in data or 'pgn' not in data:
            return jsonify({'error': 'Missing match_id or pgn'}), 400

        match = H2hMatch.query.get(data['match_id'])
        if not match:
            return jsonify({'error': 'Match not found'}), 404

        match.pgn = data['pgn']
        match.status = 'completed'
        match.completed_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'ok': True})

    @app.route('/api/h2h/fail', methods=['POST'])
    def h2h_fail():
        """Mark a H2H match as failed."""
        if not verify_worker_api_key():
            return jsonify({'error': 'Invalid API key'}), 401
        data = request.get_json()
        if not data or 'match_id' not in data:
            return jsonify({'error': 'Missing match_id'}), 400

        match = H2hMatch.query.get(data['match_id'])
        if not match:
            return jsonify({'error': 'Match not found'}), 404

        match.status = 'failed'
        match.completed_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'ok': True})

    @app.route('/api/h2h/status/<int:match_id>')
    def h2h_status(match_id):
        """Poll for H2H match status (live updates)."""
        match = H2hMatch.query.get(match_id)
        if not match:
            return jsonify({'error': 'Match not found'}), 404
        return jsonify({
            'match_id': match.id,
            'engine1_tag': match.engine1_tag,
            'engine2_tag': match.engine2_tag,
            'total_games': match.total_games,
            'games_played': match.games_played,
            'engine1_wins': match.engine1_wins,
            'engine2_wins': match.engine2_wins,
            'draws': match.draws,
            'status': match.status,
            'avg_nps': match.avg_nps,
            'timemult': match.timemult,
            'effective_tc': match.effective_tc,
        })

    @app.route('/h2h')
    def h2h_dashboard():
        """Head-to-head matches dashboard with BayesElo leaderboard."""
        import math
        import re
        import subprocess
        import tempfile

        matches = H2hMatch.query.order_by(H2hMatch.created_at.desc()).all()

        # Build BayesElo leaderboard from all completed match PGNs
        leaderboard = []
        completed_matches = [cm for cm in matches if cm.status == 'completed' and cm.pgn and len(cm.pgn) > 100]
        try:
            if completed_matches:
                bayeselo_bin = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'bin', 'bayeselo')
                if os.path.isfile(bayeselo_bin):
                    # Concatenate all PGNs
                    with tempfile.NamedTemporaryFile(mode='w', suffix='.pgn', delete=False) as tmp:
                        for cm in completed_matches:
                            tmp.write(cm.pgn)
                            tmp.write('\n')
                        tmp_path = tmp.name

                    try:
                        commands = f"readpgn {tmp_path}\nelo\nmm\nratings\nx\nx\n"
                        result = subprocess.run(
                            [bayeselo_bin], input=commands, capture_output=True, text=True, timeout=30
                        )
                        for line in result.stdout.splitlines():
                            rm = re.match(r'\s*(\d+)\s+(\S+)\s+(-?\d+)\s+(-?\d+)\s+(-?\d+)', line)
                            if rm:
                                leaderboard.append({
                                    'rank': int(rm.group(1)),
                                    'engine': rm.group(2),
                                    'elo': int(rm.group(3)),
                                    'plus': int(rm.group(4)),
                                    'minus': int(rm.group(5)),
                                })
                    except (subprocess.TimeoutExpired, FileNotFoundError):
                        pass
                    finally:
                        os.unlink(tmp_path)

                # Compute per-engine game counts from match stats
                engine_games = {}
                for cm in completed_matches:
                    g = cm.games_played or 0
                    engine_games[cm.engine1_tag] = engine_games.get(cm.engine1_tag, 0) + g
                    engine_games[cm.engine2_tag] = engine_games.get(cm.engine2_tag, 0) + g
                for entry in leaderboard:
                    entry['games'] = engine_games.get(entry['engine'], 0)
        except Exception:
            leaderboard = []

        # Normalize leaderboard so lowest = 0
        if leaderboard:
            min_elo = min(e['elo'] for e in leaderboard)
            for entry in leaderboard:
                entry['elo_diff'] = entry['elo'] - min_elo

        # Compute Elo diff for each match
        matches_data = []
        for m in matches:
            total = m.engine1_wins + m.engine2_wins + m.draws
            elo_diff = None
            if total > 0:
                score = (m.engine1_wins + m.draws * 0.5) / total
                if 0 < score < 1:
                    elo_diff = round(-400 * math.log10(1 / score - 1), 1)
                elif score >= 1:
                    elo_diff = 999
                elif score <= 0:
                    elo_diff = -999
            matches_data.append({
                'id': m.id,
                'engine1_tag': m.engine1_tag,
                'engine2_tag': m.engine2_tag,
                'time_control': m.time_control,
                'total_games': m.total_games,
                'games_played': m.games_played,
                'engine1_wins': m.engine1_wins,
                'engine2_wins': m.engine2_wins,
                'draws': m.draws,
                'status': m.status,
                'elo_diff': elo_diff,
                'avg_nps': m.avg_nps,
                'timemult': m.timemult,
                'effective_tc': m.effective_tc,
                'created_at': m.created_at,
                'completed_at': m.completed_at,
            })

        return render_template(
            'h2h.html',
            matches=matches_data,
            leaderboard=leaderboard,
        )

    # =========================================================================
    # NNUE Data Generation Endpoints
    # =========================================================================

    @app.route('/api/datagen/create', methods=['POST'])
    def datagen_create():
        """Create a new datagen job."""
        if not verify_worker_api_key():
            return jsonify({'error': 'Invalid API key'}), 401
        data = request.get_json()
        if not data:
            return jsonify({'error': 'Missing JSON body'}), 400
        job = DatagenJob(
            engine_tag=data['engine_tag'],
            depth=data['depth'],
            total_games=data['total_games'],
            s3_path=data.get('s3_path'),
        )
        db.session.add(job)
        db.session.commit()
        return jsonify({'job_id': job.id})

    @app.route('/api/datagen/progress', methods=['POST'])
    def datagen_progress():
        """Report progress for a datagen job."""
        if not verify_worker_api_key():
            return jsonify({'error': 'Invalid API key'}), 401
        data = request.get_json()
        if not data or 'job_id' not in data:
            return jsonify({'error': 'Missing job_id'}), 400

        job = DatagenJob.query.get(data['job_id'])
        if not job:
            return jsonify({'error': 'Job not found'}), 404

        if 'games_completed' in data:
            job.games_completed = data['games_completed']
        if 'positions_generated' in data:
            job.positions_generated = data['positions_generated']
        if 'games_per_second' in data:
            job.games_per_second = data['games_per_second']
        if 'positions_per_second' in data:
            job.positions_per_second = data['positions_per_second']
        if 'status' in data:
            job.status = data['status']
            if data['status'] in ('completed', 'failed'):
                job.completed_at = datetime.utcnow()
        job.last_update_at = datetime.utcnow()
        db.session.commit()
        return jsonify({'ok': True})

    @app.route('/datagen')
    def datagen_dashboard():
        """NNUE data generation dashboard."""
        jobs = DatagenJob.query.order_by(DatagenJob.created_at.desc()).all()

        jobs_data = []
        for j in jobs:
            elapsed = None
            if j.created_at:
                end = j.completed_at or datetime.utcnow()
                elapsed = (end - j.created_at).total_seconds()

            eta = None
            if j.games_per_second and j.games_per_second > 0 and j.games_completed < j.total_games:
                remaining = j.total_games - j.games_completed
                eta = remaining / j.games_per_second

            jobs_data.append({
                'id': j.id,
                'engine_tag': j.engine_tag,
                'depth': j.depth,
                'total_games': j.total_games,
                'games_completed': j.games_completed,
                'positions_generated': j.positions_generated,
                'status': j.status,
                's3_path': j.s3_path,
                'games_per_second': j.games_per_second,
                'positions_per_second': j.positions_per_second,
                'elapsed': elapsed,
                'eta': eta,
                'created_at': j.created_at,
                'last_update_at': j.last_update_at,
            })

        # Aggregate totals across all running jobs
        running = [j for j in jobs_data if j['status'] == 'running']
        totals = {
            'total_games': sum(j['total_games'] for j in running),
            'games_completed': sum(j['games_completed'] for j in running),
            'positions_generated': sum(j['positions_generated'] for j in running),
            'games_per_second': sum(j['games_per_second'] or 0 for j in running),
            'positions_per_second': sum(j['positions_per_second'] or 0 for j in running),
            'workers': len(running),
        }
        if totals['games_per_second'] > 0 and totals['games_completed'] < totals['total_games']:
            totals['eta'] = (totals['total_games'] - totals['games_completed']) / totals['games_per_second']
        else:
            totals['eta'] = None

        # Also sum positions from completed jobs
        all_positions = sum(j['positions_generated'] for j in jobs_data)

        return render_template('datagen.html', jobs=jobs_data, totals=totals, all_positions=all_positions)
