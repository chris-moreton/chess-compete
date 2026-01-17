"""
Flask routes for the competition dashboard.
"""

from flask import render_template, request, redirect, url_for
from web.queries import (
    get_dashboard_data, get_unique_hostnames, get_time_range,
    clear_elo_cache, recalculate_all_and_store
)
from web.models import Cup, CupRound, CupMatch, Engine, Game
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
            engine_type=engine_type
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
            engine_type=engine_type
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
