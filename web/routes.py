"""
Flask routes for the competition dashboard.
"""

from flask import render_template, request, redirect, url_for
from web.queries import get_dashboard_data, get_unique_hostnames, get_time_range, clear_elo_cache


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

    @app.route('/clear-cache', methods=['POST'])
    def clear_cache():
        """Clear the ELO filter cache, forcing recalculation."""
        num_cache, num_ratings = clear_elo_cache()
        # Redirect back to dashboard (preserving any query params would be complex, just go to root)
        return redirect(url_for('dashboard'))

    @app.route('/health')
    def health():
        """Health check endpoint."""
        return {'status': 'ok'}
