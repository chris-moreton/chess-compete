"""
Database queries for the competition dashboard.
"""

import math
import re
from sqlalchemy import func

# ELO calculation constants (must match compete.py)
K_FACTOR_PROVISIONAL = 40
K_FACTOR_ESTABLISHED = 20
PROVISIONAL_GAMES = 30
DEFAULT_ELO = 1500


def get_engine_type(name: str) -> str:
    """
    Determine the engine type from its name for display purposes.

    Returns:
        'stockfish' - for sf-* engines
        'official' - for vX.X.X official releases (e.g., v1.0.13)
        'dev' - for development versions (e.g., v031-delta-pruning)
        'other' - for everything else
    """
    if name.startswith('sf-'):
        return 'stockfish'
    # Official releases: vX.X.X pattern (semantic versioning)
    if re.match(r'^v\d+\.\d+\.\d+$', name):
        return 'official'
    # Dev versions: vXXX-name pattern
    if re.match(r'^v\d{2,3}-', name):
        return 'dev'
    return 'other'


def engine_matches_type_filter(engine_name: str, engine_type_filter: str | None) -> bool:
    """
    Check if an engine matches the engine type filter.

    Args:
        engine_name: Name of the engine
        engine_type_filter: 'rusty' for Rusty Rival (v*), 'stockfish' for Stockfish (sf*), None for all

    Returns:
        True if the engine matches the filter
    """
    if engine_type_filter is None:
        return True
    if engine_type_filter == 'rusty':
        return engine_name.startswith('v')
    if engine_type_filter == 'stockfish':
        return engine_name.startswith('sf')
    return True


def get_db():
    """Get db instance (late import to avoid circular imports)."""
    from web.database import db
    return db


def get_models():
    """Get models (late import to avoid circular imports)."""
    from web.models import Engine, Game, EloFilterCache, EloFilterRating
    return Engine, Game, EloFilterCache, EloFilterRating


def get_or_create_filter_cache(min_time_ms: int, max_time_ms: int, hostname: str | None, engine_type: str | None):
    """
    Find or create a filter cache entry.

    Args:
        min_time_ms: Minimum time per move in milliseconds
        max_time_ms: Maximum time per move in milliseconds
        hostname: Hostname filter (None = any host)
        engine_type: Engine type filter (None = all, 'rusty' = v*, 'stockfish' = sf*)

    Returns:
        EloFilterCache instance
    """
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    cache = EloFilterCache.query.filter_by(
        min_time_ms=min_time_ms,
        max_time_ms=max_time_ms,
        hostname=hostname,
        engine_type=engine_type
    ).first()

    if not cache:
        cache = EloFilterCache(
            min_time_ms=min_time_ms,
            max_time_ms=max_time_ms,
            hostname=hostname,
            engine_type=engine_type
        )
        db.session.add(cache)
        db.session.commit()

    return cache


def recalculate_elos_incremental(min_time_ms: int, max_time_ms: int, hostname: str | None, engine_type: str | None = None):
    """
    Incrementally recalculate ELO ratings for engines under the given filter.

    Algorithm:
    1. Get or create filter cache entry
    2. Load cached engine ratings (or use initial_elo for new engines)
    3. Query games with id > last_game_id matching filters
    4. Apply ELO updates sequentially
    5. Save updated ratings and last_game_id

    Args:
        min_time_ms: Minimum time per move in milliseconds
        max_time_ms: Maximum time per move in milliseconds
        hostname: Hostname filter (None = any host)
        engine_type: Engine type filter (None = all, 'rusty' = v*, 'stockfish' = sf*)
                     When set, BOTH players must be of this engine type

    Returns:
        dict: {engine_id: (elo, games_played)}
    """
    from sqlalchemy.orm import aliased
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    cache = get_or_create_filter_cache(min_time_ms, max_time_ms, hostname, engine_type)

    # Load existing ratings for this filter
    ratings = {}
    for r in EloFilterRating.query.filter_by(filter_id=cache.id):
        ratings[r.engine_id] = (float(r.elo), r.games_played)

    # Load initial ELOs for engines not yet in cache (only matching engine type)
    for engine in Engine.query.filter(Engine.active == True):
        if engine.id not in ratings:
            if engine_matches_type_filter(engine.name, engine_type):
                ratings[engine.id] = (float(engine.initial_elo or DEFAULT_ELO), 0)

    # Query new games matching filters
    # Need to join with Engine table to filter by engine type
    WhiteEngine = aliased(Engine)
    BlackEngine = aliased(Engine)

    query = db.session.query(Game).join(
        WhiteEngine, Game.white_engine_id == WhiteEngine.id
    ).join(
        BlackEngine, Game.black_engine_id == BlackEngine.id
    ).filter(
        Game.id > cache.last_game_id,
        Game.is_rated == True,
    )

    # Apply time filter - handle NULL values
    query = query.filter(
        (Game.time_per_move_ms >= min_time_ms) | (Game.time_per_move_ms.is_(None))
    )
    query = query.filter(
        (Game.time_per_move_ms <= max_time_ms) | (Game.time_per_move_ms.is_(None))
    )

    # Apply hostname filter
    if hostname is not None:
        query = query.filter(Game.hostname == hostname)

    # Apply engine type filter - BOTH players must match
    if engine_type == 'rusty':
        query = query.filter(WhiteEngine.name.like('v%'))
        query = query.filter(BlackEngine.name.like('v%'))
    elif engine_type == 'stockfish':
        query = query.filter(WhiteEngine.name.like('sf%'))
        query = query.filter(BlackEngine.name.like('sf%'))

    games = query.order_by(Game.id).all()

    # Apply ELO updates sequentially
    for game in games:
        white_elo, white_games = ratings.get(game.white_engine_id, (float(DEFAULT_ELO), 0))
        black_elo, black_games = ratings.get(game.black_engine_id, (float(DEFAULT_ELO), 0))

        # Calculate expected scores using standard Elo formula
        white_expected = 1 / (1 + 10 ** ((black_elo - white_elo) / 400))
        black_expected = 1 - white_expected

        # Actual scores from game result
        white_actual = float(game.white_score)
        black_actual = float(game.black_score)

        # K-factors based on games played
        white_k = K_FACTOR_PROVISIONAL if white_games < PROVISIONAL_GAMES else K_FACTOR_ESTABLISHED
        black_k = K_FACTOR_PROVISIONAL if black_games < PROVISIONAL_GAMES else K_FACTOR_ESTABLISHED

        # Calculate ELO changes
        white_change = white_k * (white_actual - white_expected)
        black_change = black_k * (black_actual - black_expected)

        # Update ratings
        ratings[game.white_engine_id] = (white_elo + white_change, white_games + 1)
        ratings[game.black_engine_id] = (black_elo + black_change, black_games + 1)

    # Save to database if there were new games
    if games:
        cache.last_game_id = games[-1].id

        for engine_id, (elo, games_played) in ratings.items():
            existing = EloFilterRating.query.filter_by(
                filter_id=cache.id, engine_id=engine_id
            ).first()

            if existing:
                existing.elo = elo
                existing.games_played = games_played
            else:
                db.session.add(EloFilterRating(
                    filter_id=cache.id,
                    engine_id=engine_id,
                    elo=elo,
                    games_played=games_played
                ))

        db.session.commit()

    return ratings


def get_engines_ranked_by_elo(active_only=True, min_time_ms=0, max_time_ms=999999999, hostname=None, engine_type=None):
    """
    Get engines sorted by Elo rating descending.

    Args:
        active_only: Only include active engines
        min_time_ms: Minimum time per move filter
        max_time_ms: Maximum time per move filter
        hostname: Hostname filter (None = any)
        engine_type: Engine type filter (None = all, 'rusty' = v*, 'stockfish' = sf*)

    Returns:
        List of dicts with engine info and ratings (elo, bayes_elo, ordo)
    """
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    # Recalculate ELOs incrementally
    ratings = recalculate_elos_incremental(min_time_ms, max_time_ms, hostname, engine_type)

    # Get BayesElo and Ordo ratings from cache (if available)
    cache = EloFilterCache.query.filter_by(
        min_time_ms=min_time_ms,
        max_time_ms=max_time_ms,
        hostname=hostname,
        engine_type=engine_type
    ).first()

    bayes_ordo_ratings = {}
    if cache:
        for r in EloFilterRating.query.filter_by(filter_id=cache.id):
            bayes_ordo_ratings[r.engine_id] = {
                'bayes_elo': float(r.bayes_elo) if r.bayes_elo is not None else None,
                'ordo': float(r.ordo) if r.ordo is not None else None
            }

    # Get engines
    query = Engine.query
    if active_only:
        query = query.filter(Engine.active == True)

    engines = query.all()

    # Build result list with ratings (only include engines matching type filter)
    result = []
    for engine in engines:
        if not engine_matches_type_filter(engine.name, engine_type):
            continue
        elo, games_played = ratings.get(engine.id, (float(engine.initial_elo or DEFAULT_ELO), 0))
        extra = bayes_ordo_ratings.get(engine.id, {'bayes_elo': None, 'ordo': None})
        result.append({
            'engine': engine,
            'elo': elo,
            'bayes_elo': extra['bayes_elo'],
            'ordo': extra['ordo'],
            'games_played': games_played
        })

    # Sort by ELO descending
    result.sort(key=lambda x: x['elo'], reverse=True)

    return result


def get_h2h_raw_data(min_time_ms=0, max_time_ms=999999999, hostname=None, engine_type=None):
    """
    Get raw head-to-head data from games table.
    Only includes rated games matching the filters.

    Args:
        min_time_ms: Minimum time per move filter
        max_time_ms: Maximum time per move filter
        hostname: Hostname filter (None = any)
        engine_type: Engine type filter (None = all, 'rusty' = v*, 'stockfish' = sf*)
                     When set, BOTH players must be of this engine type

    Returns:
        dict: {(white_id, black_id): {'white_points': float, 'black_points': float, 'games': int}}
    """
    from sqlalchemy.orm import aliased
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    WhiteEngine = aliased(Engine)
    BlackEngine = aliased(Engine)

    query = db.session.query(
        Game.white_engine_id,
        Game.black_engine_id,
        func.sum(Game.white_score).label('white_points'),
        func.sum(Game.black_score).label('black_points'),
        func.count(Game.id).label('total_games')
    ).join(
        WhiteEngine, Game.white_engine_id == WhiteEngine.id
    ).join(
        BlackEngine, Game.black_engine_id == BlackEngine.id
    ).filter(
        Game.is_rated == True
    )

    # Apply time filter - handle NULL values
    query = query.filter(
        (Game.time_per_move_ms >= min_time_ms) | (Game.time_per_move_ms.is_(None))
    )
    query = query.filter(
        (Game.time_per_move_ms <= max_time_ms) | (Game.time_per_move_ms.is_(None))
    )

    # Apply hostname filter
    if hostname is not None:
        query = query.filter(Game.hostname == hostname)

    # Apply engine type filter - BOTH players must match
    if engine_type == 'rusty':
        query = query.filter(WhiteEngine.name.like('v%'))
        query = query.filter(BlackEngine.name.like('v%'))
    elif engine_type == 'stockfish':
        query = query.filter(WhiteEngine.name.like('sf%'))
        query = query.filter(BlackEngine.name.like('sf%'))

    results = query.group_by(
        Game.white_engine_id,
        Game.black_engine_id
    ).all()

    h2h = {}
    for row in results:
        key = (row.white_engine_id, row.black_engine_id)
        h2h[key] = {
            'white_points': float(row.white_points or 0),
            'black_points': float(row.black_points or 0),
            'games': row.total_games
        }

    return h2h


def calculate_expected_score(elo_a: float, elo_b: float, num_games: int) -> float:
    """
    Calculate expected score for player A against player B over num_games.
    Uses standard Elo formula: E = 1 / (1 + 10^((Rb - Ra) / 400))
    """
    if num_games == 0:
        return 0
    expected_per_game = 1 / (1 + 10 ** ((elo_b - elo_a) / 400))
    return expected_per_game * num_games


def stability_to_color(stability: int) -> str:
    """
    Convert stability score (0-100) to a background color.

    - 100 = dark green (perfect match with expected)
    - 50 = white/neutral
    - 0 = dark red (poor match)

    Args:
        stability: Score from 0-100

    Returns:
        CSS color string
    """
    if stability >= 50:
        # Green gradient: 50 = white, 100 = dark green (#4caf50)
        intensity = (stability - 50) / 50.0  # 0 to 1
        r = int(255 - (255 - 76) * intensity)   # 255 -> 76
        g = int(255 - (255 - 175) * intensity)  # 255 -> 175
        b = int(255 - (255 - 80) * intensity)   # 255 -> 80
    else:
        # Red gradient: 50 = white, 0 = dark red (#e57373)
        intensity = (50 - stability) / 50.0  # 0 to 1
        r = int(255 - (255 - 229) * intensity)  # 255 -> 229
        g = int(255 - (255 - 115) * intensity)  # 255 -> 115
        b = int(255 - (255 - 115) * intensity)  # 255 -> 115

    return f'rgb({r}, {g}, {b})'


def calculate_excess_deviation(deviation: float, num_games: int) -> float:
    """
    Calculate how many standard deviations beyond tolerance the result is.
    Returns 0 if within tolerance, positive value if outside.

    Args:
        deviation: Points above/below expected
        num_games: Number of games played

    Returns:
        Excess deviation in standard deviation units (0 = within tolerance)
    """
    if num_games == 0:
        return 0.0

    # Same tolerance calculation as deviation_to_color
    tolerance = 0.5 * math.sqrt(num_games)
    tolerance = max(1.0, tolerance)

    excess = abs(deviation) - tolerance
    if excess <= 0:
        return 0.0

    # Normalize by tolerance to get "excess standard deviations"
    return excess / tolerance


def deviation_to_color(deviation: float, num_games: int) -> str:
    """
    Convert deviation from expected to a background color.

    - White/neutral if within tolerance of expected (normal variance)
    - Green if overperforming (positive deviation beyond tolerance)
    - Red if underperforming (negative deviation beyond tolerance)

    Args:
        deviation: Points above (positive) or below (negative) expected
        num_games: Number of games played

    Returns:
        CSS color string
    """
    if num_games == 0:
        return '#f5f5f5'  # No games - neutral gray

    # Tolerance based on statistical variance
    # Standard deviation of game outcomes ≈ 0.5 * sqrt(n)
    # Use ~1 standard deviation as tolerance (generous)
    tolerance = 0.5 * math.sqrt(num_games)
    # Minimum tolerance of 1.0 points for very small sample sizes
    tolerance = max(1.0, tolerance)

    # If within tolerance, return neutral white
    if abs(deviation) <= tolerance:
        return '#ffffff'

    # Calculate excess deviation beyond tolerance
    excess = abs(deviation) - tolerance

    # Scale intensity: how much beyond tolerance relative to expected variance
    # Use 2 more standard deviations as the range for full intensity
    # So total range is 1 SD (tolerance) to 3 SD (full color)
    full_intensity_threshold = tolerance * 2  # 2 more SDs
    intensity = min(1.0, excess / full_intensity_threshold)

    # Apply sqrt to make gradients more visible at lower intensities
    intensity = math.sqrt(intensity)

    if deviation > 0:
        # Green: overperforming
        # From white (#ffffff) to green (#4caf50)
        r = int(255 - (255 - 76) * intensity)   # 255 -> 76
        g = int(255 - (255 - 175) * intensity)  # 255 -> 175
        b = int(255 - (255 - 80) * intensity)   # 255 -> 80
    else:
        # Red: underperforming
        # From white (#ffffff) to red (#e57373)
        r = int(255 - (255 - 229) * intensity)  # 255 -> 229
        g = int(255 - (255 - 115) * intensity)  # 255 -> 115
        b = int(255 - (255 - 115) * intensity)  # 255 -> 115

    return f'rgb({r}, {g}, {b})'


def build_h2h_grid(engines, h2h_raw):
    """
    Build the H2H grid for the dashboard.

    Args:
        engines: List of engine dicts with 'engine', 'elo', 'games_played'
        h2h_raw: Raw H2H data from get_h2h_raw_data()

    Returns:
        List of row dicts with engine info, cells, and stability score
    """
    # Build lookups
    engine_elos = {e['engine'].id: e['elo'] for e in engines}

    grid = []
    for row_idx, row_engine in enumerate(engines):
        row_id = row_engine['engine'].id
        row_elo = row_engine['elo']
        row_rank = row_idx + 1

        cells = []
        # Track deviations for stability calculation
        total_excess_deviation = 0.0
        total_opponent_games = 0
        # Track total games against visible engines (for accurate games_played count)
        visible_games_played = 0

        for col_idx, col_engine in enumerate(engines):
            col_id = col_engine['engine'].id
            col_elo = col_engine['elo']
            col_rank = col_idx + 1

            # Same engine - diagonal
            if row_id == col_id:
                cells.append({
                    'score': '-',
                    'bg_color': '#e0e0e0',
                    'games': 0,
                    'tooltip': ''
                })
                continue

            # Get H2H data from both directions
            # row_engine as white vs col_engine
            as_white = h2h_raw.get((row_id, col_id), {'white_points': 0, 'black_points': 0, 'games': 0})
            # row_engine as black vs col_engine (col as white)
            as_black = h2h_raw.get((col_id, row_id), {'white_points': 0, 'black_points': 0, 'games': 0})

            # Calculate row_engine's total points against col_engine
            row_points = as_white['white_points'] + as_black['black_points']
            col_points = as_white['black_points'] + as_black['white_points']
            total_games = as_white['games'] + as_black['games']

            # Add to visible games count (all games against visible opponents)
            visible_games_played += total_games

            if total_games == 0:
                cells.append({
                    'score': '-',
                    'bg_color': '#f5f5f5',
                    'games': 0,
                    'tooltip': 'No games played'
                })
                continue

            # Calculate expected score for row_engine against col_engine
            expected_row = calculate_expected_score(row_elo, col_elo, total_games)
            deviation = row_points - expected_row  # Positive = overperforming, negative = underperforming

            # Track for stability calculation (weighted by games)
            excess = calculate_excess_deviation(deviation, total_games)
            total_excess_deviation += excess * total_games
            total_opponent_games += total_games

            # Color based purely on deviation from expected:
            # - Within tolerance -> white
            # - Overperforming (positive) -> green
            # - Underperforming (negative) -> red
            bg_color = deviation_to_color(deviation, total_games)

            # Build tooltip with more detail
            expected_str = f"{expected_row:.1f}"
            if deviation > 0:
                dev_str = f"+{deviation:.1f}"
            else:
                dev_str = f"{deviation:.1f}"
            tooltip = f"{total_games} games | Expected: {expected_str} | Actual: {row_points:.0f} ({dev_str})"

            cells.append({
                'score': f"{row_points:.0f}-{col_points:.0f}",
                'bg_color': bg_color,
                'games': total_games,
                'tooltip': tooltip
            })

        # Calculate stability score (0-100, higher = more stable/settled)
        # Average excess deviation weighted by games, then convert to 0-100
        if total_opponent_games > 0:
            avg_excess = total_excess_deviation / total_opponent_games
            # Map avg_excess to 0-100: 0 excess = 100, 2+ excess SDs = 0
            stability = max(0, 100 - (avg_excess * 50))
        else:
            stability = 0  # No games = no stability data

        stability_rounded = round(stability)
        grid.append({
            'rank': row_rank,
            'engine_name': row_engine['engine'].name,
            'engine_type': get_engine_type(row_engine['engine'].name),
            'elo': row_elo,
            'bayes_elo': row_engine.get('bayes_elo'),
            'ordo': row_engine.get('ordo'),
            'games_played': visible_games_played,  # Use games against visible engines, not cached total
            'stability': stability_rounded,
            'stability_color': stability_to_color(stability_rounded),
            'least_stable': False,
            'cells': cells
        })

    # Mark the engine with lowest stability (only consider engines with games)
    engines_with_games = [row for row in grid if row['games_played'] > 0]
    if engines_with_games:
        min_stability = min(row['stability'] for row in engines_with_games)
        for row in grid:
            if row['stability'] == min_stability and row['games_played'] > 0:
                row['least_stable'] = True
                break  # Only mark one engine

    return grid


def get_last_played_engines():
    """
    Get the engine names from the most recently played game.
    Returns tuple of (engine1_name, engine2_name) or (None, None) if no games.
    """
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    last_game = db.session.query(Game).order_by(Game.created_at.desc()).first()

    if not last_game:
        return (None, None)

    white_engine = db.session.query(Engine).filter(Engine.id == last_game.white_engine_id).first()
    black_engine = db.session.query(Engine).filter(Engine.id == last_game.black_engine_id).first()

    return (white_engine.name if white_engine else None, black_engine.name if black_engine else None)


def get_unique_hostnames():
    """Get list of unique hostnames from games table."""
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    results = db.session.query(Game.hostname).distinct().filter(
        Game.hostname.isnot(None)
    ).order_by(Game.hostname).all()

    return [r[0] for r in results]


def get_time_range():
    """Get min and max time_per_move_ms from games table."""
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    result = db.session.query(
        func.min(Game.time_per_move_ms),
        func.max(Game.time_per_move_ms)
    ).filter(
        Game.time_per_move_ms.isnot(None)
    ).first()

    min_time = result[0] if result[0] is not None else 0
    max_time = result[1] if result[1] is not None else 10000

    return min_time, max_time


def get_dashboard_data(active_only=True, min_time_ms=0, max_time_ms=999999999, hostname=None, engine_type=None,
                       sort_column=None, sort_direction=None):
    """
    Get all data needed for the dashboard.

    Args:
        active_only: Only include active engines
        min_time_ms: Minimum time per move filter
        max_time_ms: Maximum time per move filter
        hostname: Hostname filter (None = any)
        engine_type: Engine type filter (None = all, 'rusty' = v*, 'stockfish' = sf*)
        sort_column: Column to sort by ('name', 'elo', 'bayes', 'ordo', 'stability', 'games')
        sort_direction: Sort direction ('asc' or 'desc')

    Returns:
        (engines, grid, column_headers, last_played_engines)
    """
    engines = get_engines_ranked_by_elo(
        active_only=active_only,
        min_time_ms=min_time_ms,
        max_time_ms=max_time_ms,
        hostname=hostname,
        engine_type=engine_type
    )

    if not engines:
        return [], [], [], (None, None)

    h2h_raw = get_h2h_raw_data(
        min_time_ms=min_time_ms,
        max_time_ms=max_time_ms,
        hostname=hostname,
        engine_type=engine_type
    )

    # Build grid first (needs engines in ELO order for stability calculation)
    grid = build_h2h_grid(engines, h2h_raw)

    # Now apply requested sort to both engines and grid together
    if sort_column and sort_column != 'elo':
        # Create combined list for sorting
        combined = list(zip(engines, grid))

        # Define sort key based on column
        if sort_column == 'name':
            key_func = lambda x: x[0]['engine'].name.lower()
            default_reverse = False
        elif sort_column == 'bayes':
            key_func = lambda x: x[1].get('bayes_elo') or 0
            default_reverse = True
        elif sort_column == 'ordo':
            key_func = lambda x: x[1].get('ordo') or 0
            default_reverse = True
        elif sort_column == 'stability':
            key_func = lambda x: x[1].get('stability', 0)
            default_reverse = True
        elif sort_column == 'games':
            key_func = lambda x: x[1].get('games_played', 0)
            default_reverse = True
        else:
            key_func = lambda x: x[0]['elo']
            default_reverse = True

        # Determine sort direction
        reverse = default_reverse
        if sort_direction == 'asc':
            reverse = False
        elif sort_direction == 'desc':
            reverse = True

        combined.sort(key=key_func, reverse=reverse)
        engines, grid = zip(*combined)
        engines = list(engines)
        grid = list(grid)

        # Update ranks in grid
        for i, row in enumerate(grid):
            row['rank'] = i + 1
    elif sort_direction == 'asc':
        # ELO column with ascending order (reverse the default)
        combined = list(zip(engines, grid))
        combined.sort(key=lambda x: x[0]['elo'], reverse=False)
        engines, grid = zip(*combined)
        engines = list(engines)
        grid = list(grid)
        for i, row in enumerate(grid):
            row['rank'] = i + 1

    column_headers = [(i + 1, e['engine'].name) for i, e in enumerate(engines)]
    last_played = get_last_played_engines()

    return engines, grid, column_headers, last_played


def clear_elo_cache():
    """
    Clear all ELO filter cache entries and their associated ratings.
    This forces a full recalculation the next time any filter is used.

    Returns:
        tuple: (num_cache_entries_deleted, num_ratings_deleted)
    """
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    # Count before deleting
    num_ratings = EloFilterRating.query.count()
    num_cache = EloFilterCache.query.count()

    # Delete all ratings first (due to foreign key), then cache entries
    # Note: CASCADE should handle ratings, but be explicit
    EloFilterRating.query.delete()
    EloFilterCache.query.delete()

    db.session.commit()

    return num_cache, num_ratings


def get_game_results_for_filter(min_time_ms: int, max_time_ms: int, hostname: str | None, engine_type: str | None):
    """
    Get all game results matching the filter criteria.

    Returns:
        List of tuples: (white_engine_id, black_engine_id, white_score, black_score)
    """
    from sqlalchemy.orm import aliased
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    WhiteEngine = aliased(Engine)
    BlackEngine = aliased(Engine)

    query = db.session.query(
        Game.white_engine_id,
        Game.black_engine_id,
        Game.white_score,
        Game.black_score
    ).join(
        WhiteEngine, Game.white_engine_id == WhiteEngine.id
    ).join(
        BlackEngine, Game.black_engine_id == BlackEngine.id
    ).filter(
        Game.is_rated == True
    )

    # Apply time filter
    query = query.filter(
        (Game.time_per_move_ms >= min_time_ms) | (Game.time_per_move_ms.is_(None))
    )
    query = query.filter(
        (Game.time_per_move_ms <= max_time_ms) | (Game.time_per_move_ms.is_(None))
    )

    # Apply hostname filter
    if hostname is not None:
        query = query.filter(Game.hostname == hostname)

    # Apply engine type filter - BOTH players must match
    if engine_type == 'rusty':
        query = query.filter(WhiteEngine.name.like('v%'))
        query = query.filter(BlackEngine.name.like('v%'))
    elif engine_type == 'stockfish':
        query = query.filter(WhiteEngine.name.like('sf%'))
        query = query.filter(BlackEngine.name.like('sf%'))

    return query.all()


def calculate_bayeselo(game_results, engine_ids, draw_elo=97.0, iterations=50):
    """
    Calculate BayesElo ratings using iterative maximum likelihood estimation.

    The BayesElo model accounts for draws explicitly using a "draw elo" parameter.
    Higher draw_elo means draws are more likely (typical chess value: ~97).

    Args:
        game_results: List of (white_id, black_id, white_score, black_score)
        engine_ids: List of engine IDs to rate
        draw_elo: Draw tendency parameter (default 97 for chess)
        iterations: Number of optimization iterations

    Returns:
        dict: {engine_id: bayes_elo_rating}
    """
    if not game_results or len(engine_ids) < 2:
        return {eid: 1500.0 for eid in engine_ids}

    # Initialize ratings at 0 (BayesElo uses 0-centered ratings internally)
    ratings = {eid: 0.0 for eid in engine_ids}

    # Count games for each engine
    games_count = {eid: 0 for eid in engine_ids}
    for white_id, black_id, _, _ in game_results:
        if white_id in games_count:
            games_count[white_id] += 1
        if black_id in games_count:
            games_count[black_id] += 1

    # Pre-compute draw elo factor
    draw_factor = 10 ** (draw_elo / 400.0)

    # Iterative optimization
    for _ in range(iterations):
        # Accumulate expected vs actual scores
        expected = {eid: 0.0 for eid in engine_ids}
        actual = {eid: 0.0 for eid in engine_ids}

        for white_id, black_id, white_score, black_score in game_results:
            if white_id not in ratings or black_id not in ratings:
                continue

            # Rating difference factor
            diff = (ratings[white_id] - ratings[black_id]) / 400.0
            white_factor = 10 ** diff
            black_factor = 10 ** (-diff)

            # Total for normalization
            total = white_factor + draw_factor + black_factor

            # Expected scores (win=1, draw=0.5, loss=0)
            white_expected = (white_factor + 0.5 * draw_factor) / total
            black_expected = (black_factor + 0.5 * draw_factor) / total

            expected[white_id] += white_expected
            expected[black_id] += black_expected
            actual[white_id] += float(white_score)
            actual[black_id] += float(black_score)

        # Update ratings based on actual - expected
        for eid in engine_ids:
            if games_count[eid] > 0 and expected[eid] > 0:
                # Adjust rating based on performance difference
                diff = actual[eid] - expected[eid]
                # Scale adjustment by number of games (like K-factor)
                k = 40 if games_count[eid] < 30 else 20
                ratings[eid] += k * diff / max(1, games_count[eid] / 10)

    # Convert to standard Elo scale (center around 1500)
    if ratings:
        avg_rating = sum(ratings.values()) / len(ratings)
        ratings = {eid: r - avg_rating + 1500 for eid, r in ratings.items()}

    return ratings


def calculate_ordo(game_results, engine_ids, iterations=100):
    """
    Calculate Ordo ratings using maximum likelihood estimation.

    Ordo uses the Bradley-Terry model with logistic distribution,
    finding ratings that maximize the likelihood of observed results.
    Unlike sequential Elo, Ordo gives the same result regardless of game order.

    Args:
        game_results: List of (white_id, black_id, white_score, black_score)
        engine_ids: List of engine IDs to rate
        iterations: Number of optimization iterations

    Returns:
        dict: {engine_id: ordo_rating}
    """
    if not game_results or len(engine_ids) < 2:
        return {eid: 1500.0 for eid in engine_ids}

    # Build win/draw/loss matrix
    # scores[i][j] = (wins for i against j, draws, losses)
    engine_idx = {eid: i for i, eid in enumerate(engine_ids)}
    n = len(engine_ids)

    wins = [[0.0] * n for _ in range(n)]
    draws = [[0.0] * n for _ in range(n)]

    for white_id, black_id, white_score, black_score in game_results:
        if white_id not in engine_idx or black_id not in engine_idx:
            continue

        wi = engine_idx[white_id]
        bi = engine_idx[black_id]
        ws = float(white_score)

        if ws == 1.0:
            wins[wi][bi] += 1
        elif ws == 0.0:
            wins[bi][wi] += 1
        else:  # draw
            draws[wi][bi] += 1
            draws[bi][wi] += 1

    # Initialize ratings
    ratings = [0.0] * n

    # Iterative optimization using MM algorithm (minorization-maximization)
    for _ in range(iterations):
        new_ratings = [0.0] * n

        for i in range(n):
            numerator = 0.0
            denominator = 0.0

            for j in range(n):
                if i == j:
                    continue

                total_games = wins[i][j] + wins[j][i] + draws[i][j]
                if total_games == 0:
                    continue

                # Score for player i against j
                score_i = wins[i][j] + 0.5 * draws[i][j]

                # Expected score based on current ratings
                diff = (ratings[i] - ratings[j]) / 400.0
                expected_i = 1.0 / (1.0 + 10 ** (-diff))

                numerator += score_i
                denominator += total_games * expected_i

            if denominator > 0:
                # Update formula for Bradley-Terry MM algorithm
                new_ratings[i] = ratings[i] + 400.0 * math.log10(max(0.001, numerator / denominator))
            else:
                new_ratings[i] = ratings[i]

        ratings = new_ratings

    # Convert to dict and center around 1500
    if ratings:
        avg_rating = sum(ratings) / len(ratings)
        return {engine_ids[i]: r - avg_rating + 1500 for i, r in enumerate(ratings)}

    return {eid: 1500.0 for eid in engine_ids}


def calculate_all_ratings(min_time_ms: int, max_time_ms: int, hostname: str | None, engine_type: str | None):
    """
    Calculate Elo, BayesElo, and Ordo ratings for all engines matching the filter.

    This performs a full recalculation (not incremental) for all three rating systems.

    Args:
        min_time_ms: Minimum time per move in milliseconds
        max_time_ms: Maximum time per move in milliseconds
        hostname: Hostname filter (None = any host)
        engine_type: Engine type filter (None = all, 'rusty' = v*, 'stockfish' = sf*)

    Returns:
        dict: {engine_id: {'elo': float, 'bayes_elo': float, 'ordo': float, 'games_played': int}}
    """
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    # Get all game results for this filter
    game_results = get_game_results_for_filter(min_time_ms, max_time_ms, hostname, engine_type)

    # Get engine IDs that match the filter
    engine_ids = []
    for engine in Engine.query.filter(Engine.active == True):
        if engine_matches_type_filter(engine.name, engine_type):
            engine_ids.append(engine.id)

    # Count games per engine
    games_count = {eid: 0 for eid in engine_ids}
    for white_id, black_id, _, _ in game_results:
        if white_id in games_count:
            games_count[white_id] += 1
        if black_id in games_count:
            games_count[black_id] += 1

    # Calculate all three rating systems
    # 1. Sequential Elo (standard calculation)
    elo_ratings = calculate_sequential_elo(game_results, engine_ids)

    # 2. BayesElo
    bayes_ratings = calculate_bayeselo(game_results, engine_ids)

    # 3. Ordo
    ordo_ratings = calculate_ordo(game_results, engine_ids)

    # Combine results
    results = {}
    for eid in engine_ids:
        results[eid] = {
            'elo': elo_ratings.get(eid, 1500.0),
            'bayes_elo': bayes_ratings.get(eid, 1500.0),
            'ordo': ordo_ratings.get(eid, 1500.0),
            'games_played': games_count.get(eid, 0)
        }

    return results


def calculate_sequential_elo(game_results, engine_ids):
    """
    Calculate standard Elo ratings sequentially (same as incremental but from scratch).

    Args:
        game_results: List of (white_id, black_id, white_score, black_score)
        engine_ids: List of engine IDs to rate

    Returns:
        dict: {engine_id: elo_rating}
    """
    # Get initial ELOs
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    ratings = {}
    games_count = {}

    for engine in Engine.query.filter(Engine.id.in_(engine_ids)):
        ratings[engine.id] = float(engine.initial_elo or DEFAULT_ELO)
        games_count[engine.id] = 0

    # Process games sequentially
    for white_id, black_id, white_score, black_score in game_results:
        if white_id not in ratings or black_id not in ratings:
            continue

        white_elo = ratings[white_id]
        black_elo = ratings[black_id]
        white_games = games_count[white_id]
        black_games = games_count[black_id]

        # Expected scores
        white_expected = 1 / (1 + 10 ** ((black_elo - white_elo) / 400))
        black_expected = 1 - white_expected

        # K-factors
        white_k = K_FACTOR_PROVISIONAL if white_games < PROVISIONAL_GAMES else K_FACTOR_ESTABLISHED
        black_k = K_FACTOR_PROVISIONAL if black_games < PROVISIONAL_GAMES else K_FACTOR_ESTABLISHED

        # Update ratings
        ratings[white_id] += white_k * (float(white_score) - white_expected)
        ratings[black_id] += black_k * (float(black_score) - black_expected)
        games_count[white_id] += 1
        games_count[black_id] += 1

    return ratings


def recalculate_all_and_store(min_time_ms: int, max_time_ms: int, hostname: str | None, engine_type: str | None):
    """
    Recalculate all rating systems and store in database.

    This is called when Force Recalculate is clicked.

    Args:
        min_time_ms: Minimum time per move in milliseconds
        max_time_ms: Maximum time per move in milliseconds
        hostname: Hostname filter (None = any host)
        engine_type: Engine type filter (None = all)

    Returns:
        dict: The calculated ratings
    """
    db = get_db()
    Engine, Game, EloFilterCache, EloFilterRating = get_models()

    # Get or create filter cache
    cache = get_or_create_filter_cache(min_time_ms, max_time_ms, hostname, engine_type)

    # Calculate all ratings
    all_ratings = calculate_all_ratings(min_time_ms, max_time_ms, hostname, engine_type)

    # Get the last game ID for this filter
    from sqlalchemy.orm import aliased
    WhiteEngine = aliased(Engine)
    BlackEngine = aliased(Engine)

    query = db.session.query(func.max(Game.id)).join(
        WhiteEngine, Game.white_engine_id == WhiteEngine.id
    ).join(
        BlackEngine, Game.black_engine_id == BlackEngine.id
    ).filter(Game.is_rated == True)

    # Apply filters
    query = query.filter(
        (Game.time_per_move_ms >= min_time_ms) | (Game.time_per_move_ms.is_(None))
    )
    query = query.filter(
        (Game.time_per_move_ms <= max_time_ms) | (Game.time_per_move_ms.is_(None))
    )
    if hostname is not None:
        query = query.filter(Game.hostname == hostname)
    if engine_type == 'rusty':
        query = query.filter(WhiteEngine.name.like('v%'))
        query = query.filter(BlackEngine.name.like('v%'))
    elif engine_type == 'stockfish':
        query = query.filter(WhiteEngine.name.like('sf%'))
        query = query.filter(BlackEngine.name.like('sf%'))

    last_game_id = query.scalar() or 0

    # Update cache
    cache.last_game_id = last_game_id

    # Store ratings
    for engine_id, rating_data in all_ratings.items():
        existing = EloFilterRating.query.filter_by(
            filter_id=cache.id, engine_id=engine_id
        ).first()

        if existing:
            existing.elo = rating_data['elo']
            existing.bayes_elo = rating_data['bayes_elo']
            existing.ordo = rating_data['ordo']
            existing.games_played = rating_data['games_played']
        else:
            db.session.add(EloFilterRating(
                filter_id=cache.id,
                engine_id=engine_id,
                elo=rating_data['elo'],
                bayes_elo=rating_data['bayes_elo'],
                ordo=rating_data['ordo'],
                games_played=rating_data['games_played']
            ))

    db.session.commit()

    return all_ratings
