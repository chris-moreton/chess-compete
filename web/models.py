"""
SQLAlchemy models for chess engine competition tracking.
"""

from datetime import datetime
from web.database import db


class Engine(db.Model):
    """Chess engine metadata."""
    __tablename__ = 'engines'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), unique=True, nullable=False)
    binary_path = db.Column(db.String(500))
    active = db.Column(db.Boolean, default=True)
    initial_elo = db.Column(db.Integer, default=1500)
    uci_options = db.Column(db.JSON)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    def __repr__(self):
        return f'<Engine {self.name}>'


class Game(db.Model):
    """Individual game result."""
    __tablename__ = 'games'

    id = db.Column(db.Integer, primary_key=True)
    white_engine_id = db.Column(db.Integer, db.ForeignKey('engines.id'), nullable=False)
    black_engine_id = db.Column(db.Integer, db.ForeignKey('engines.id'), nullable=False)
    result = db.Column(db.String(10), nullable=False)  # '1-0', '0-1', '1/2-1/2', '*'
    white_score = db.Column(db.Numeric(2, 1), nullable=False)  # 1.0, 0.5, or 0.0
    black_score = db.Column(db.Numeric(2, 1), nullable=False)
    date_played = db.Column(db.Date, nullable=False)
    time_control = db.Column(db.String(50))
    time_per_move_ms = db.Column(db.Integer)  # Time per move in milliseconds
    hostname = db.Column(db.String(100))  # Machine that played the game
    opening_name = db.Column(db.String(100))
    opening_fen = db.Column(db.Text)
    pgn = db.Column(db.Text)  # Full PGN content of the game
    is_rated = db.Column(db.Boolean, default=True)  # False for EPD test games
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    white_engine = db.relationship('Engine', foreign_keys=[white_engine_id], backref='games_as_white')
    black_engine = db.relationship('Engine', foreign_keys=[black_engine_id], backref='games_as_black')

    def __repr__(self):
        return f'<Game {self.id}: {self.result}>'


class EloFilterCache(db.Model):
    """Cache entry for a specific filter combination."""
    __tablename__ = 'elo_filter_cache'

    id = db.Column(db.Integer, primary_key=True)
    min_time_ms = db.Column(db.Integer, nullable=False, default=0)
    max_time_ms = db.Column(db.Integer, nullable=False, default=999999999)
    hostname = db.Column(db.String(100))  # NULL = any host
    engine_type = db.Column(db.String(20))  # NULL = all, 'rusty' = v*, 'stockfish' = sf*
    last_game_id = db.Column(db.Integer, nullable=False, default=0)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('min_time_ms', 'max_time_ms', 'hostname', 'engine_type'),
    )

    def __repr__(self):
        return f'<EloFilterCache {self.min_time_ms}-{self.max_time_ms}ms, host={self.hostname}, engine={self.engine_type}>'


class EloFilterRating(db.Model):
    """Cached ratings for an engine under a specific filter."""
    __tablename__ = 'elo_filter_ratings'

    id = db.Column(db.Integer, primary_key=True)
    filter_id = db.Column(db.Integer, db.ForeignKey('elo_filter_cache.id', ondelete='CASCADE'), nullable=False)
    engine_id = db.Column(db.Integer, db.ForeignKey('engines.id', ondelete='CASCADE'), nullable=False)
    elo = db.Column(db.Numeric(7, 2), nullable=False)
    bayes_elo = db.Column(db.Numeric(7, 2))  # NULL until Force Recalculate
    ordo = db.Column(db.Numeric(7, 2))  # NULL until Force Recalculate
    games_played = db.Column(db.Integer, nullable=False, default=0)

    # Relationships
    filter_cache = db.relationship('EloFilterCache', backref='ratings')
    engine = db.relationship('Engine', backref='filter_ratings')

    __table_args__ = (
        db.UniqueConstraint('filter_id', 'engine_id'),
    )

    def __repr__(self):
        return f'<EloFilterRating engine={self.engine_id}, elo={self.elo}>'


class Cup(db.Model):
    """Knockout cup competition."""
    __tablename__ = 'cups'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    status = db.Column(db.String(20), nullable=False, default='in_progress')  # 'in_progress', 'completed'
    num_participants = db.Column(db.Integer, nullable=False)
    games_per_match = db.Column(db.Integer, nullable=False, default=10)
    time_per_move_ms = db.Column(db.Integer)  # NULL if using time range
    time_low_ms = db.Column(db.Integer)  # For time range mode
    time_high_ms = db.Column(db.Integer)  # For time range mode
    winner_engine_id = db.Column(db.Integer, db.ForeignKey('engines.id'))
    hostname = db.Column(db.String(100))  # Machine running the cup
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    # Relationships
    winner_engine = db.relationship('Engine', foreign_keys=[winner_engine_id])
    rounds = db.relationship('CupRound', backref='cup', order_by='CupRound.round_number',
                             cascade='all, delete-orphan')

    def __repr__(self):
        return f'<Cup {self.name}>'


class CupRound(db.Model):
    """A round within a cup competition."""
    __tablename__ = 'cup_rounds'

    id = db.Column(db.Integer, primary_key=True)
    cup_id = db.Column(db.Integer, db.ForeignKey('cups.id', ondelete='CASCADE'), nullable=False)
    round_number = db.Column(db.Integer, nullable=False)  # 1 = first round, 2 = second, etc.
    round_name = db.Column(db.String(50))  # "Round of 16", "Quarterfinals", "Semifinals", "Final"
    status = db.Column(db.String(20), nullable=False, default='pending')  # 'pending', 'in_progress', 'completed'

    # Relationships
    matches = db.relationship('CupMatch', backref='cup_round', order_by='CupMatch.match_order',
                              cascade='all, delete-orphan')

    __table_args__ = (
        db.UniqueConstraint('cup_id', 'round_number'),
    )

    def __repr__(self):
        return f'<CupRound {self.round_name}>'


class CupMatch(db.Model):
    """A match between two engines in a cup round."""
    __tablename__ = 'cup_matches'

    id = db.Column(db.Integer, primary_key=True)
    round_id = db.Column(db.Integer, db.ForeignKey('cup_rounds.id', ondelete='CASCADE'), nullable=False)
    match_order = db.Column(db.Integer, nullable=False)  # Position in bracket (1, 2, 3, 4...)

    # Participants (engine2 can be NULL for byes)
    engine1_id = db.Column(db.Integer, db.ForeignKey('engines.id'), nullable=False)
    engine2_id = db.Column(db.Integer, db.ForeignKey('engines.id'))  # NULL = bye
    engine1_seed = db.Column(db.Integer)
    engine2_seed = db.Column(db.Integer)

    # Results
    engine1_points = db.Column(db.Numeric(4, 1), default=0)  # e.g., 5.5
    engine2_points = db.Column(db.Numeric(4, 1), default=0)
    games_played = db.Column(db.Integer, default=0)
    winner_engine_id = db.Column(db.Integer, db.ForeignKey('engines.id'))
    status = db.Column(db.String(20), nullable=False, default='pending')  # 'pending', 'bye', 'in_progress', 'completed'
    is_tiebreaker = db.Column(db.Boolean, default=False)
    decided_by_coin_flip = db.Column(db.Boolean, default=False)

    # Relationships
    engine1 = db.relationship('Engine', foreign_keys=[engine1_id])
    engine2 = db.relationship('Engine', foreign_keys=[engine2_id])
    winner = db.relationship('Engine', foreign_keys=[winner_engine_id])

    __table_args__ = (
        db.UniqueConstraint('round_id', 'match_order'),
    )

    def __repr__(self):
        e1 = self.engine1.name if self.engine1 else 'TBD'
        e2 = self.engine2.name if self.engine2 else 'BYE'
        return f'<CupMatch {e1} vs {e2}>'


class EpdTestRun(db.Model):
    """Metadata for an EPD solve test run."""
    __tablename__ = 'epd_test_runs'

    id = db.Column(db.Integer, primary_key=True)
    epd_file = db.Column(db.String(255), nullable=False)  # e.g., "eet.epd"
    total_positions = db.Column(db.Integer, nullable=False)
    timeout_seconds = db.Column(db.Float, nullable=False)
    score_tolerance = db.Column(db.Integer, nullable=False)
    hostname = db.Column(db.String(255))
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    # Relationships
    results = db.relationship('EpdTestResult', backref='run', cascade='all, delete-orphan')

    def __repr__(self):
        return f'<EpdTestRun {self.epd_file} @ {self.created_at}>'


class EpdTestResult(db.Model):
    """Result for a single position in an EPD solve test."""
    __tablename__ = 'epd_test_results'

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer, db.ForeignKey('epd_test_runs.id', ondelete='CASCADE'), nullable=False)
    engine_id = db.Column(db.Integer, db.ForeignKey('engines.id', ondelete='CASCADE'), nullable=False)
    position_id = db.Column(db.String(255), nullable=False)  # e.g., "E_E_T 001"
    position_index = db.Column(db.Integer, nullable=False)  # 1-based position in EPD file
    fen = db.Column(db.Text, nullable=False)
    test_type = db.Column(db.String(10), nullable=False)  # 'bm' or 'am'
    expected_moves = db.Column(db.String(255), nullable=False)  # comma-separated
    solved = db.Column(db.Boolean, nullable=False)
    move_found = db.Column(db.String(20))  # The move the engine played
    solve_time_ms = db.Column(db.Integer)  # NULL if failed
    final_depth = db.Column(db.Integer)
    score_cp = db.Column(db.Integer)  # Centipawn eval (NULL if mate)
    score_mate = db.Column(db.Integer)  # Mate in N (NULL if not mate)
    score_valid = db.Column(db.Boolean)  # NULL if no ce check
    timed_out = db.Column(db.Boolean, nullable=False, default=False)

    # Relationships
    engine = db.relationship('Engine', backref='epd_test_results')

    __table_args__ = (
        db.Index('idx_epd_results_run', 'run_id'),
        db.Index('idx_epd_results_engine', 'engine_id'),
        db.Index('idx_epd_results_position', 'position_id'),
    )

    def __repr__(self):
        status = 'SOLVED' if self.solved else 'FAILED'
        return f'<EpdTestResult {self.position_id}: {status}>'


class SpsaIteration(db.Model):
    """A single SPSA tuning iteration tracking games between perturbed engine pairs."""
    __tablename__ = 'spsa_iterations'

    id = db.Column(db.Integer, primary_key=True)
    iteration_number = db.Column(db.Integer, nullable=False)

    # Engine binaries (paths to shared location, not registered engines)
    plus_engine_path = db.Column(db.String(500), nullable=False)
    minus_engine_path = db.Column(db.String(500), nullable=False)
    base_engine_path = db.Column(db.String(500))  # Unperturbed base params engine
    ref_engine_path = db.Column(db.String(500))   # Reference engine (e.g., Stockfish)

    # Time control (consistent with other compete modes: timelow/timehigh)
    timelow_ms = db.Column(db.Integer, nullable=False)   # e.g., 250 (0.25s)
    timehigh_ms = db.Column(db.Integer, nullable=False)  # e.g., 1000 (1.0s)

    # Game tracking (workers increment these atomically)
    target_games = db.Column(db.Integer, nullable=False, default=150)
    games_played = db.Column(db.Integer, nullable=False, default=0)
    plus_wins = db.Column(db.Integer, nullable=False, default=0)
    minus_wins = db.Column(db.Integer, nullable=False, default=0)
    draws = db.Column(db.Integer, nullable=False, default=0)

    # Reference game tracking (base engine vs Stockfish)
    ref_games_played = db.Column(db.Integer, nullable=False, default=0)
    ref_wins = db.Column(db.Integer, nullable=False, default=0)
    ref_losses = db.Column(db.Integer, nullable=False, default=0)
    ref_draws = db.Column(db.Integer, nullable=False, default=0)

    # Status: pending (waiting for workers), in_progress, complete
    status = db.Column(db.String(20), nullable=False, default='pending')

    # Parameter snapshot (JSON for reproducibility and analysis)
    base_parameters = db.Column(db.JSON)       # θ values before perturbation
    plus_parameters = db.Column(db.JSON)       # θ + δ values
    minus_parameters = db.Column(db.JSON)      # θ - δ values
    perturbation_signs = db.Column(db.JSON)    # +1 or -1 for each parameter

    # Results (filled when complete)
    gradient_estimate = db.Column(db.JSON)     # Calculated gradient per parameter
    elo_diff = db.Column(db.Numeric(7, 2))     # Plus vs minus Elo difference
    ref_elo_estimate = db.Column(db.Numeric(7, 2))  # Estimated Elo vs reference engine

    # Timestamps
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    completed_at = db.Column(db.DateTime)

    __table_args__ = (
        db.Index('idx_spsa_iteration_number', 'iteration_number'),
        db.Index('idx_spsa_status', 'status'),
    )

    def __repr__(self):
        return f'<SpsaIteration {self.iteration_number}: {self.status} ({self.games_played}/{self.target_games})>'
