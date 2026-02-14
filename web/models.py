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
    points_earned = db.Column(db.Integer)  # STS points (0-10, NULL if not STS format)

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


class SpsaRun(db.Model):
    """A logical SPSA tuning run, grouping iterations together."""
    __tablename__ = 'spsa_runs'

    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.Text)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    is_active = db.Column(db.Boolean, default=True)  # Only one should be active at a time
    effective_iteration_offset = db.Column(db.Integer, nullable=False, default=0)
    timelow = db.Column(db.Float, nullable=False, default=0.1)   # Seconds per move (low)
    timehigh = db.Column(db.Float, nullable=False, default=0.2)  # Seconds per move (high)
    games_per_iteration = db.Column(db.Integer, nullable=False, default=500)
    max_iterations = db.Column(db.Integer, nullable=False, default=1500)
    spsa_a = db.Column(db.Float, nullable=False, default=1.0)
    spsa_c = db.Column(db.Float, nullable=False, default=1.0)
    spsa_big_a = db.Column(db.Float, nullable=False, default=50)
    spsa_alpha = db.Column(db.Float, nullable=False, default=0.602)
    spsa_gamma = db.Column(db.Float, nullable=False, default=0.101)
    max_elo_diff = db.Column(db.Float, nullable=False, default=100.0)
    max_gradient_factor = db.Column(db.Float, nullable=False, default=3.0)
    ref_enabled = db.Column(db.Boolean, nullable=False, default=True)
    ref_ratio = db.Column(db.Float, nullable=False, default=0.25)
    active_groups = db.Column(db.JSON)  # JSON array of group names, null = all groups active

    # Relationships
    iterations = db.relationship('SpsaIteration', backref='run', lazy='dynamic')

    def __repr__(self):
        return f'<SpsaRun {self.id}: {self.name}>'


class SpsaIteration(db.Model):
    """A single SPSA tuning iteration tracking games between perturbed engine pairs."""
    __tablename__ = 'spsa_iterations'

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer, db.ForeignKey('spsa_runs.id'), nullable=True)
    iteration_number = db.Column(db.Integer, nullable=False)
    effective_iteration = db.Column(db.Integer)  # Used for learning rate calculation (can be reset)

    # Reference engine path (e.g., Stockfish) — stored by master for auditing
    ref_engine_path = db.Column(db.String(500))

    # Time control (consistent with other compete modes: timelow/timehigh)
    timelow_ms = db.Column(db.Integer, nullable=False)   # e.g., 250 (0.25s)
    timehigh_ms = db.Column(db.Integer, nullable=False)  # e.g., 1000 (1.0s)

    # Game tracking (workers increment these atomically)
    target_games = db.Column(db.Integer, nullable=False, default=150)
    games_played = db.Column(db.Integer, nullable=False, default=0)
    plus_wins = db.Column(db.Integer, nullable=False, default=0)
    minus_wins = db.Column(db.Integer, nullable=False, default=0)
    draws = db.Column(db.Integer, nullable=False, default=0)

    # Reference game tracking (new base engine vs Stockfish)
    ref_target_games = db.Column(db.Integer, nullable=False, default=100)
    ref_games_played = db.Column(db.Integer, nullable=False, default=0)
    ref_wins = db.Column(db.Integer, nullable=False, default=0)
    ref_losses = db.Column(db.Integer, nullable=False, default=0)
    ref_draws = db.Column(db.Integer, nullable=False, default=0)

    # Status: pending (SPSA phase), building (master building new base),
    #         ref_pending (ref phase), complete
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

    # LLM-generated analysis (every N iterations)
    llm_report = db.Column(db.Text)

    __table_args__ = (
        db.Index('idx_spsa_iteration_number', 'iteration_number'),
        db.Index('idx_spsa_status', 'status'),
        db.Index('idx_spsa_run_id', 'run_id'),
        db.UniqueConstraint('run_id', 'iteration_number', name='uq_spsa_run_iteration'),
    )

    def __repr__(self):
        return f'<SpsaIteration {self.iteration_number}: {self.status} ({self.games_played}/{self.target_games})>'


class SpsaParam(db.Model):
    """A tunable parameter for an SPSA run, storing value and bounds."""
    __tablename__ = 'spsa_params'

    id = db.Column(db.Integer, primary_key=True)
    run_id = db.Column(db.Integer, db.ForeignKey('spsa_runs.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    value = db.Column(db.Float, nullable=False)
    min_value = db.Column(db.Float, nullable=False)
    max_value = db.Column(db.Float, nullable=False)
    step = db.Column(db.Float, nullable=False)
    group = db.Column(db.String(50), nullable=False, default='ungrouped')

    # Relationships
    run = db.relationship('SpsaRun', backref='params')

    __table_args__ = (
        db.UniqueConstraint('run_id', 'name', name='uq_spsa_param_run_name'),
    )

    def __repr__(self):
        return f'<SpsaParam {self.name}={self.value} (run={self.run_id})>'


class SpsaWorker(db.Model):
    """Summary of an SPSA worker's current state and lifetime stats."""
    __tablename__ = 'spsa_workers'

    id = db.Column(db.Integer, primary_key=True)
    worker_name = db.Column(db.String(100), unique=True, nullable=False)
    last_iteration_id = db.Column(db.Integer, db.ForeignKey('spsa_iterations.id'))
    last_phase = db.Column(db.String(20))  # 'spsa' or 'ref'
    total_games = db.Column(db.Integer, nullable=False, default=0)
    total_spsa_games = db.Column(db.Integer, nullable=False, default=0)
    total_ref_games = db.Column(db.Integer, nullable=False, default=0)
    avg_nps = db.Column(db.Integer)  # Rolling average NPS
    last_seen_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)
    first_seen_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    last_iteration = db.relationship('SpsaIteration', foreign_keys=[last_iteration_id])
    heartbeats = db.relationship('SpsaWorkerHeartbeat', backref='worker', lazy='dynamic',
                                 cascade='all, delete-orphan')

    def __repr__(self):
        return f'<SpsaWorker {self.worker_name}: {self.total_games} games>'


class SpsaWorkerHeartbeat(db.Model):
    """Individual heartbeat/report from an SPSA worker."""
    __tablename__ = 'spsa_worker_heartbeats'

    id = db.Column(db.Integer, primary_key=True)
    worker_id = db.Column(db.Integer, db.ForeignKey('spsa_workers.id', ondelete='CASCADE'), nullable=False)
    iteration_id = db.Column(db.Integer, db.ForeignKey('spsa_iterations.id'))
    phase = db.Column(db.String(20), nullable=False)  # 'spsa' or 'ref'
    games_reported = db.Column(db.Integer, nullable=False)
    avg_nps = db.Column(db.Integer)  # NPS for this batch
    created_at = db.Column(db.DateTime, nullable=False, default=datetime.utcnow)

    # Relationships
    iteration = db.relationship('SpsaIteration', foreign_keys=[iteration_id])

    __table_args__ = (
        db.Index('idx_spsa_worker_heartbeats_worker', 'worker_id'),
        db.Index('idx_spsa_worker_heartbeats_created', 'created_at'),
    )

    def __repr__(self):
        return f'<SpsaWorkerHeartbeat {self.worker_id}: {self.games_reported} games>'
