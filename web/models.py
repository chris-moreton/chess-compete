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
    """Cached ELO rating for an engine under a specific filter."""
    __tablename__ = 'elo_filter_ratings'

    id = db.Column(db.Integer, primary_key=True)
    filter_id = db.Column(db.Integer, db.ForeignKey('elo_filter_cache.id', ondelete='CASCADE'), nullable=False)
    engine_id = db.Column(db.Integer, db.ForeignKey('engines.id', ondelete='CASCADE'), nullable=False)
    elo = db.Column(db.Numeric(7, 2), nullable=False)
    games_played = db.Column(db.Integer, nullable=False, default=0)

    # Relationships
    filter_cache = db.relationship('EloFilterCache', backref='ratings')
    engine = db.relationship('Engine', backref='filter_ratings')

    __table_args__ = (
        db.UniqueConstraint('filter_id', 'engine_id'),
    )

    def __repr__(self):
        return f'<EloFilterRating engine={self.engine_id}, elo={self.elo}>'
