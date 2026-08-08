#!/usr/bin/env python3
"""
Flask application for chess engine competition dashboard.
"""

import os
from pathlib import Path
from flask import Flask
from dotenv import load_dotenv
from sqlalchemy.pool import NullPool

# Load environment variables from project root
project_root = Path(__file__).parent.parent
load_dotenv(project_root / '.env')

# Import db from separate module to avoid circular imports
from web.database import db


def create_app():
    """Application factory."""
    app = Flask(__name__)

    # Configuration
    # Convert postgresql:// to postgresql+psycopg:// for psycopg3 compatibility
    db_url = os.getenv('DATABASE_URL', '')
    if db_url.startswith('postgresql://'):
        db_url = db_url.replace('postgresql://', 'postgresql+psycopg://', 1)
    app.config['SQLALCHEMY_DATABASE_URI'] = db_url
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Connect on demand, hold nothing idle.
    #
    # Without this, SQLAlchemy's default QueuePool keeps up to 15 connections
    # (pool_size 5 + max_overflow 10) alive per engine — and this factory is
    # called from 21 places, each building its own engine with its own pool,
    # none of them disposed. During SPSA that exhausted the server's connection
    # slots: the master logged repeated "too many clients already" retries and
    # got as far as attempt 6 of 10 before recovering.
    #
    # Pooling amortises connection setup for frequent queries. Our workload is
    # the opposite — a game runs for minutes, then writes one row — so the pool
    # bought nothing and cost the one resource we were short of.
    app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {'poolclass': NullPool}
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')
    app.config['MAX_CONTENT_LENGTH'] = 50 * 1024 * 1024  # 50MB for PGN uploads

    # Initialize extensions
    db.init_app(app)

    # Register routes (import here to avoid circular imports)
    with app.app_context():
        from web import routes
        routes.register_routes(app)

    return app


if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
