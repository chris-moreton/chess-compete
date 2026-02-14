#!/usr/bin/env python3
"""
Flask application for chess engine competition dashboard.
"""

import os
from pathlib import Path
from flask import Flask
from dotenv import load_dotenv

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
    app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-key-change-in-production')

    # Initialize extensions
    db.init_app(app)

    # Register routes (import here to avoid circular imports)
    with app.app_context():
        from web import routes
        routes.register_routes(app)

        # Lightweight schema migration for columns added by new code
        _migrate_schema(db)

    return app


def _migrate_schema(db):
    """Add any missing columns that new code depends on. Idempotent."""
    migrations = [
        ("spsa_iterations", "llm_report", "ALTER TABLE spsa_iterations ADD COLUMN llm_report TEXT"),
    ]
    for table, column, ddl in migrations:
        try:
            db.session.execute(db.text(f"SELECT {column} FROM {table} LIMIT 1"))
            db.session.commit()
        except Exception:
            db.session.rollback()
            try:
                db.session.execute(db.text(ddl))
                db.session.commit()
            except Exception:
                db.session.rollback()


if __name__ == '__main__':
    app = create_app()
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)
