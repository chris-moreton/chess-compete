# Chess Engine Competition Framework

A comprehensive engine vs engine testing harness with Elo tracking, automatic engine management, and multiple competition modes.

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Setup](#setup)
3. [Engine Management](#engine-management)
4. [Competition Modes](#competition-modes)
5. [Command Reference](#command-reference)
6. [Elo Rating System](#elo-rating-system)
7. [Web Dashboard](#web-dashboard)
8. [Development](#development)
9. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Initialize and enable an engine (downloads automatically)
python -m compete --init rusty v1.0.17
python -m compete --init stockfish latest

# Run a head-to-head match
python -m compete v1.0.17 sf-2400 --games 100 --time 1.0

# Random mode (continuous random pairings from active engines)
python -m compete --random --games 100 --time 0.5

# Cup mode (knockout tournament)
python -m compete --cup --games 10 --time 1.0
```

---

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL database

### Step 1: Clone and Install Dependencies

```bash
git clone https://github.com/chris-moreton/chess-compete.git
cd chess-compete
python -m venv .venv

# Linux/macOS
source .venv/bin/activate

# Windows PowerShell
& .\.venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### Step 2: Database Setup

Create a PostgreSQL database and run this schema:

```sql
CREATE TABLE engines (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    binary_path VARCHAR(500),
    active BOOLEAN DEFAULT TRUE,
    initial_elo INTEGER DEFAULT 1500,
    uci_options JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE games (
    id SERIAL PRIMARY KEY,
    white_engine_id INTEGER NOT NULL REFERENCES engines(id),
    black_engine_id INTEGER NOT NULL REFERENCES engines(id),
    result VARCHAR(10) NOT NULL,
    white_score NUMERIC(2,1) NOT NULL,
    black_score NUMERIC(2,1) NOT NULL,
    date_played DATE NOT NULL,
    time_control VARCHAR(50),
    opening_name VARCHAR(100),
    opening_fen TEXT,
    pgn TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE elo_ratings (
    id SERIAL PRIMARY KEY,
    engine_id INTEGER NOT NULL UNIQUE REFERENCES engines(id),
    elo NUMERIC(7,2) NOT NULL,
    games_played INTEGER DEFAULT 0,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_games_white_engine ON games(white_engine_id);
CREATE INDEX idx_games_black_engine ON games(black_engine_id);
CREATE INDEX idx_games_date ON games(date_played);
CREATE INDEX idx_engines_active ON engines(active);
```

### Step 3: Configure Environment

Create a `.env` file:

```
DATABASE_URL=postgresql://username:password@hostname:5432/database_name
```

### Step 4: Initialize Engines

Use `--init` to download and enable engines:

```bash
# Download and enable Stockfish (enables all sf-* variants)
python -m compete --init stockfish latest

# Download and enable Rusty Rival
python -m compete --init rusty v1.0.17

# Download and enable Java Rival
python -m compete --init java 38
```

---

## Engine Management

### Automatic Engine Initialization

**Engines are automatically downloaded when needed.** When you start a competition:

- **Random/Gauntlet/Cup modes**: All matching engines from the database are checked. Missing engines are automatically downloaded.
- **Other modes**: Only the specified engines are checked and downloaded if missing.

This means you can enable an engine on one machine, and other machines will automatically download it when they start a competition.

### The --init Command

`--init` downloads an engine and enables it in the database:

```bash
# Stockfish (enables all sf-* variants: sf-1400, sf-1600, ..., sf-3000, sf-full)
python -m compete --init stockfish latest

# Rusty Rival
python -m compete --init rusty v1.0.17

# Java Rival
python -m compete --init java 38
```

### Enabling and Disabling Engines

```bash
# List all engines with their status
python -m compete --list

# Disable engines (won't be selected in random/gauntlet/cup mode)
python -m compete --disable sf-1400 sf-full

# Enable engines
python -m compete --enable java-rival-38 v1.0.17
```

### Directory Structure

```
chess-compete/
  engines/
    stockfish/
      stockfish-windows-x86-64-avx2.exe   # Auto-downloaded
    v1.0.17/
      rusty-rival-v1.0.17-windows-x86_64.exe
    java-rival-38.0.0/
      rivalchess-v38.0.0.jar
  openings/
    eet.epd    # EPD test files
  results/
    competitions/   # PGN output
```

### Engine Discovery

The script automatically discovers engines:

1. **Stockfish**: `engines/stockfish/` - creates virtual engines `sf-1400` through `sf-3000` and `sf-full`
2. **Rusty Rival**: `engines/v*/` directories
3. **Java Rival**: `engines/java-rival-*/` directories with `.jar` files

---

## Competition Modes

### Head-to-Head Match (2 engines)

```bash
python -m compete v1.0.17 sf-2400 --games 100 --time 1.0
```

- Each opening played twice (once per side)
- Uses built-in opening book (250 positions)
- Elo ratings **are updated**

### Round-Robin League (3+ engines)

```bash
python -m compete v1.0.17 sf-2400 sf-2600 sf-2800 --games 50 --time 1.0
```

- All possible pairings played
- Shows league table after each game
- Elo ratings **are updated**

### Gauntlet Mode

Test one engine against all other active engines:

```bash
# Basic gauntlet
python -m compete v1.0.17 --gauntlet --games 50 --time 0.5

# Gauntlet against only Rusty engines
python -m compete v1.0.17 --gauntlet --games 50 --enginetype rusty

# Gauntlet including inactive engines
python -m compete v1.0.17 --gauntlet --games 50 --includeinactive

# Gauntlet with random time per round
python -m compete v1.0.17 --gauntlet --games 50 --timelow 0.5 --timehigh 2.0
```

- Tests one engine against all matching **active** engines (or all if `--includeinactive`)
- Elo ratings **are updated**

### Random Mode

Continuous random pairings:

```bash
# Basic random mode
python -m compete --random --games 100 --time 0.5

# Weighted: favor engines with fewer games
python -m compete --random --weighted --games 100 --time 0.5

# Random with only Stockfish engines
python -m compete --random --games 100 --enginetype stockfish

# Random with only Rusty engines including inactive
python -m compete --random --games 100 --enginetype rusty --includeinactive

# Random with variable time (0.5-2.0 seconds per move)
python -m compete --random --games 100 --timelow 0.5 --timehigh 2.0
```

- Continuous random pairings from matching engines
- Re-checks engine list before each match (live enable/disable)
- Elo ratings **are updated**

### Cup Mode (Knockout Tournament)

Seeded knockout tournament with bracket visualization:

```bash
# Basic cup with all active engines
python -m compete --cup --games 10 --time 1.0

# Cup with custom name
python -m compete --cup --games 10 --time 1.0 --cup-name "Winter Championship 2026"

# Cup limited to top 8 engines by Ordo rating
python -m compete --cup --cup-engines 8 --games 10 --time 1.0

# Cup with only Rusty engines
python -m compete --cup --games 10 --enginetype rusty

# Cup with only Stockfish engines including inactive
python -m compete --cup --games 10 --enginetype stockfish --includeinactive

# Cup with random time per game pair (0.5-2.0 seconds)
python -m compete --cup --games 10 --timelow 0.5 --timehigh 2.0
```

Features:
- Seeded brackets based on Ordo rating (top seeds get byes for non-power-of-2)
- `--games N` = number of game **pairs** (total games = N × 2)
- Tiebreaker: play additional pairs until decisive (max 10 pairs), then coin flip
- All games saved to database as rated games
- View brackets on web dashboard at `/cups`

### EPD Mode

```bash
python -m compete v1.0.17 sf-2800 --epd eet.epd --time 1.0
```

- Play through positions from an EPD file
- Elo ratings **are NOT updated**

---

## Command Reference

### Competition Mode Options

| Option | Description |
|--------|-------------|
| `--random` | Random mode: continuous random pairings |
| `--weighted` | With `--random`: favor engines with fewer games |
| `--gauntlet` | Gauntlet mode: test one engine against all others |
| `--cup` | Cup mode: knockout tournament with seeded brackets |
| `--epd FILE` | EPD mode: play through positions from file |

### Engine Filter Options

| Option | Description |
|--------|-------------|
| `--enginetype TYPE` | Filter engines by type: `rusty` or `stockfish` |
| `--includeinactive` | Include inactive engines (for random/gauntlet/cup modes) |

### Cup Mode Options

| Option | Description |
|--------|-------------|
| `--cup-engines N` | Limit cup to top N engines by Ordo rating |
| `--cup-name NAME` | Custom name for the cup competition |

### Time Control Options

| Option | Description |
|--------|-------------|
| `--time T`, `-t T` | Fixed time per move in seconds (default: 1.0) |
| `--timelow T` | Minimum time per move (use with `--timehigh`) |
| `--timehigh T` | Maximum time per move (use with `--timelow`) |

When using `--timelow` and `--timehigh`, a random time is selected for each match/round.

### Game Options

| Option | Description |
|--------|-------------|
| `--games N`, `-g N` | Number of games/rounds/pairs (default: 100) |
| `--no-book` | Disable opening book (start from initial position) |

### Engine Management Options

| Option | Description |
|--------|-------------|
| `--list` | List all engines with their active status and Elo |
| `--enable ENGINE...` | Enable one or more engines |
| `--disable ENGINE...` | Disable one or more engines |
| `--init TYPE VERSION [ELO]` | Download and enable an engine |

### Engine Name Shorthand

| Shorthand | Resolves to |
|-----------|-------------|
| `v1` | `v001-baseline` |
| `v12` | `v012-my-test-version` |
| `v1.0.17` | `v1.0.17` |
| `sf-2400` | Stockfish at 2400 Elo |

---

## Elo Rating System

- **Initial rating**: From `initial_elo` column (default: 1500)
- **K-factor**: 40 for provisional (<30 games), 20 for established
- **Provisional**: Marked with `?` until 30+ games played

| Mode | Updates Elo? |
|------|--------------|
| Head-to-head | Yes |
| Round-robin | Yes |
| Gauntlet | Yes |
| Random | Yes |
| Cup | Yes |
| EPD | **No** |

---

## Web Dashboard

A Flask-based web interface displays statistics, Elo ratings, and cup brackets.

```bash
# Linux/macOS
source .venv/bin/activate
flask --app web.app run

# Windows
.venv\Scripts\activate.bat
flask --app web.app run
```

Access at `http://localhost:5000`

### Dashboard Pages

| URL | Description |
|-----|-------------|
| `/` | Main dashboard with H2H grid and Elo ratings |
| `/engines` | Engine management (enable/disable) |
| `/engine/<name>` | Detailed H2H stats for a specific engine |
| `/cups` | List of all cup competitions |
| `/cup/<id>` | Bracket view for a specific cup |

### Dashboard Features

- Filter by time range, hostname, and engine type
- Sort by Elo, BayesElo, or Ordo ratings
- Color-coded performance vs expected results
- Force recalculate ratings button
- Auto-refresh every 30 seconds

---

## Development

### Running Tests

```bash
# Run all tests
python -m pytest tests/

# Run tests with verbose output
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/test_game.py -v
```

### Project Structure

```
compete/
├── __init__.py        # Package exports
├── __main__.py        # Entry point (python -m compete)
├── cli.py             # Argument parsing, main()
├── constants.py       # K_FACTOR_*, DEFAULT_ELO, DB_* constants
├── database.py        # DB operations, Elo tracking
├── engine_manager.py  # Engine init, discovery, paths
├── game.py            # play_game(), calculate_elo_difference()
├── competitions.py    # run_match, run_league, run_gauntlet, run_random, run_epd
├── cup.py             # run_cup() knockout tournament
└── openings.py        # OPENING_BOOK data, load_epd_positions()

web/
├── app.py             # Flask application factory
├── routes.py          # Web routes
├── queries.py         # Database queries for dashboard
├── models.py          # SQLAlchemy models
├── database.py        # Database connection
├── templates/         # Jinja2 templates
└── static/            # CSS, JS assets

tests/
├── test_constants.py
├── test_openings.py
├── test_database.py
├── test_engine_manager.py
├── test_game.py
├── test_cup.py
└── test_competitions.py
```

---

## Troubleshooting

### Engine Not Found

If auto-download fails:
1. Check internet connection
2. Check the engine exists in GitHub releases
3. Try manual download to `engines/` directory

### Database Connection Error

Verify your `.env` file has the correct `DATABASE_URL`.

### Stockfish Not Working

Ensure the binary is executable:
```bash
chmod +x engines/stockfish/stockfish-*
```

### EPD File Not Found

Provide the full path or place in `openings/` directory.

### Cup Shows Wrong Number of Games

Remember that `--games N` in cup mode means N **pairs** of games (total = N × 2).

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (required) |
| `ENGINES_DIR` | Override engines directory location (optional) |
| `COMPUTER_NAME` | Override hostname in PGN output (optional) |
