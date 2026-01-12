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
8. [Troubleshooting](#troubleshooting)

---

## Quick Start

```bash
# Initialize and enable an engine (downloads automatically)
python compete.py --init rusty v1.0.17
python compete.py --init stockfish latest

# Run a head-to-head match
python compete.py v1.0.17 sf-2400 --games 100 --time 1.0

# Random mode (continuous random pairings from active engines)
python compete.py --random --games 100 --time 0.5
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
python compete.py --init stockfish latest

# Download and enable Rusty Rival
python compete.py --init rusty v1.0.17

# Download and enable Java Rival
python compete.py --init java 38
```

---

## Engine Management

### Automatic Engine Initialization

**Engines are automatically downloaded when needed.** When you start a competition:

- **Random/Gauntlet modes**: All active engines from the database are checked. Missing engines are automatically downloaded.
- **Other modes**: Only the specified engines are checked and downloaded if missing.

This means you can enable an engine on one machine, and other machines will automatically download it when they start a competition.

### The --init Command

`--init` downloads an engine and enables it in the database:

```bash
# Stockfish (enables all sf-* variants: sf-1400, sf-1600, ..., sf-3000, sf-full)
python compete.py --init stockfish latest

# Rusty Rival
python compete.py --init rusty v1.0.17

# Java Rival
python compete.py --init java 38
```

### Enabling and Disabling Engines

```bash
# List all engines with their status
python compete.py --list

# Disable engines (won't be selected in random/gauntlet mode)
python compete.py --disable sf-1400 sf-full

# Enable engines
python compete.py --enable java-rival-38 v1.0.17
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
python compete.py v1.0.17 sf-2400 --games 100 --time 1.0
```

- Each opening played twice (once per side)
- Uses built-in opening book (250 positions)
- Elo ratings **are updated**

### Round-Robin League (3+ engines)

```bash
python compete.py v1.0.17 sf-2400 sf-2600 sf-2800 --games 50 --time 1.0
```

- All possible pairings played
- Shows league table after each game
- Elo ratings **are updated**

### Gauntlet Mode

```bash
python compete.py v1.0.17 --gauntlet --games 50 --time 0.5
```

- Tests one engine against all **active** engines
- Elo ratings **are updated**

### Random Mode

```bash
python compete.py --random --games 100 --time 0.5

# Weighted: favor engines with fewer games
python compete.py --random --weighted --games 100 --time 0.5
```

- Continuous random pairings from **active** engines
- Re-checks active engines before each match (live enable/disable)
- Elo ratings **are updated**

### EPD Mode

```bash
python compete.py v1.0.17 sf-2800 --epd eet.epd --time 1.0
```

- Play through positions from an EPD file
- Elo ratings **are NOT updated**

---

## Command Reference

### Options

| Option | Description |
|--------|-------------|
| `--games N`, `-g N` | Number of games/rounds (default: 100) |
| `--time T`, `-t T` | Time per move in seconds (default: 1.0) |
| `--timelow T` | Minimum time per move (use with `--timehigh`) |
| `--timehigh T` | Maximum time per move (use with `--timelow`) |
| `--no-book` | Disable opening book |
| `--gauntlet` | Gauntlet mode |
| `--random` | Random mode |
| `--weighted` | With `--random`: favor engines with fewer games |
| `--epd FILE` | EPD mode |
| `--list` | List all engines with their active status |
| `--enable ENGINE...` | Enable engines |
| `--disable ENGINE...` | Disable engines |
| `--init TYPE VERSION` | Download and enable engine |

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
| EPD | **No** |

---

## Web Dashboard

A Flask-based web interface displays statistics and Elo ratings.

```bash
# Linux/macOS
source .venv/bin/activate
flask --app web.app run

# Windows
.venv\Scripts\activate.bat
flask --app web.app run
```

Access at `http://localhost:5000`

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

---

## Environment Variables

| Variable | Description |
|----------|-------------|
| `DATABASE_URL` | PostgreSQL connection string (required) |
| `ENGINES_DIR` | Override engines directory location (optional) |
| `COMPUTER_NAME` | Override hostname in PGN output (optional) |
