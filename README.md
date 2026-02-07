# Chess Engine Competition Framework

A comprehensive engine vs engine testing harness with Elo tracking, automatic engine management, multiple competition modes, and distributed SPSA parameter tuning.

**Key Features:**
- **Competition Modes**: Head-to-head matches, round-robin leagues, gauntlet testing, knockout cups, random pairings
- **Elo Tracking**: Automatic rating updates with BayesElo and Ordo calculations
- **SPSA Tuning**: Distributed parameter optimization across multiple machines
- **Web Dashboard**: Live statistics, H2H grids, cup brackets, EPD test results
- **Engine Management**: Auto-download engines, enable/disable, filter by type

---

## Table of Contents

1. [Quick Start](#quick-start)
2. [Setup](#setup)
3. [Engine Management](#engine-management)
4. [Competition Modes](#competition-modes)
5. [Parallel Execution](#parallel-execution)
6. [SPSA Tuning](#spsa-tuning)
7. [Command Reference](#command-reference)
8. [Elo Rating System](#elo-rating-system)
9. [Web Dashboard](#web-dashboard)
10. [Development](#development)
11. [Troubleshooting](#troubleshooting)

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

# SPSA tuning (run master on one machine, workers on many)
python -m compete.spsa.master          # Master: orchestrates iterations
python -m compete --spsa -c 8          # Worker: plays games (run on multiple machines)
```

---

## Setup

### Prerequisites

- Python 3.10+
- PostgreSQL database
- Rust toolchain (only required for SPSA tuning - to compile engine variants)

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

Create a PostgreSQL database and run the full schema:

```sql
-- Engines
CREATE TABLE engines (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) UNIQUE NOT NULL,
    binary_path VARCHAR(500),
    active BOOLEAN DEFAULT TRUE,
    initial_elo INTEGER DEFAULT 1500,
    uci_options JSONB,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_engines_active ON engines(active);

-- Games
CREATE TABLE games (
    id SERIAL PRIMARY KEY,
    white_engine_id INTEGER NOT NULL REFERENCES engines(id),
    black_engine_id INTEGER NOT NULL REFERENCES engines(id),
    result VARCHAR(10) NOT NULL,
    white_score NUMERIC(2,1) NOT NULL,
    black_score NUMERIC(2,1) NOT NULL,
    date_played DATE NOT NULL,
    time_control VARCHAR(50),
    time_per_move_ms INTEGER,
    hostname VARCHAR(100),
    opening_name VARCHAR(100),
    opening_fen TEXT,
    pgn TEXT,
    is_rated BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_games_white_engine ON games(white_engine_id);
CREATE INDEX idx_games_black_engine ON games(black_engine_id);
CREATE INDEX idx_games_date ON games(date_played);

-- Elo filter cache (for filtered rating views)
CREATE TABLE elo_filter_cache (
    id SERIAL PRIMARY KEY,
    min_time_ms INTEGER NOT NULL DEFAULT 0,
    max_time_ms INTEGER NOT NULL DEFAULT 999999999,
    hostname VARCHAR(100),
    engine_type VARCHAR(20),
    last_game_id INTEGER NOT NULL DEFAULT 0,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE (min_time_ms, max_time_ms, hostname, engine_type)
);

CREATE TABLE elo_filter_ratings (
    id SERIAL PRIMARY KEY,
    filter_id INTEGER NOT NULL REFERENCES elo_filter_cache(id) ON DELETE CASCADE,
    engine_id INTEGER NOT NULL REFERENCES engines(id) ON DELETE CASCADE,
    elo NUMERIC(7,2) NOT NULL,
    bayes_elo NUMERIC(7,2),
    ordo NUMERIC(7,2),
    games_played INTEGER NOT NULL DEFAULT 0,
    UNIQUE (filter_id, engine_id)
);

-- Cup competitions
CREATE TABLE cups (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    status VARCHAR(20) NOT NULL DEFAULT 'in_progress',
    num_participants INTEGER NOT NULL,
    games_per_match INTEGER NOT NULL DEFAULT 10,
    time_per_move_ms INTEGER,
    time_low_ms INTEGER,
    time_high_ms INTEGER,
    winner_engine_id INTEGER REFERENCES engines(id),
    hostname VARCHAR(100),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP
);

CREATE TABLE cup_rounds (
    id SERIAL PRIMARY KEY,
    cup_id INTEGER NOT NULL REFERENCES cups(id) ON DELETE CASCADE,
    round_number INTEGER NOT NULL,
    round_name VARCHAR(50),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    UNIQUE (cup_id, round_number)
);

CREATE TABLE cup_matches (
    id SERIAL PRIMARY KEY,
    round_id INTEGER NOT NULL REFERENCES cup_rounds(id) ON DELETE CASCADE,
    match_order INTEGER NOT NULL,
    engine1_id INTEGER NOT NULL REFERENCES engines(id),
    engine2_id INTEGER REFERENCES engines(id),
    engine1_seed INTEGER,
    engine2_seed INTEGER,
    engine1_points NUMERIC(4,1) DEFAULT 0,
    engine2_points NUMERIC(4,1) DEFAULT 0,
    games_played INTEGER DEFAULT 0,
    winner_engine_id INTEGER REFERENCES engines(id),
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    is_tiebreaker BOOLEAN DEFAULT FALSE,
    decided_by_coin_flip BOOLEAN DEFAULT FALSE,
    UNIQUE (round_id, match_order)
);

-- EPD test results
CREATE TABLE epd_test_runs (
    id SERIAL PRIMARY KEY,
    epd_file VARCHAR(255) NOT NULL,
    total_positions INTEGER NOT NULL,
    timeout_seconds FLOAT NOT NULL,
    score_tolerance INTEGER NOT NULL,
    hostname VARCHAR(255),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE epd_test_results (
    id SERIAL PRIMARY KEY,
    run_id INTEGER NOT NULL REFERENCES epd_test_runs(id) ON DELETE CASCADE,
    engine_id INTEGER NOT NULL REFERENCES engines(id) ON DELETE CASCADE,
    position_id VARCHAR(255) NOT NULL,
    position_index INTEGER NOT NULL,
    fen TEXT NOT NULL,
    test_type VARCHAR(10) NOT NULL,
    expected_moves VARCHAR(255) NOT NULL,
    solved BOOLEAN NOT NULL,
    move_found VARCHAR(20),
    solve_time_ms INTEGER,
    final_depth INTEGER,
    score_cp INTEGER,
    score_mate INTEGER,
    score_valid BOOLEAN,
    timed_out BOOLEAN NOT NULL DEFAULT FALSE,
    points_earned INTEGER
);

CREATE INDEX idx_epd_results_run ON epd_test_results(run_id);
CREATE INDEX idx_epd_results_engine ON epd_test_results(engine_id);
CREATE INDEX idx_epd_results_position ON epd_test_results(position_id);

-- SPSA tuning
CREATE TABLE spsa_runs (
    id SERIAL PRIMARY KEY,
    name VARCHAR(100) NOT NULL,
    description TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    is_active BOOLEAN DEFAULT TRUE
);

CREATE TABLE spsa_iterations (
    id SERIAL PRIMARY KEY,
    run_id INTEGER REFERENCES spsa_runs(id),
    iteration_number INTEGER NOT NULL,
    effective_iteration INTEGER,
    plus_engine_path VARCHAR(500) NOT NULL,
    minus_engine_path VARCHAR(500) NOT NULL,
    base_engine_path VARCHAR(500),
    ref_engine_path VARCHAR(500),
    timelow_ms INTEGER NOT NULL,
    timehigh_ms INTEGER NOT NULL,
    target_games INTEGER NOT NULL DEFAULT 150,
    games_played INTEGER NOT NULL DEFAULT 0,
    plus_wins INTEGER NOT NULL DEFAULT 0,
    minus_wins INTEGER NOT NULL DEFAULT 0,
    draws INTEGER NOT NULL DEFAULT 0,
    ref_target_games INTEGER NOT NULL DEFAULT 100,
    ref_games_played INTEGER NOT NULL DEFAULT 0,
    ref_wins INTEGER NOT NULL DEFAULT 0,
    ref_losses INTEGER NOT NULL DEFAULT 0,
    ref_draws INTEGER NOT NULL DEFAULT 0,
    status VARCHAR(20) NOT NULL DEFAULT 'pending',
    base_parameters JSONB,
    plus_parameters JSONB,
    minus_parameters JSONB,
    perturbation_signs JSONB,
    gradient_estimate JSONB,
    elo_diff NUMERIC(7,2),
    ref_elo_estimate NUMERIC(7,2),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    completed_at TIMESTAMP,
    UNIQUE (run_id, iteration_number)
);

CREATE INDEX idx_spsa_iteration_number ON spsa_iterations(iteration_number);
CREATE INDEX idx_spsa_status ON spsa_iterations(status);
CREATE INDEX idx_spsa_run_id ON spsa_iterations(run_id);

CREATE TABLE spsa_workers (
    id SERIAL PRIMARY KEY,
    worker_name VARCHAR(100) UNIQUE NOT NULL,
    last_iteration_id INTEGER REFERENCES spsa_iterations(id),
    last_phase VARCHAR(20),
    total_games INTEGER NOT NULL DEFAULT 0,
    total_spsa_games INTEGER NOT NULL DEFAULT 0,
    total_ref_games INTEGER NOT NULL DEFAULT 0,
    avg_nps INTEGER,
    last_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP,
    first_seen_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE spsa_worker_heartbeats (
    id SERIAL PRIMARY KEY,
    worker_id INTEGER NOT NULL REFERENCES spsa_workers(id) ON DELETE CASCADE,
    iteration_id INTEGER REFERENCES spsa_iterations(id),
    phase VARCHAR(20) NOT NULL,
    games_reported INTEGER NOT NULL,
    avg_nps INTEGER,
    created_at TIMESTAMP NOT NULL DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_spsa_worker_heartbeats_worker ON spsa_worker_heartbeats(worker_id);
CREATE INDEX idx_spsa_worker_heartbeats_created ON spsa_worker_heartbeats(created_at);
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
  compete/
    spsa/
      config.toml     # Tuning hyperparameters
      params.toml     # Parameter values (updated by SPSA)
      master.py       # Master orchestration
      worker.py       # Worker game playing
      build.py        # Engine building
  engines/
    stockfish/
      stockfish-windows-x86-64-avx2.exe   # Auto-downloaded
    v1.0.17/
      rusty-rival-v1.0.17-windows-x86_64.exe
    java-rival-38.0.0/
      rivalchess-v38.0.0.jar
    spsa/             # SPSA-built engines (ephemeral)
      spsa-plus/
        rusty-rival.exe
      spsa-minus/
        rusty-rival.exe
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

### EPD Solve Mode (Test Suite)

Test an engine's ability to find correct moves with time-to-solution tracking and score validation:

```bash
# Test all active engines (default)
python -m compete --epd-solve eet.epd --timeout 30

# Test specific engines
python -m compete v1.0.17 sf-2400 --epd-solve eet.epd --timeout 30

# Filter by engine type
python -m compete --epd-solve eet.epd --timeout 30 --enginetype rusty
python -m compete --epd-solve eet.epd --timeout 30 --enginetype stockfish

# Include inactive engines
python -m compete --epd-solve eet.epd --timeout 30 --includeinactive

# Single position with verbose output (for debugging)
python -m compete v1.0.17 --epd-solve eet.epd --position 10 --timeout 30
python -m compete v1.0.17 --epd-solve eet.epd -p "E_E_T 010" --timeout 30

# Don't save results to database
python -m compete v1.0.17 --epd-solve eet.epd --timeout 30 --no-store

# With stricter score validation (±30 centipawns tolerance)
python -m compete v1.0.17 --epd-solve eet.epd --timeout 30 --score-tolerance 30
```

Features:
- **Time-to-solution**: Search until the engine finds the expected `bm` (best move)
- **Avoid moves**: Test that engine does NOT play `am` (avoid move) positions
- **Score validation**: Verify engine's evaluation matches expected `ce` (centipawn) value
- **Database storage**: Results saved to database for viewing on web dashboard
- **Single position mode**: Verbose output showing all engine info lines for debugging
- Supports EPD operations: `bm` (best moves), `am` (avoid moves), `ce` (centipawn eval), `dm` (direct mate)
- Reports solve times, success rates, and breakdown statistics
- Elo ratings **are NOT updated**

EPD File Format:
```
8/8/p2p3p/3k2p1/PP6/3K1P1P/8/8 b - - bm Kc6; ce +150; id "Endgame 1";
r1bqkb1r/pppp1ppp/2n2n2/4p2Q/2B1P3/8/PPPP1PPP/RNB1K1NR w KQkq - bm Qxf7+; dm 2; id "Scholar's Mate";
4k3/8/3PP1p1/8/3K3p/8/3n2PP/8 b - - am Nf1; id "Avoid move test";
```

---

## Parallel Execution

All competition modes support parallel game execution using the `--concurrency` / `-c` flag:

```bash
# Run 8 games in parallel
python -m compete --random --games 1000 --timelow 0.25 --timehigh 1.0 -c 8

# Gauntlet with 6 parallel games
python -m compete v1.0.17 --gauntlet --games 50 -c 6

# Cup tournament with parallel games within each match
python -m compete --cup --games 10 -c 4

# League with parallel execution
python -m compete v1.0.17 sf-2400 sf-2600 --games 50 -c 8
```

### Why Parallelization Works

Chess engines are single-threaded, so on an 8-core machine:
- 1 game uses 1 core, 7 cores idle
- 8 parallel games use 8 cores, each at ~85-95% NPS
- Total throughput: ~7x improvement

### NPS Impact

| Parallel Games | NPS per Game | Notes |
|----------------|--------------|-------|
| 1 | 100% | Full turbo boost |
| 4 | 95-98% | Minor cache pressure |
| 8 | 85-95% | Turbo reduction, shared L3 |

The slight per-game slowdown is vastly outweighed by throughput gains.

---

## SPSA Tuning

SPSA (Simultaneous Perturbation Stochastic Approximation) mode enables distributed parameter tuning for chess engines. It perturbs all parameters simultaneously and estimates gradients from game results.

### How SPSA Works

1. **Perturbation**: For each iteration, randomly perturb ALL parameters by ±delta
2. **Build**: Compile two engine versions: one with +delta params ("plus"), one with -delta ("minus")
3. **Play**: Workers play games between plus and minus engines
4. **Gradient**: Estimate gradient from win rate difference: `g = (plus_wins - minus_wins) / delta`
5. **Update**: Move parameters in the gradient direction: `param += step_size * g`
6. **Repeat**: Continue for N iterations, with step size decreasing over time

### Architecture

All SPSA tooling lives in `compete/spsa/`:

- **Master** (`master.py`): Orchestrates iterations - creates perturbed params, builds engines, waits for games, calculates gradients, updates params
- **Workers** (`worker.py`): Poll database for work, play games between plus/minus engines, report results
- **Build** (`build.py`): Modifies `engine_constants.rs` with parameter values, compiles engine, restores original
- **Database**: Coordinates via `spsa_iterations` table

### Configuration Files

**`compete/spsa/params.toml`** - Parameters being tuned:
```toml
[see_prune_max_depth]
value = 10.0          # Current value (updated by SPSA)
min = 3               # Lower bound
max = 14              # Upper bound
step = 1              # Perturbation delta
```

**`compete/spsa/config.toml`** - Tuning settings:
```toml
[time_control]
timelow = 0.25        # Min seconds per move
timehigh = 1.0        # Max seconds per move

[games]
games_per_iteration = 150

[spsa]
max_iterations = 500
a = 1.0               # Step size numerator
c = 1.0               # Perturbation multiplier
A = 50                # Stability constant

[build]
rusty_rival_path = "../rusty-rival"    # Path to engine source
engines_output_path = "engines/spsa"   # Where to put built engines

[reference]
engine_path = "../rusty-rival/engines/stockfish"  # Path to Stockfish (file or directory)
engine_elo = 2600     # Assumed Elo of reference engine
enabled = true        # Set to false to disable reference games
```

### Running the Master

The master runs on a single machine and orchestrates the tuning:

```bash
cd chess-compete
python -m compete.spsa.master
```

The master will:
1. Read current parameter values from `params.toml`
2. Generate random perturbations (±delta for each parameter)
3. Build plus and minus engines by modifying `engine_constants.rs` and compiling
4. Create an iteration record in the database with engine paths
5. Wait for workers to complete the target number of games
6. Calculate gradient estimate from results
7. Update parameter values in `params.toml`
8. Repeat until max iterations reached

### Running Workers

Workers can run on multiple machines (including the master machine):

```bash
# Start SPSA worker with 8 parallel games
python -m compete --spsa -c 8
```

Workers will:
1. Poll the database for pending iterations
2. Build engines locally from the parameter values stored in the iteration record
3. Run games with random openings and time controls
4. Update aggregate results atomically (no individual game saves)
5. Continue until iteration is complete, then fetch next

### Remote Workers (Docker)

For running workers on remote machines without database access, use the HTTP worker mode. Workers communicate via API instead of direct database connection.

#### Option 1: Docker (Recommended for Remote Machines)

```bash
# On the remote machine
git clone https://github.com/chris-moreton/chess-compete.git
cd chess-compete/docker

# Configure
cp .env.example .env
# Edit .env:
#   SPSA_API_KEY=your-api-key
#   CONCURRENCY=4

# Build and run
docker-compose up -d

# Check logs
docker-compose logs -f

# Scale to multiple containers
docker-compose up -d --scale worker=3

# Stop
docker-compose down
```

#### Option 2: Direct HTTP Worker (No Docker)

If you have Python and Rust installed:

```bash
# Clone both repos
git clone https://github.com/chris-moreton/chess-compete.git
git clone https://github.com/chris-moreton/rusty-rival.git

cd chess-compete
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows
pip install -r requirements.txt

# Configure .env
echo "SPSA_API_URL=https://chess-compete-production.up.railway.app" >> .env
echo "SPSA_API_KEY=your-api-key" >> .env

# Run the HTTP worker
python -m compete --spsa-http -c 4
```

#### HTTP Worker vs Database Worker

| Feature | `--spsa` (DB Worker) | `--spsa-http` (HTTP Worker) |
|---------|---------------------|----------------------------|
| Connection | Direct PostgreSQL | HTTPS API |
| Credentials | DATABASE_URL | API key only |
| Use case | Local/trusted machines | Remote/Docker workers |
| Setup | Needs DB access | Just needs API URL + key |

### Engine Build Process

When the master or workers build engines:

1. Read `rusty-rival/src/engine_constants.rs`
2. Replace constant values using regex patterns (e.g., `pub const SEE_PRUNE_MAX_DEPTH: u8 = 6;`)
3. Run `cargo build --release` in the rusty-rival directory
4. Copy the binary to the output path
5. **Restore the original `engine_constants.rs`** (so the source tree stays clean)

The mapping from `params.toml` names to Rust constants is defined in `build.py`.

### Key Features

- **No game saves**: Only aggregate results (wins/losses/draws) stored in `spsa_iterations`
- **No engine registration**: SPSA engines are ephemeral, loaded directly from path
- **Random openings**: Each game uses a random opening from the standard book
- **Time variety**: Random time per game within the iteration's `timelow`-`timehigh` range
- **Distributed**: Multiple workers can contribute to the same iteration
- **Crash recovery**: Master resumes incomplete iterations; workers can restart safely
- **Reference games**: Track actual engine strength by playing against Stockfish each iteration

### Adding New Parameters

To add parameters for tuning:

1. Add the parameter to `params.toml` with appropriate bounds and step size
2. Add the regex pattern mapping in `build.py` (`PARAM_MAPPINGS` or `ARRAY_PARAM_MAPPINGS`)
3. Restart the master (it reads params.toml fresh each iteration)

### Interpreting Results

Monitor parameter progression on the web dashboard at `/spsa`. The dashboard shows:

- **Estimated Elo vs Stockfish**: Actual engine strength over iterations (from reference games)
- **Parameter Stability**: Rolling standard deviation - lower = more converged
- **Parameter Progression**: Individual charts for each tuned parameter
- **Summary Table**: First value, current value, and percentage change

Look for:
- **Clear trends**: Parameter moving consistently in one direction (strong signal)
- **Hitting bounds**: If a parameter hits min/max, consider expanding the bounds
- **Oscillation**: Noisy movement may indicate weak signal or high variance - consider more games per iteration
- **Stability**: Parameters near starting values were likely already well-tuned
- **Elo improvement**: Rising Elo vs Stockfish indicates the tuning is working

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
| `--epd-solve FILE` | EPD solve mode: test engine's ability to find correct moves |
| `--spsa` | SPSA worker mode: poll for iterations and run games |

### Engine Filter Options

| Option | Description |
|--------|-------------|
| `--enginetype TYPE` | Filter engines by type: `rusty` or `stockfish` |
| `--includeinactive` | Include inactive engines (for random/gauntlet/cup/epd-solve modes) |

### Cup Mode Options

| Option | Description |
|--------|-------------|
| `--cup-engines N` | Limit cup to top N engines by Ordo rating |
| `--cup-name NAME` | Custom name for the cup competition |

### EPD Solve Mode Options

| Option | Description |
|--------|-------------|
| `--timeout T` | Timeout per position in seconds (default: 30.0) |
| `--score-tolerance N` | Tolerance for score validation in centipawns (default: 50) |
| `--position N`, `-p N` | Solve only position N (number or ID) with verbose output |
| `--no-store` | Don't save results to database |

### Time Control Options

| Option | Description |
|--------|-------------|
| `--time T`, `-t T` | Fixed time per move in seconds (default: 1.0) |
| `--timelow T` | Minimum time per move (use with `--timehigh`) |
| `--timehigh T` | Maximum time per move (use with `--timelow`) |

When using `--timelow` and `--timehigh`, a random time is selected for each match/round.

### Parallel Execution Options

| Option | Description |
|--------|-------------|
| `--concurrency N`, `-c N` | Number of games to run in parallel (default: 1) |

### SPSA Mode Options

| Option | Description |
|--------|-------------|
| `--spsa` | SPSA worker mode with direct database access |
| `--spsa-http` | SPSA worker mode via HTTP API (for remote/Docker workers) |
| `--api-url URL` | API base URL for HTTP worker (or set `SPSA_API_URL`) |
| `--api-key KEY` | API key for HTTP worker (or set `SPSA_API_KEY`) |

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
| SPSA | **No** |

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
| `/epd-tests` | EPD test results overview with engine percentages |
| `/epd-tests/<file>` | Detailed position-by-position results grid |
| `/spsa` | SPSA parameter tuning progress and charts |

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
├── openings.py        # OPENING_BOOK data, load_epd_positions()
└── spsa/
    ├── __init__.py    # SPSA module exports
    ├── master.py      # SPSA master orchestration
    ├── worker.py      # SPSA worker (game playing)
    ├── build.py       # Engine building utilities
    ├── config.toml    # Tuning hyperparameters
    └── params.toml    # Parameter values and bounds

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
| `DATABASE_URL` | PostgreSQL connection string (required for DB worker) |
| `ENGINES_DIR` | Override engines directory location (optional) |
| `COMPUTER_NAME` | Override hostname in PGN output (optional) |
| `SPSA_API_URL` | API endpoint for HTTP worker (required for `--spsa-http`) |
| `SPSA_API_KEY` | API authentication key (required for `--spsa-http`) |
| `SPSA_WORKER_API_KEY` | Server-side API key for authenticating workers (set on web server) |
