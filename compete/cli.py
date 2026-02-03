"""
Command-line interface for the chess engine competition harness.
"""

import argparse
import multiprocessing
import os
import sys
from pathlib import Path

# Set stack size for Rust engines (rusty-rival requires this)
os.environ['RUST_MIN_STACK'] = '4097152'

# Fix for macOS: use 'fork' start method for multiprocessing
# macOS switched to 'spawn' in Python 3.8 which causes issues with ProcessPoolExecutor
if sys.platform == 'darwin':
    try:
        multiprocessing.set_start_method('fork')
    except RuntimeError:
        pass  # Already set

from compete.engine_manager import (
    get_engines_dir,
    init_engine,
    init_stockfish,
    list_engines_status,
    set_engine_active,
    resolve_engine_name,
    get_active_engines,
    ensure_engines_initialized,
)
from compete.openings import OPENING_BOOK, load_epd_positions
from compete.competitions import (
    run_match,
    run_league,
    run_gauntlet,
    run_random,
    run_epd,
    run_epd_solve,
    run_epd_solve_single,
)
from compete.cup import run_cup


def main():
    """Main entry point for the CLI."""
    # Required for Windows multiprocessing support
    multiprocessing.freeze_support()

    parser = argparse.ArgumentParser(
        description="Chess engine competition harness",
        epilog="Engine names can be shorthand (v1, v10) or full (v1-baseline, v10-arrayvec-movelist)"
    )
    parser.add_argument("engines", nargs="*",
                        help="Engine version names (e.g., v1, v10, or v1-baseline)")
    parser.add_argument("--games", "-g", type=int, default=100,
                        help="Number of games per pairing (default: 100)")
    parser.add_argument("--concurrency", "-c", type=int, default=1,
                        help="Number of games to run in parallel (default: 1)")
    parser.add_argument("--time", "-t", type=float, default=None,
                        help="Time per move in seconds (default: 1.0)")
    parser.add_argument("--timelow", type=float, default=None,
                        help="Minimum time per move (use with --timehigh for random range)")
    parser.add_argument("--timehigh", type=float, default=None,
                        help="Maximum time per move (use with --timelow for random range)")
    parser.add_argument("--no-book", action="store_true",
                        help="Disable opening book (start all games from initial position)")
    parser.add_argument("--gauntlet", action="store_true",
                        help="Gauntlet mode: test one engine against all others in engines directory")
    parser.add_argument("--random", action="store_true",
                        help="Random mode: randomly pair engines for 2-game matches")
    parser.add_argument("--weighted", action="store_true",
                        help="With --random: weight selection by inverse game count (fewer games = higher chance)")
    parser.add_argument("--epd", type=str, default=None,
                        help="EPD file mode: play through positions from an EPD file sequentially")
    parser.add_argument("--epd-solve", type=str, default=None, metavar="FILE",
                        help="EPD solve mode: test engine's ability to find correct moves")
    parser.add_argument("--timeout", type=float, default=30.0,
                        help="Timeout per position in seconds for --epd-solve (default: 30.0)")
    parser.add_argument("--score-tolerance", type=int, default=50,
                        help="Tolerance for score validation in centipawns (default: 50)")
    parser.add_argument("--no-store", action="store_true",
                        help="Don't save EPD solve results to database")
    parser.add_argument("--position", "-p", type=str, default=None,
                        help="With --epd-solve: solve only this position (number or ID) with verbose output")
    parser.add_argument("--enable", type=str, nargs="+", metavar="ENGINE",
                        help="Enable one or more engines (set active=True)")
    parser.add_argument("--disable", type=str, nargs="+", metavar="ENGINE",
                        help="Disable one or more engines (set active=False)")
    parser.add_argument("--list", action="store_true",
                        help="List all engines with their active status")
    parser.add_argument("--init", type=str, nargs="+", metavar="ARG",
                        help="Download, initialize, and enable an engine. "
                             "Usage: --init TYPE VERSION [ELO]. "
                             "TYPE is 'rusty', 'java', or 'stockfish'. "
                             "ELO is optional starting strength (default: derived from engine name). "
                             "Examples: --init rusty v1.0.17, --init java 38 2400, --init stockfish latest")
    parser.add_argument("--cup", action="store_true",
                        help="Cup mode: knockout tournament with seeded brackets")
    parser.add_argument("--cup-engines", type=int, default=None, metavar="N",
                        help="Limit cup to top N engines by Ordo rating (default: all active)")
    parser.add_argument("--cup-name", type=str, default=None, metavar="NAME",
                        help="Custom name for the cup competition")
    parser.add_argument("--enginetype", type=str, default=None, choices=["rusty", "stockfish"],
                        help="Filter engines by type (rusty or stockfish)")
    parser.add_argument("--includeinactive", action="store_true",
                        help="Include inactive engines in cup competition")
    parser.add_argument("--spsa", action="store_true",
                        help="SPSA worker mode: poll for iterations and run games")
    parser.add_argument("--spsa-batch", type=int, default=1, metavar="N",
                        help="SPSA: games per batch before database update (default: 1)")

    args = parser.parse_args()

    # Handle --list command
    if args.list:
        engines = list_engines_status()
        if not engines:
            print("No engines found in database")
            sys.exit(0)
        print(f"\n{'Engine':<30} {'Status':<10} {'Elo':>8} {'Games':>8}")
        print("-" * 58)
        for name, active, elo, games in engines:
            status = "active" if active else "disabled"
            print(f"{name:<30} {status:<10} {elo:>8.0f} {games:>8}")
        print()
        sys.exit(0)

    # Handle --init command (downloads and enables engine)
    if args.init:
        if len(args.init) < 2 or len(args.init) > 3:
            print("Error: --init requires 2-3 arguments: TYPE VERSION [ELO]")
            sys.exit(1)

        engine_type = args.init[0]
        version = args.init[1]
        starting_elo = None
        if len(args.init) == 3:
            try:
                starting_elo = float(args.init[2])
            except ValueError:
                print(f"Error: Invalid Elo value '{args.init[2]}'. Must be a number.")
                sys.exit(1)

        # Handle stockfish specially
        if engine_type.lower() == "stockfish":
            if init_stockfish():
                # Enable all stockfish variants
                for elo in [1400, 1600, 1800, 2000, 2200, 2400, 2500, 2600, 2700, 2800, 3000]:
                    set_engine_active(f"sf-{elo}", True)
                set_engine_active("sf-full", True)
                print("Enabled all Stockfish engines")
                sys.exit(0)
            else:
                sys.exit(1)

        # For rusty/java engines, init and enable
        if init_engine(engine_type, version):
            # Determine the engine name to enable
            if engine_type == "rusty":
                engine_name = version if version.startswith("v") else f"v{version}"
            elif engine_type == "java":
                v = version[1:] if version.startswith("v") else version
                v = v.split(".")[0]  # "38.0.0" -> "38"
                engine_name = f"java-rival-{v}"
            else:
                engine_name = version

            if set_engine_active(engine_name, True, starting_elo):
                elo_msg = f" with starting Elo {starting_elo:.0f}" if starting_elo else ""
                print(f"Enabled: {engine_name}{elo_msg}")
            sys.exit(0)
        else:
            sys.exit(1)

    # Handle --enable command
    if args.enable:
        for engine in args.enable:
            if set_engine_active(engine, True):
                print(f"Enabled: {engine}")
            else:
                print(f"Error: Engine '{engine}' not found in database")
        sys.exit(0)

    # Handle --disable command
    if args.disable:
        for engine in args.disable:
            if set_engine_active(engine, False):
                print(f"Disabled: {engine}")
            else:
                print(f"Error: Engine '{engine}' not found in database")
        sys.exit(0)

    # Validate time arguments
    if args.timelow is not None or args.timehigh is not None:
        if args.timelow is None or args.timehigh is None:
            print("Error: --timelow and --timehigh must be used together")
            sys.exit(1)
        if args.timelow > args.timehigh:
            print("Error: --timelow must be less than or equal to --timehigh")
            sys.exit(1)
        if args.time is not None:
            print("Error: Cannot use --time with --timelow/--timehigh")
            sys.exit(1)
        time_per_move = None  # Will use range
        time_low = args.timelow
        time_high = args.timehigh
    else:
        time_per_move = args.time if args.time is not None else 1.0
        time_low = None
        time_high = None

    script_dir = Path(__file__).parent.parent  # Go up from compete/ to project root
    engine_dir = get_engines_dir()
    results_dir = script_dir / "results" / "competitions"
    results_dir.mkdir(parents=True, exist_ok=True)

    use_opening_book = not args.no_book

    # Resolve shorthand engine names to full names
    resolved_engines = [resolve_engine_name(e, engine_dir) for e in args.engines]

    # Print resolved names if different from input
    for orig, resolved in zip(args.engines, resolved_engines):
        if orig != resolved:
            print(f"Resolved '{orig}' -> '{resolved}'")

    # Auto-initialize missing engines
    if args.random or args.gauntlet or args.cup or (args.epd_solve and not resolved_engines):
        # For random/gauntlet/cup/epd-solve (no engines specified): ensure all matching engines from database are initialized
        engines_to_init = get_active_engines(engine_dir, args.enginetype, args.includeinactive)
        if engines_to_init:
            filter_desc = []
            if args.enginetype:
                filter_desc.append(f"type={args.enginetype}")
            if args.includeinactive:
                filter_desc.append("including inactive")
            filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
            print(f"Checking {len(engines_to_init)} engines{filter_str}...")
            if not ensure_engines_initialized(engines_to_init, engine_dir):
                print("Error: Failed to initialize some engines")
                sys.exit(1)
    elif resolved_engines:
        # For other modes: ensure specified engines are initialized
        if not ensure_engines_initialized(resolved_engines, engine_dir):
            print("Error: Failed to initialize some engines")
            sys.exit(1)

    if args.spsa:
        # SPSA worker mode: poll for iterations and run games
        from compete.spsa import run_spsa_worker
        run_spsa_worker(
            concurrency=args.concurrency,
            batch_size=args.spsa_batch,
            poll_interval=10
        )
    elif args.cup:
        # Cup mode: knockout tournament with seeded brackets
        if args.engines:
            print("Warning: Engine arguments ignored in cup mode (uses active engines by Ordo rating)")
        run_cup(engine_dir, args.cup_engines, args.games, time_per_move or 1.0,
                args.cup_name, time_low, time_high,
                engine_type=args.enginetype, include_inactive=args.includeinactive,
                concurrency=args.concurrency)
    elif args.epd_solve:
        # EPD solve mode: test engine's ability to find correct moves
        epd_path = Path(args.epd_solve)
        if not epd_path.exists():
            # Try looking in openings directory
            epd_path = script_dir / "openings" / args.epd_solve
            if not epd_path.exists():
                print(f"Error: EPD file not found: {args.epd_solve}")
                print(f"Searched: {Path(args.epd_solve).absolute()} and {epd_path}")
                sys.exit(1)

        # If no engines specified, use all active engines (with optional filters)
        epd_engines = resolved_engines
        if not epd_engines:
            epd_engines = get_active_engines(engine_dir, args.enginetype, args.includeinactive)
            if not epd_engines:
                filter_desc = []
                if args.enginetype:
                    filter_desc.append(f"type={args.enginetype}")
                if not args.includeinactive:
                    filter_desc.append("active only")
                filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
                print(f"Error: No engines found{filter_str}")
                sys.exit(1)
            filter_desc = []
            if args.enginetype:
                filter_desc.append(f"type={args.enginetype}")
            if args.includeinactive:
                filter_desc.append("including inactive")
            filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
            print(f"Using {len(epd_engines)} engines{filter_str}")

        if args.position:
            # Single position verbose mode
            run_epd_solve_single(epd_engines, engine_dir, epd_path,
                                 args.position, args.timeout, args.score_tolerance)
        else:
            # Full test suite mode
            run_epd_solve(epd_engines, engine_dir, epd_path,
                          args.timeout, results_dir, args.score_tolerance,
                          store_results=not args.no_store)
    elif args.epd:
        # EPD mode: play through positions from an EPD file
        epd_path = Path(args.epd)
        if not epd_path.exists():
            # Try looking in openings directory
            epd_path = script_dir / "openings" / args.epd
            if not epd_path.exists():
                print(f"Error: EPD file not found: {args.epd}")
                print(f"Searched: {Path(args.epd).absolute()} and {epd_path}")
                sys.exit(1)
        if len(resolved_engines) < 2:
            print("Error: EPD mode requires at least 2 engines")
            sys.exit(1)
        if time_per_move is None:
            print("Error: --timelow/--timehigh not supported in EPD mode, use --time")
            sys.exit(1)
        run_epd(resolved_engines, engine_dir, epd_path, time_per_move, results_dir)
    elif args.random:
        # Random mode: randomly pair engines for matches
        if args.engines:
            print("Warning: Engine arguments ignored in random mode")
        run_random(engine_dir, args.games, time_per_move, results_dir, args.weighted, time_low, time_high,
                   engine_type=args.enginetype, include_inactive=args.includeinactive,
                   concurrency=args.concurrency)
    elif args.gauntlet:
        # Gauntlet mode: test one engine against all others
        if len(resolved_engines) != 1:
            print("Error: Gauntlet mode requires exactly one engine (the challenger)")
            sys.exit(1)
        run_gauntlet(resolved_engines[0], engine_dir, args.games, time_per_move or 1.0, results_dir,
                     time_low, time_high,
                     engine_type=args.enginetype, include_inactive=args.includeinactive,
                     concurrency=args.concurrency)
    elif len(resolved_engines) >= 3:
        # Round-robin league for 3+ engines
        run_league(resolved_engines, engine_dir, args.games, time_per_move or 1.0, results_dir,
                   use_opening_book, time_low, time_high, concurrency=args.concurrency)
    elif len(resolved_engines) == 2:
        # Head-to-head match for exactly 2 engines
        run_match(resolved_engines[0], resolved_engines[1], engine_dir,
                  args.games, time_per_move or 1.0, use_opening_book, time_low, time_high,
                  concurrency=args.concurrency)
    else:
        print("Error: At least 2 engines are required (or use --random or --gauntlet mode)")
        sys.exit(1)


if __name__ == "__main__":
    main()
