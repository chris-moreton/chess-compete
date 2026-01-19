"""
Competition modes: match, league, gauntlet, random, and EPD.
"""

import os
import random
import socket
import subprocess
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from itertools import combinations
from pathlib import Path

import chess
import chess.engine

from compete.constants import PROVISIONAL_GAMES, DEFAULT_ELO
from compete.database import load_elo_ratings, save_game_to_db, get_initial_elo, save_epd_test_run
from compete.engine_manager import get_engine_info, get_active_engines
from compete.game import play_game, calculate_elo_difference, GameConfig, GameResult, play_game_from_config
from compete.openings import OPENING_BOOK, load_epd_positions, load_epd_for_solving
from compete.epd import EpdPosition, SolveResult


def run_match(engine1_name: str, engine2_name: str, engine_dir: Path,
              num_games: int, time_per_move: float,
              use_opening_book: bool = True,
              time_low: float = None, time_high: float = None,
              concurrency: int = 1) -> dict:
    """Run a match between two engines, alternating colors.

    Args:
        concurrency: Number of games to run in parallel (default: 1 = sequential)
    """

    engine1_path, engine1_uci_options = get_engine_info(engine1_name, engine_dir)
    engine2_path, engine2_uci_options = get_engine_info(engine2_name, engine_dir)

    use_time_range = time_low is not None and time_high is not None

    # Load persistent Elo ratings
    elo_ratings = load_elo_ratings()

    # Initialize engines if needed
    for name in [engine1_name, engine2_name]:
        if name not in elo_ratings:
            initial = get_initial_elo(name)
            elo_ratings[name] = {"elo": initial, "games": 0}

    # Store starting Elo for display
    start_elo = {
        engine1_name: elo_ratings[engine1_name]["elo"],
        engine2_name: elo_ratings[engine2_name]["elo"]
    }

    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}
    engine1_points = 0.0
    engine2_points = 0.0

    # Prepare openings - each opening played twice (once per side)
    if use_opening_book:
        # Calculate how many openings we need
        num_opening_pairs = (num_games + 1) // 2

        # Shuffle and cycle through openings
        openings = OPENING_BOOK.copy()
        random.shuffle(openings)

        # Extend if we need more openings than available
        while len(openings) < num_opening_pairs:
            extra = OPENING_BOOK.copy()
            random.shuffle(extra)
            openings.extend(extra)

        openings = openings[:num_opening_pairs]
    else:
        openings = None

    print(f"\n{'='*70}")
    print(f"Match: {engine1_name} vs {engine2_name}")
    if use_time_range:
        print(f"Games: {num_games}, Time: {time_low}-{time_high}s/move (random per game pair)")
    else:
        print(f"Games: {num_games}, Time: {time_per_move}s/move")
    if use_opening_book:
        print(f"Opening book: {len(OPENING_BOOK)} positions (randomized)")
    else:
        print("Opening book: disabled")
    if concurrency > 1:
        print(f"Concurrency: {concurrency} games in parallel")
    print(f"{'='*70}")

    # Show starting Elo
    print(f"\nStarting Elo:")
    for name in [engine1_name, engine2_name]:
        data = elo_ratings[name]
        prov = "?" if data["games"] < PROVISIONAL_GAMES else ""
        print(f"  {name}: {data['elo']:.0f} ({data['games']} games){prov}")
    print(flush=True)

    # Prepare game configurations
    game_configs = []
    pair_time = None

    for i in range(num_games):
        # Get opening for this game pair
        if openings:
            opening_idx = i // 2
            opening_fen, opening_name = openings[opening_idx]
        else:
            opening_fen, opening_name = None, None

        # Select time for this game pair (same time for both colors)
        if i % 2 == 0:
            if use_time_range:
                pair_time = random.uniform(time_low, time_high)
            else:
                pair_time = time_per_move
        game_time = pair_time

        # Alternate colors
        if i % 2 == 0:
            white, black = engine1_name, engine2_name
            white_path, black_path = engine1_path, engine2_path
            white_uci, black_uci = engine1_uci_options, engine2_uci_options
            is_engine1_white = True
        else:
            white, black = engine2_name, engine1_name
            white_path, black_path = engine2_path, engine1_path
            white_uci, black_uci = engine2_uci_options, engine1_uci_options
            is_engine1_white = False

        # Convert Path to string for pickling
        white_path_str = str(white_path) if not isinstance(white_path, list) else white_path
        black_path_str = str(black_path) if not isinstance(black_path, list) else black_path

        config = GameConfig(
            game_index=i,
            white_name=white,
            black_name=black,
            white_path=white_path_str,
            black_path=black_path_str,
            white_uci_options=white_uci,
            black_uci_options=black_uci,
            time_per_move=game_time,
            opening_fen=opening_fen,
            opening_name=opening_name,
            is_engine1_white=is_engine1_white
        )
        game_configs.append(config)

    # Run games (parallel or sequential)
    hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())
    completed_games = 0

    if concurrency > 1:
        # Parallel execution
        with ProcessPoolExecutor(max_workers=concurrency) as executor:
            futures = {executor.submit(play_game_from_config, cfg): cfg for cfg in game_configs}

            for future in as_completed(futures):
                try:
                    game_result = future.result()
                    completed_games += 1

                    # Update results
                    results[game_result.result] += 1

                    # Calculate points
                    if game_result.result == "1-0":
                        if game_result.is_engine1_white:
                            engine1_points += 1
                        else:
                            engine2_points += 1
                    elif game_result.result == "0-1":
                        if game_result.is_engine1_white:
                            engine2_points += 1
                        else:
                            engine1_points += 1
                    elif game_result.result == "1/2-1/2":
                        engine1_points += 0.5
                        engine2_points += 0.5

                    # Save to database
                    save_game_to_db(
                        game_result.white_name, game_result.black_name,
                        game_result.result, f"{game_result.time_per_move:.2f}s/move",
                        game_result.opening_name, game_result.opening_fen,
                        game_result.pgn,
                        time_per_move_ms=int(game_result.time_per_move * 1000),
                        hostname=hostname
                    )

                    # Progress update with NPS
                    nps_info = ""
                    if game_result.white_nps or game_result.black_nps:
                        w_nps = f"{game_result.white_nps // 1000}k" if game_result.white_nps else "?"
                        b_nps = f"{game_result.black_nps // 1000}k" if game_result.black_nps else "?"
                        nps_info = f" [NPS: {w_nps}/{b_nps}]"
                    print(f"Game {completed_games:3d}/{num_games}: "
                          f"{game_result.white_name} vs {game_result.black_name} -> {game_result.result}{nps_info}  "
                          f"| Score: {engine1_name} {engine1_points:.1f} - {engine2_points:.1f} {engine2_name}",
                          flush=True)

                except Exception as e:
                    print(f"  Error in game: {e}", flush=True)
                    results["*"] += 1
                    completed_games += 1
    else:
        # Sequential execution (original behavior)
        for config in game_configs:
            result, game = play_game(
                config.white_path, config.black_path,
                config.white_name, config.black_name,
                config.time_per_move, config.opening_fen, config.opening_name,
                config.white_uci_options, config.black_uci_options
            )
            results[result] += 1

            # Save to database with full PGN
            save_game_to_db(config.white_name, config.black_name, result,
                            f"{config.time_per_move:.2f}s/move",
                            config.opening_name, config.opening_fen, str(game),
                            time_per_move_ms=int(config.time_per_move * 1000),
                            hostname=hostname)

            # Calculate points
            if result == "1-0":
                if config.is_engine1_white:
                    engine1_points += 1
                else:
                    engine2_points += 1
            elif result == "0-1":
                if config.is_engine1_white:
                    engine2_points += 1
                else:
                    engine1_points += 1
            elif result == "1/2-1/2":
                engine1_points += 0.5
                engine2_points += 0.5

            # Live update with NPS
            completed_games += 1
            opening_info = f" [{config.opening_name}]" if config.opening_name else ""
            # Extract NPS from game headers
            white_nps = int(game.headers.get("WhiteNPS", 0)) or None
            black_nps = int(game.headers.get("BlackNPS", 0)) or None
            nps_info = ""
            if white_nps or black_nps:
                w_nps = f"{white_nps // 1000}k" if white_nps else "?"
                b_nps = f"{black_nps // 1000}k" if black_nps else "?"
                nps_info = f" [NPS: {w_nps}/{b_nps}]"
            print(f"Game {completed_games:3d}/{num_games}: {config.white_name} vs {config.black_name} -> {result}{nps_info}{opening_info}  "
                  f"| Score: {engine1_name} {engine1_points:.1f} - {engine2_points:.1f} {engine2_name}",
                  flush=True)

    # Calculate Elo difference for this match
    # From engine1's perspective: wins are when engine1 won
    engine1_wins = int(engine1_points - (results["1/2-1/2"] * 0.5))
    engine1_losses = int(engine2_points - (results["1/2-1/2"] * 0.5))
    elo_diff, elo_error = calculate_elo_difference(
        engine1_wins, engine1_losses, results["1/2-1/2"]
    )

    # Print summary
    print(f"\n{'='*70}")
    print("FINAL RESULTS")
    print(f"{'='*70}")
    print(f"{engine1_name}: {engine1_points:.1f} points")
    print(f"{engine2_name}: {engine2_points:.1f} points")
    print(f"\nResults: +{engine1_wins} -{engine1_losses} ={results['1/2-1/2']}")
    win_rate = engine1_points / num_games * 100
    print(f"Win rate: {win_rate:.1f}%")

    if elo_diff > 0:
        print(f"\nMatch Elo difference: {engine1_name} is +{elo_diff:.0f} (±{elo_error:.0f}) stronger")
    else:
        print(f"\nMatch Elo difference: {engine2_name} is +{-elo_diff:.0f} (±{elo_error:.0f}) stronger")

    # Show updated Elo ratings
    print(f"\nUpdated Elo ratings:")
    for name in [engine1_name, engine2_name]:
        data = elo_ratings[name]
        delta = data["elo"] - start_elo[name]
        prov = "?" if data["games"] < PROVISIONAL_GAMES else ""
        print(f"  {name}: {data['elo']:.0f} ({delta:+.0f}) - {data['games']} games{prov}")

    print(f"{'='*70}\n")

    return {
        engine1_name: engine1_points,
        engine2_name: engine2_points,
        "results": results,
        "elo_diff": elo_diff,
        "elo_error": elo_error
    }


def print_league_table(competitors: set[str], games_this_comp: dict[str, int],
                       points_this_comp: dict[str, float], round_num: int, is_final: bool = False,
                       competitors_only: bool = False, game_num: int = 0, total_games: int = 0,
                       session_start_elo: dict[str, float] = None):
    """
    Print the league standings.

    Fetches fresh Elo ratings from the database to ensure accuracy across
    multiple concurrent competitions.

    Args:
        competitors: Set of engine names in the current competition
        games_this_comp: Dict of games played this competition per engine
        points_this_comp: Dict of points scored this competition per engine
        round_num: Current round number
        is_final: Whether this is the final standings
        competitors_only: If True, only show engines in current competition
        game_num: Current game number (for per-game display)
        total_games: Total games in competition (for per-game display)
        session_start_elo: Dict of starting Elo for each engine this session
    """
    # Fetch fresh ratings from database
    ratings = load_elo_ratings()

    if competitors_only:
        # Compact table showing only competitors, sorted by points
        sorted_competitors = sorted(
            [(name, points_this_comp.get(name, 0), ratings.get(name, {"elo": 0})["elo"]) for name in competitors],
            key=lambda x: (-x[1], -x[2])  # Sort by points desc, then Elo desc
        )

        print(f"\n{'='*70}")
        print(f"STANDINGS ({game_num}/{total_games} games)")
        print(f"{'='*70}")
        print(f"{'#':<4}{'Engine':<28}{'Points':<10}{'Elo':<10}{'Change':<10}")
        print(f"{'-'*70}")

        for rank, (name, points, elo) in enumerate(sorted_competitors, 1):
            engine_data = ratings.get(name, {"games": 0})
            prov = "?" if engine_data["games"] < PROVISIONAL_GAMES else ""
            if session_start_elo and name in session_start_elo:
                change = elo - session_start_elo[name]
                change_str = f"{change:+.0f}"
            else:
                change_str = ""
            print(f"{rank:<4}{name:<28}{points:<10.1f}{elo:<10.0f}{change_str:<10}{prov}")

        print(f"{'='*70}")
        return

    # Full table (original behavior)
    header = "FINAL LEAGUE STANDINGS" if is_final else f"STANDINGS AFTER ROUND {round_num}"
    print(f"\n{'='*95}")
    print(header)
    print(f"{'='*95}")
    print(f"{'Elo#':<6}{'Engine':<28}{'Elo':<10}{'Comp#':<7}{'Points':<10}{'Games':<8}{'Total':<8}{'Status':<10}")
    print(f"{'-'*95}")

    # Sort all engines by Elo (descending)
    sorted_engines = sorted(ratings.items(), key=lambda x: -x[1]["elo"])

    # Calculate competition rankings (by points)
    comp_rankings = {}
    if competitors:
        sorted_by_points = sorted(
            [(name, points_this_comp.get(name, 0)) for name in competitors],
            key=lambda x: -x[1]
        )
        for rank, (name, _) in enumerate(sorted_by_points, 1):
            comp_rankings[name] = rank

    for elo_rank, (name, data) in enumerate(sorted_engines, 1):
        elo = data["elo"]
        total_games_played = data["games"]
        comp_games = games_this_comp.get(name, 0)
        points = points_this_comp.get(name, 0)

        # Highlight current competitors
        if name in competitors:
            marker = "*"
            comp_rank = str(comp_rankings[name])
            status = "playing" if not is_final else "done"
        else:
            marker = " "
            comp_rank = "-"
            status = ""
            points = "-"

        # Mark provisional ratings
        if total_games_played < PROVISIONAL_GAMES:
            prov = "?"
        else:
            prov = ""

        display_name = f"{marker}{name}"
        points_str = f"{points:.1f}" if isinstance(points, float) else points
        print(f"{elo_rank:<6}{display_name:<28}{elo:<10.0f}{comp_rank:<7}{points_str:<10}{comp_games:<8}{total_games_played:<8}{status}{prov}")

    print(f"{'='*95}")
    print(f"  Elo# = overall ranking, Comp# = current competition ranking")
    print(f"  * = in current competition, ? = provisional rating (<{PROVISIONAL_GAMES} games)")
    print()


def run_epd(engine_names: list[str], engine_dir: Path, epd_file: Path,
            time_per_move: float, results_dir: Path):
    """
    Run a competition using positions from an EPD file.
    Each position is played twice (once with each engine as white).
    For 2 engines, it's head-to-head. For 3+ engines, it's round-robin per position.
    """
    # Load EPD positions
    positions = load_epd_positions(epd_file)
    if not positions:
        print(f"Error: No valid positions found in {epd_file}")
        sys.exit(1)

    print(f"\n{'='*70}")
    print(f"EPD Competition: {epd_file.name}")
    print(f"Positions: {len(positions)}")
    print(f"Engines: {', '.join(engine_names)}")
    print(f"Time: {time_per_move}s/move")
    print(f"Note: Elo ratings NOT updated (EPD mode uses specialized positions)")
    print(f"{'='*70}")

    # Get engine info
    engine_info = {}
    for name in engine_names:
        path, uci_options = get_engine_info(name, engine_dir)
        engine_info[name] = {"path": path, "uci_options": uci_options}

    # Load persistent Elo ratings
    elo_ratings = load_elo_ratings()

    # Initialize engines if needed
    for name in engine_names:
        if name not in elo_ratings:
            initial = get_initial_elo(name)
            elo_ratings[name] = {"elo": initial, "games": 0}

    # Show starting Elo
    print(f"\nStarting Elo:")
    for name in engine_names:
        data = elo_ratings[name]
        prov = "?" if data["games"] < PROVISIONAL_GAMES else ""
        print(f"  {name}: {data['elo']:.0f} ({data['games']} games){prov}")
    print(flush=True)

    # Create pairings
    pairings = list(combinations(engine_names, 2))

    # Calculate total games: positions * 2 (both colors) * pairings
    total_games = len(positions) * 2 * len(pairings)
    print(f"Total games: {total_games} ({len(positions)} positions x 2 colors x {len(pairings)} pairings)")
    print()

    # Session tracking
    session_points = {name: 0.0 for name in engine_names}
    session_games = {name: 0 for name in engine_names}

    game_num = 0

    # Iterate through each position
    for pos_idx, (fen, pos_id) in enumerate(positions):
        print(f"\n--- Position {pos_idx + 1}/{len(positions)}: {pos_id} ---")

        # For each pairing, play the position twice (swap colors)
        for engine1_name, engine2_name in pairings:
            engine1 = engine_info[engine1_name]
            engine2 = engine_info[engine2_name]

            # Game 1: engine1 as white
            game_num += 1
            result, game = play_game(
                engine1["path"], engine2["path"],
                engine1_name, engine2_name,
                time_per_move, fen, pos_id,
                engine1["uci_options"], engine2["uci_options"]
            )

            # Save to database with full PGN (not rated - EPD test games)
            hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())
            save_game_to_db(engine1_name, engine2_name, result, f"{time_per_move}s/move",
                            pos_id, fen, str(game), is_rated=False,
                            time_per_move_ms=int(time_per_move * 1000),
                            hostname=hostname)

            # Update session stats
            session_games[engine1_name] += 1
            session_games[engine2_name] += 1
            if result == "1-0":
                session_points[engine1_name] += 1
            elif result == "0-1":
                session_points[engine2_name] += 1
            elif result == "1/2-1/2":
                session_points[engine1_name] += 0.5
                session_points[engine2_name] += 0.5

            print(f"  Game {game_num}/{total_games}: {engine1_name} vs {engine2_name} -> {result}")

            # Game 2: engine2 as white (swap colors)
            game_num += 1
            result, game = play_game(
                engine2["path"], engine1["path"],
                engine2_name, engine1_name,
                time_per_move, fen, pos_id,
                engine2["uci_options"], engine1["uci_options"]
            )

            # Save to database with full PGN (not rated - EPD test games)
            hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())
            save_game_to_db(engine2_name, engine1_name, result, f"{time_per_move}s/move",
                            pos_id, fen, str(game), is_rated=False,
                            time_per_move_ms=int(time_per_move * 1000),
                            hostname=hostname)

            # Update session stats
            session_games[engine1_name] += 1
            session_games[engine2_name] += 1
            if result == "1-0":
                session_points[engine2_name] += 1
            elif result == "0-1":
                session_points[engine1_name] += 1
            elif result == "1/2-1/2":
                session_points[engine1_name] += 0.5
                session_points[engine2_name] += 0.5

            print(f"  Game {game_num}/{total_games}: {engine2_name} vs {engine1_name} -> {result}")

        # Show standings after each position
        print_league_table(
            set(engine_names), session_games,
            session_points, 0, False, True, game_num, total_games
        )

    # Final summary
    print(f"\n{'='*70}")
    print("EPD COMPETITION COMPLETE")
    print(f"{'='*70}")
    print(f"Positions: {len(positions)}")
    print(f"Games: {total_games}")

    print(f"\nFinal Standings:")
    sorted_results = sorted(session_points.items(), key=lambda x: -x[1])
    for rank, (name, points) in enumerate(sorted_results, 1):
        games = session_games[name]
        prov = "?" if elo_ratings[name]["games"] < PROVISIONAL_GAMES else ""
        print(f"  {rank}. {name}: {points:.1f}/{games} (Elo: {elo_ratings[name]['elo']:.0f}{prov})")

    print(f"{'='*70}")


def run_league(engine_names: list[str], engine_dir: Path,
               games_per_pairing: int, time_per_move: float, results_dir: Path,
               use_opening_book: bool = True,
               time_low: float = None, time_high: float = None,
               concurrency: int = 1):
    """Run a round-robin league with interleaved pairings."""

    pairings = list(combinations(engine_names, 2))
    num_pairings = len(pairings)
    use_time_range = time_low is not None and time_high is not None

    # Games per pairing should be even (play each opening from both sides)
    if games_per_pairing % 2 != 0:
        games_per_pairing += 1
        print(f"Adjusted games per pairing to {games_per_pairing} (must be even)")

    # Number of complete rounds (each round = 2 games per pairing = one opening played both ways)
    num_rounds = games_per_pairing // 2
    total_games = num_rounds * num_pairings * 2

    # Load persistent Elo ratings
    elo_ratings = load_elo_ratings()

    # Initialize new engines with average Elo
    competitors = set(engine_names)
    for name in engine_names:
        if name not in elo_ratings:
            initial = get_initial_elo(name)
            elo_ratings[name] = {"elo": initial, "games": 0}

    # Track games and points in this competition
    games_this_comp = {name: 0 for name in engine_names}
    points_this_comp = {name: 0.0 for name in engine_names}

    print(f"\n{'='*95}")
    print("ROUND ROBIN TOURNAMENT")
    print(f"{'='*95}")
    print(f"Engines: {', '.join(engine_names)}")
    print(f"Pairings: {num_pairings}")
    print(f"Games per pairing: {games_per_pairing}")
    print(f"Rounds: {num_rounds} (each round = 2 games per pairing)")
    print(f"Total games: {total_games}")
    if use_time_range:
        print(f"Time: {time_low}-{time_high}s/move (random per round)")
    else:
        print(f"Time: {time_per_move}s/move")
    if use_opening_book:
        print(f"Opening book: {len(OPENING_BOOK)} positions")
    print(f"{'='*95}")

    # Show starting Elo ratings
    print("\nStarting Elo ratings for competitors:")
    for name in sorted(engine_names, key=lambda n: -elo_ratings[n]["elo"]):
        data = elo_ratings[name]
        prov = "?" if data["games"] < PROVISIONAL_GAMES else ""
        print(f"  {name}: {data['elo']:.0f} ({data['games']} games){prov}")
    print()

    # Initialize head-to-head tracking (for summary at end)
    head_to_head = {}
    for e1, e2 in pairings:
        key = (e1, e2) if e1 < e2 else (e2, e1)
        head_to_head[key] = (0, 0, 0)

    # Prepare openings
    if use_opening_book:
        openings = OPENING_BOOK.copy()
        random.shuffle(openings)
        # Extend if we need more
        while len(openings) < num_rounds:
            extra = OPENING_BOOK.copy()
            random.shuffle(extra)
            openings.extend(extra)
    else:
        openings = [(None, None)] * num_rounds

    # Get engine paths and UCI options
    engine_info = {name: get_engine_info(name, engine_dir) for name in engine_names}

    game_num = 0

    # Play rounds
    for round_idx in range(num_rounds):
        opening_fen, opening_name = openings[round_idx]

        # Select time for this round
        if use_time_range:
            round_time = random.uniform(time_low, time_high)
        else:
            round_time = time_per_move

        print(f"\n--- Round {round_idx + 1}/{num_rounds}: {opening_name or 'Starting position'} ({round_time:.2f}s/move) ---\n")

        # Play each pairing twice (once per color) for this opening
        for pairing_idx, (engine1, engine2) in enumerate(pairings):
            match_label = f"Match {pairing_idx + 1}/{num_pairings}"

            for color_swap in [False, True]:
                game_num += 1

                if color_swap:
                    white, black = engine2, engine1
                else:
                    white, black = engine1, engine2

                white_path, white_uci = engine_info[white]
                black_path, black_uci = engine_info[black]

                result, game = play_game(white_path, black_path, white, black,
                                         round_time, opening_fen, opening_name,
                                         white_uci, black_uci)

                # Save to database with full PGN
                hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())
                save_game_to_db(white, black, result, f"{round_time:.2f}s/move",
                                opening_name, opening_fen, str(game),
                                time_per_move_ms=int(round_time * 1000),
                                hostname=hostname)

                # Update games and points for this competition
                games_this_comp[white] += 1
                games_this_comp[black] += 1

                if result == "1-0":
                    points_this_comp[white] += 1.0
                elif result == "0-1":
                    points_this_comp[black] += 1.0
                elif result == "1/2-1/2":
                    points_this_comp[white] += 0.5
                    points_this_comp[black] += 0.5

                # Update head-to-head tracking
                key = (engine1, engine2) if engine1 < engine2 else (engine2, engine1)
                e1_wins, e2_wins, draws = head_to_head[key]

                if result == "1-0":
                    if white == key[0]:
                        e1_wins += 1
                    else:
                        e2_wins += 1
                elif result == "0-1":
                    if black == key[0]:
                        e1_wins += 1
                    else:
                        e2_wins += 1
                elif result == "1/2-1/2":
                    draws += 1

                head_to_head[key] = (e1_wins, e2_wins, draws)

                # Print game result
                color_label = "(colors swapped)" if color_swap else ""
                print(f"Game {game_num:3d}/{total_games} {match_label}: {white} vs {black} -> {result} {color_label}")

                # Print compact standings after each game
                print_league_table(competitors, games_this_comp, points_this_comp,
                                   round_idx + 1, competitors_only=True, game_num=game_num, total_games=total_games)

    # Print final standings (full table)
    print_league_table(competitors, games_this_comp, points_this_comp, num_rounds, is_final=True)

    # Print head-to-head results
    print(f"{'='*95}")
    print("HEAD-TO-HEAD RESULTS (this competition)")
    print(f"{'='*95}")
    for (e1, e2), (w1, w2, d) in sorted(head_to_head.items()):
        total = w1 + w2 + d
        if total > 0:
            elo_diff, elo_err = calculate_elo_difference(w1, w2, d)
            print(f"{e1} vs {e2}: +{w1} -{w2} ={d}  (Elo diff: {elo_diff:+.0f} ±{elo_err:.0f})")
    print(f"{'='*95}\n")


def run_gauntlet(challenger_name: str, engine_dir: Path,
                 num_rounds: int, time_per_move: float, results_dir: Path,
                 time_low: float = None, time_high: float = None,
                 engine_type: str = None, include_inactive: bool = False,
                 concurrency: int = 1):
    """
    Test a challenger engine against all other engines in the engines directory.
    Plays in rounds: each round consists of 2 games (1 as white, 1 as black) against each opponent.
    Each game uses a random opening.

    Args:
        engine_type: Filter opponents by type ('rusty' or 'stockfish', None = all)
        include_inactive: If True, include inactive engines as opponents
    """
    # Find all engines matching filters except the challenger
    all_engines = get_active_engines(engine_dir, engine_type, include_inactive)
    opponents = [e for e in all_engines if e != challenger_name]
    use_time_range = time_low is not None and time_high is not None

    if not opponents:
        print(f"Error: No opponent engines found in {engine_dir}")
        sys.exit(1)

    games_per_opponent = num_rounds * 2  # 2 games per round (1 white, 1 black)
    total_games = len(opponents) * games_per_opponent

    # Load persistent Elo ratings
    elo_ratings = load_elo_ratings()

    # Initialize challenger if needed
    if challenger_name not in elo_ratings:
        initial = get_initial_elo(challenger_name)
        elo_ratings[challenger_name] = {"elo": initial, "games": 0}

    # Initialize opponents if needed
    for opponent in opponents:
        if opponent not in elo_ratings:
            initial = get_initial_elo(opponent)
            elo_ratings[opponent] = {"elo": initial, "games": 0}

    # Store starting Elo
    start_elo = elo_ratings[challenger_name]["elo"]

    print(f"\n{'='*70}")
    print("GAUNTLET TEST")
    print(f"{'='*70}")
    print(f"Challenger: {challenger_name}")
    print(f"Opponents: {len(opponents)}")
    print(f"Rounds: {num_rounds} (2 games per opponent per round)")
    print(f"Games per opponent: {games_per_opponent}")
    print(f"Total games: {total_games}")
    if use_time_range:
        print(f"Time: {time_low}-{time_high}s/move (random per round)")
    else:
        print(f"Time: {time_per_move}s/move")
    print(f"Opening book: {len(OPENING_BOOK)} positions (random selection)")
    print(f"{'='*70}")

    # Show starting Elo
    print(f"\nChallenger starting Elo: {start_elo:.0f} ({elo_ratings[challenger_name]['games']} games)")
    print(f"\nOpponents:")
    for opp in sorted(opponents, key=lambda n: -elo_ratings.get(n, {}).get("elo", DEFAULT_ELO)):
        data = elo_ratings.get(opp, {"elo": DEFAULT_ELO, "games": 0})
        prov = "?" if data["games"] < PROVISIONAL_GAMES else ""
        print(f"  {opp}: {data['elo']:.0f} ({data['games']} games){prov}")
    print()

    # Get engine paths and UCI options
    challenger_path, challenger_uci = get_engine_info(challenger_name, engine_dir)
    opponent_info = {opp: get_engine_info(opp, engine_dir) for opp in opponents}

    # Track results per opponent
    results_per_opponent = {opp: {"wins": 0, "losses": 0, "draws": 0} for opp in opponents}
    game_num = 0

    # Play in rounds
    for round_idx in range(num_rounds):
        # Select time for this round
        if use_time_range:
            round_time = random.uniform(time_low, time_high)
        else:
            round_time = time_per_move

        print(f"\n--- Round {round_idx + 1}/{num_rounds} ({round_time:.2f}s/move) ---\n")

        for opponent in opponents:
            opponent_path, opponent_uci = opponent_info[opponent]

            # Game 1: Challenger as white
            game_num += 1
            opening_fen, opening_name = random.choice(OPENING_BOOK)

            result, game = play_game(challenger_path, opponent_path,
                                      challenger_name, opponent,
                                      round_time, opening_fen, opening_name,
                                      challenger_uci, opponent_uci)

            # Save to database with full PGN
            hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())
            save_game_to_db(challenger_name, opponent, result, f"{round_time:.2f}s/move",
                            opening_name, opening_fen, str(game),
                            time_per_move_ms=int(round_time * 1000),
                            hostname=hostname)

            if result == "1-0":
                results_per_opponent[opponent]["wins"] += 1
            elif result == "0-1":
                results_per_opponent[opponent]["losses"] += 1
            elif result == "1/2-1/2":
                results_per_opponent[opponent]["draws"] += 1

            print(f"Game {game_num:3d}/{total_games}: {challenger_name} (W) vs {opponent} -> {result}  [{opening_name}]")

            # Game 2: Challenger as black
            game_num += 1
            opening_fen, opening_name = random.choice(OPENING_BOOK)

            result, game = play_game(opponent_path, challenger_path,
                                      opponent, challenger_name,
                                      round_time, opening_fen, opening_name,
                                      opponent_uci, challenger_uci)

            # Save to database with full PGN
            hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())
            save_game_to_db(opponent, challenger_name, result, f"{round_time:.2f}s/move",
                            opening_name, opening_fen, str(game),
                            time_per_move_ms=int(round_time * 1000),
                            hostname=hostname)

            if result == "0-1":
                results_per_opponent[opponent]["wins"] += 1
            elif result == "1-0":
                results_per_opponent[opponent]["losses"] += 1
            elif result == "1/2-1/2":
                results_per_opponent[opponent]["draws"] += 1

            print(f"Game {game_num:3d}/{total_games}: {opponent} vs {challenger_name} (B) -> {result}  [{opening_name}]")

        # Show standings after each round
        print(f"\n--- Standings after round {round_idx + 1} ---")
        total_w = sum(r["wins"] for r in results_per_opponent.values())
        total_l = sum(r["losses"] for r in results_per_opponent.values())
        total_d = sum(r["draws"] for r in results_per_opponent.values())
        total_score = total_w + total_d * 0.5
        total_played = total_w + total_l + total_d
        current_elo = elo_ratings[challenger_name]["elo"]
        elo_change = current_elo - start_elo
        print(f"Overall: +{total_w} -{total_l} ={total_d}  Score: {total_score}/{total_played} ({total_score/total_played*100:.1f}%)  Elo: {current_elo:.0f} ({elo_change:+.0f})")

    # Print final summary
    print(f"\n{'='*70}")
    print("GAUNTLET RESULTS")
    print(f"{'='*70}")

    total_wins = sum(r["wins"] for r in results_per_opponent.values())
    total_losses = sum(r["losses"] for r in results_per_opponent.values())
    total_draws = sum(r["draws"] for r in results_per_opponent.values())
    total_score = total_wins + total_draws * 0.5
    total_played = total_wins + total_losses + total_draws

    print(f"\nChallenger: {challenger_name}")
    print(f"Overall: +{total_wins} -{total_losses} ={total_draws}  Score: {total_score}/{total_played} ({total_score/total_played*100:.1f}%)")

    overall_elo_diff, overall_elo_err = calculate_elo_difference(total_wins, total_losses, total_draws)
    print(f"Overall Elo performance: {overall_elo_diff:+.0f} ±{overall_elo_err:.0f}")

    print(f"\nResults per opponent:")
    print(f"{'Opponent':<30} {'W':>4} {'L':>4} {'D':>4} {'Score':>8} {'%':>7} {'Elo diff':>10}")
    print("-" * 70)

    for opponent in sorted(results_per_opponent.keys(), key=lambda o: elo_ratings.get(o, {}).get("elo", 0), reverse=True):
        r = results_per_opponent[opponent]
        w, l, d = r["wins"], r["losses"], r["draws"]
        score = w + d * 0.5
        total = w + l + d
        pct = score / total * 100 if total > 0 else 0
        elo_diff, _ = calculate_elo_difference(w, l, d)
        print(f"{opponent:<30} {w:>4} {l:>4} {d:>4} {score:>5.1f}/{total:<2} {pct:>6.1f}% {elo_diff:>+9.0f}")

    # Show Elo change
    final_elo = elo_ratings[challenger_name]["elo"]
    elo_change = final_elo - start_elo
    final_games = elo_ratings[challenger_name]["games"]
    prov = "?" if final_games < PROVISIONAL_GAMES else ""

    print(f"\nChallenger Elo: {start_elo:.0f} -> {final_elo:.0f} ({elo_change:+.0f}) - {final_games} games{prov}")
    print(f"{'='*70}\n")


def run_random(engine_dir: Path, num_matches: int, time_per_move: float, results_dir: Path, weighted: bool = False,
               time_low: float = None, time_high: float = None,
               engine_type: str = None, include_inactive: bool = False,
               concurrency: int = 1):
    """
    Randomly select pairs of engines and play 2-game matches (1 white, 1 black).
    Each game uses a random opening.

    If weighted=True, the first engine is selected with bias toward fewer games
    (weight = 1 / (games + 1)), while the second engine is chosen purely at random.
    This ensures under-tested engines get priority but play diverse opponents
    for better Elo calibration.

    If time_low and time_high are provided, randomly select a time for each match
    from that range. Otherwise use time_per_move.

    The active engines list is re-fetched before each match, allowing live
    enable/disable of engines without restarting the competition.

    Args:
        engine_type: Filter engines by type ('rusty' or 'stockfish', None = all)
        include_inactive: If True, include inactive engines
    """
    use_time_range = time_low is not None and time_high is not None
    # Find all engines matching filters
    all_engines = get_active_engines(engine_dir, engine_type, include_inactive)

    if len(all_engines) < 2:
        print(f"Error: Need at least 2 engines in {engine_dir}, found {len(all_engines)}")
        sys.exit(1)

    total_games = num_matches * 2

    # Load persistent Elo ratings
    elo_ratings = load_elo_ratings()

    # Initialize any missing engines
    for engine in all_engines:
        if engine not in elo_ratings:
            initial_elo = get_initial_elo(engine)
            elo_ratings[engine] = {"elo": initial_elo, "games": 0}

    print(f"\n{'='*70}")
    print(f"RANDOM MODE{' (WEIGHTED)' if weighted else ''}")
    print(f"{'='*70}")
    print(f"Active engines: {len(all_engines)}")
    if weighted:
        print("Selection: first engine weighted (fewer games = higher chance), second random")
    print(f"Matches: {num_matches} (2 games each = {total_games} total games)")
    if use_time_range:
        print(f"Time: {time_low}-{time_high}s/move (random per match)")
    else:
        print(f"Time: {time_per_move}s/move")
    print(f"Opening book: {len(OPENING_BOOK)} positions (random selection)")
    if concurrency > 1:
        print(f"Concurrency: {concurrency} games in parallel")
    print(f"{'='*70}")

    game_num = 0

    # Track session stats for standings display
    session_engines = set()
    session_games = {}  # engine -> games played this session
    session_points = {}  # engine -> points this session
    session_start_elo = {}  # engine -> Elo at start of session

    hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())

    def refresh_engines_and_check():
        """Re-fetch engines and handle changes. Returns current engine list or None if < 2."""
        nonlocal all_engines
        current_engines = get_active_engines(engine_dir, engine_type, include_inactive)

        if set(current_engines) != set(all_engines):
            added = set(current_engines) - set(all_engines)
            removed = set(all_engines) - set(current_engines)
            if added:
                print(f"\n[Engine(s) enabled: {', '.join(sorted(added))}]")
                for engine in added:
                    if engine not in elo_ratings:
                        initial_elo = get_initial_elo(engine)
                        elo_ratings[engine] = {"elo": initial_elo, "games": 0}
            if removed:
                print(f"\n[Engine(s) disabled: {', '.join(sorted(removed))}]")
            all_engines = current_engines

        if len(all_engines) < 2:
            return None
        return all_engines

    def pick_engines():
        """Pick two engines based on weighted or uniform random selection."""
        if weighted:
            weights = [1.0 / (elo_ratings[e]["games"] + 1) for e in all_engines]
            engine1 = random.choices(all_engines, weights=weights, k=1)[0]
            remaining = [e for e in all_engines if e != engine1]
            engine2 = random.choice(remaining)
        else:
            engine1, engine2 = random.sample(all_engines, 2)
        return engine1, engine2

    def process_game_result(game_result, engine1, engine2):
        """Process a completed game result and update session stats."""
        nonlocal game_num

        game_num += 1
        white = game_result.white_name
        black = game_result.black_name
        result = game_result.result

        # Track session stats
        for eng in [engine1, engine2]:
            session_engines.add(eng)
            if eng not in session_games:
                session_games[eng] = 0
            session_games[eng] += 1
            if eng not in session_points:
                session_points[eng] = 0.0

        if result == "1-0":
            session_points[white] += 1.0
        elif result == "0-1":
            session_points[black] += 1.0
        elif result == "1/2-1/2":
            session_points[white] += 0.5
            session_points[black] += 0.5

        # NPS info
        nps_info = ""
        if game_result.white_nps or game_result.black_nps:
            w_nps = f"{game_result.white_nps // 1000}k" if game_result.white_nps else "?"
            b_nps = f"{game_result.black_nps // 1000}k" if game_result.black_nps else "?"
            nps_info = f" [NPS: {w_nps}/{b_nps}]"

        print(f"Game {game_num:3d}/{total_games}: {white} vs {black} -> {result}{nps_info}")

    if concurrency > 1:
        # Parallel execution: batch matches and run games concurrently
        match_idx = 0
        while match_idx < num_matches:
            # Refresh engine list periodically
            engines = refresh_engines_and_check()
            if engines is None:
                print(f"\nWarning: Only {len(all_engines)} active engine(s), need at least 2. Waiting...")
                time.sleep(5)
                continue

            # Determine batch size (number of matches to run in parallel)
            # Each match = 2 games, so batch_size matches = batch_size * 2 games
            batch_size = min(concurrency // 2 or 1, num_matches - match_idx)
            game_configs = []

            print(f"\n--- Batch: Matches {match_idx + 1}-{match_idx + batch_size} of {num_matches} ---\n")

            # Generate configs for this batch
            for i in range(batch_size):
                engine1, engine2 = pick_engines()
                engine1_path, engine1_uci = get_engine_info(engine1, engine_dir)
                engine2_path, engine2_uci = get_engine_info(engine2, engine_dir)

                if use_time_range:
                    match_time = random.uniform(time_low, time_high)
                else:
                    match_time = time_per_move

                # Capture starting ELO
                for eng in [engine1, engine2]:
                    if eng not in session_start_elo:
                        ratings = load_elo_ratings()
                        session_start_elo[eng] = ratings.get(eng, {"elo": get_initial_elo(eng)})["elo"]

                # Convert paths to strings for pickling
                e1_path_str = str(engine1_path) if not isinstance(engine1_path, list) else engine1_path
                e2_path_str = str(engine2_path) if not isinstance(engine2_path, list) else engine2_path

                # Game 1: engine1 as white
                opening_fen1, opening_name1 = random.choice(OPENING_BOOK)
                config1 = GameConfig(
                    game_index=len(game_configs),
                    white_name=engine1, black_name=engine2,
                    white_path=e1_path_str, black_path=e2_path_str,
                    white_uci_options=engine1_uci, black_uci_options=engine2_uci,
                    time_per_move=match_time,
                    opening_fen=opening_fen1, opening_name=opening_name1,
                    is_engine1_white=True
                )
                game_configs.append((config1, engine1, engine2))

                # Game 2: engine2 as white
                opening_fen2, opening_name2 = random.choice(OPENING_BOOK)
                config2 = GameConfig(
                    game_index=len(game_configs),
                    white_name=engine2, black_name=engine1,
                    white_path=e2_path_str, black_path=e1_path_str,
                    white_uci_options=engine2_uci, black_uci_options=engine1_uci,
                    time_per_move=match_time,
                    opening_fen=opening_fen2, opening_name=opening_name2,
                    is_engine1_white=False
                )
                game_configs.append((config2, engine1, engine2))

            # Run games in parallel
            with ProcessPoolExecutor(max_workers=concurrency) as executor:
                futures = {executor.submit(play_game_from_config, cfg): (cfg, e1, e2)
                           for cfg, e1, e2 in game_configs}

                for future in as_completed(futures):
                    try:
                        game_result = future.result()
                        cfg, engine1, engine2 = futures[future]

                        # Save to database
                        save_game_to_db(
                            game_result.white_name, game_result.black_name,
                            game_result.result, f"{game_result.time_per_move:.2f}s/move",
                            game_result.opening_name, game_result.opening_fen,
                            game_result.pgn,
                            time_per_move_ms=int(game_result.time_per_move * 1000),
                            hostname=hostname
                        )

                        process_game_result(game_result, engine1, engine2)

                    except Exception as e:
                        print(f"  Error in game: {e}", flush=True)
                        game_num += 1

            # Print standings after batch
            print_league_table(session_engines, session_games, session_points,
                               0, competitors_only=True, game_num=game_num, total_games=total_games,
                               session_start_elo=session_start_elo)

            match_idx += batch_size

    else:
        # Sequential execution (original behavior)
        for match_idx in range(num_matches):
            engines = refresh_engines_and_check()
            if engines is None:
                print(f"\nWarning: Only {len(all_engines)} active engine(s), need at least 2. Waiting...")
                time.sleep(5)
                continue

            engine1, engine2 = pick_engines()
            engine1_path, engine1_uci = get_engine_info(engine1, engine_dir)
            engine2_path, engine2_uci = get_engine_info(engine2, engine_dir)

            if use_time_range:
                match_time = random.uniform(time_low, time_high)
            else:
                match_time = time_per_move

            print(f"\n--- Match {match_idx + 1}/{num_matches}: {engine1} vs {engine2} ({match_time:.2f}s/move) ---\n")

            # Capture starting ELO
            for eng in [engine1, engine2]:
                if eng not in session_start_elo:
                    ratings = load_elo_ratings()
                    session_start_elo[eng] = ratings.get(eng, {"elo": get_initial_elo(eng)})["elo"]

            # Game 1: engine1 as white
            opening_fen, opening_name = random.choice(OPENING_BOOK)
            result, game = play_game(engine1_path, engine2_path,
                                      engine1, engine2,
                                      match_time, opening_fen, opening_name,
                                      engine1_uci, engine2_uci)

            save_game_to_db(engine1, engine2, result, f"{match_time:.2f}s/move",
                            opening_name, opening_fen, str(game),
                            time_per_move_ms=int(match_time * 1000),
                            hostname=hostname)

            game_num += 1
            for eng in [engine1, engine2]:
                session_engines.add(eng)
                session_games[eng] = session_games.get(eng, 0) + 1
                if eng not in session_points:
                    session_points[eng] = 0.0

            if result == "1-0":
                session_points[engine1] += 1.0
            elif result == "0-1":
                session_points[engine2] += 1.0
            elif result == "1/2-1/2":
                session_points[engine1] += 0.5
                session_points[engine2] += 0.5

            # NPS from game
            white_nps = int(game.headers.get("WhiteNPS", 0)) or None
            black_nps = int(game.headers.get("BlackNPS", 0)) or None
            nps_info = ""
            if white_nps or black_nps:
                w_nps = f"{white_nps // 1000}k" if white_nps else "?"
                b_nps = f"{black_nps // 1000}k" if black_nps else "?"
                nps_info = f" [NPS: {w_nps}/{b_nps}]"

            print(f"Game {game_num:3d}/{total_games}: {engine1} (W) vs {engine2} -> {result}{nps_info}  [{opening_name}]")
            print_league_table(session_engines, session_games, session_points,
                               0, competitors_only=True, game_num=game_num, total_games=total_games,
                               session_start_elo=session_start_elo)

            # Game 2: engine2 as white
            opening_fen, opening_name = random.choice(OPENING_BOOK)
            result, game = play_game(engine2_path, engine1_path,
                                      engine2, engine1,
                                      match_time, opening_fen, opening_name,
                                      engine2_uci, engine1_uci)

            save_game_to_db(engine2, engine1, result, f"{match_time:.2f}s/move",
                            opening_name, opening_fen, str(game),
                            time_per_move_ms=int(match_time * 1000),
                            hostname=hostname)

            game_num += 1
            for eng in [engine1, engine2]:
                session_games[eng] = session_games.get(eng, 0) + 1
            if result == "1-0":
                session_points[engine2] += 1.0
            elif result == "0-1":
                session_points[engine1] += 1.0
            elif result == "1/2-1/2":
                session_points[engine1] += 0.5
                session_points[engine2] += 0.5

            # NPS from game
            white_nps = int(game.headers.get("WhiteNPS", 0)) or None
            black_nps = int(game.headers.get("BlackNPS", 0)) or None
            nps_info = ""
            if white_nps or black_nps:
                w_nps = f"{white_nps // 1000}k" if white_nps else "?"
                b_nps = f"{black_nps // 1000}k" if black_nps else "?"
                nps_info = f" [NPS: {w_nps}/{b_nps}]"

            print(f"Game {game_num:3d}/{total_games}: {engine2} (W) vs {engine1} -> {result}{nps_info}  [{opening_name}]")
            print_league_table(session_engines, session_games, session_points,
                               0, competitors_only=True, game_num=game_num, total_games=total_games,
                               session_start_elo=session_start_elo)

    # Print final standings
    print(f"\n{'='*70}")
    print("CURRENT ELO STANDINGS")
    print(f"{'='*70}")
    print(f"{'Rank':<6}{'Engine':<30}{'Elo':>8}{'Games':>8}")
    print("-" * 52)

    sorted_engines = sorted(elo_ratings.items(), key=lambda x: -x[1]["elo"])
    for rank, (name, data) in enumerate(sorted_engines, 1):
        prov = "?" if data["games"] < PROVISIONAL_GAMES else ""
        print(f"{rank:<6}{name:<30}{data['elo']:>8.0f}{data['games']:>7}{prov}")

    print(f"{'='*70}\n")


def solve_position(engine, board: chess.Board, position: EpdPosition,
                   timeout: float, score_tolerance: int) -> SolveResult:
    """
    Analyze position until engine finds expected move or timeout.

    Args:
        engine: chess.engine.SimpleEngine instance
        board: chess.Board set to the position
        position: EpdPosition with expected best moves
        timeout: Maximum time to search in seconds
        score_tolerance: Acceptable difference in centipawns for score validation

    Returns:
        SolveResult with solve status, time, and score validation
    """
    start_time = time.time()
    found_move = None
    found_time = None
    final_score = None
    final_depth = None
    last_move = None

    # Determine test type: bm (find best move) or am-only (avoid move)
    is_avoid_only = not position.best_moves and position.avoid_moves

    try:
        with engine.analysis(board, chess.engine.Limit(time=timeout)) as analysis:
            for info in analysis:
                elapsed = time.time() - start_time

                # Check if PV starts with an expected best move
                if "pv" in info and info["pv"]:
                    move = info["pv"][0]
                    move_san = board.san(move)
                    last_move = move_san

                    # Update score/depth
                    if "score" in info:
                        final_score = info["score"]
                    if "depth" in info:
                        final_depth = info["depth"]

                    if position.best_moves and move_san in position.best_moves:
                        # Found the correct move - record time and stop searching
                        found_move = move_san
                        found_time = elapsed
                        break
    except Exception as e:
        print(f"  Error during analysis: {e}")
        return SolveResult(
            solved=False,
            move_found=last_move,
            solve_time=None,
            final_depth=None,
            score=None,
            score_valid=None,
            timed_out=True
        )

    elapsed = time.time() - start_time

    # Validate score if 'ce' was specified
    score_valid = None
    if position.centipawn_eval is not None and final_score:
        if final_score.is_mate():
            # For mate positions, check if dm (direct mate) was specified
            if position.direct_mate is not None:
                mate_moves = final_score.white().mate()
                if mate_moves is not None:
                    score_valid = abs(mate_moves) <= position.direct_mate
            else:
                score_valid = False  # Mate != centipawn
        else:
            cp = final_score.white().score(mate_score=10000)
            if cp is not None:
                diff = abs(cp - position.centipawn_eval)
                score_valid = diff <= score_tolerance

    # Determine if solved based on test type
    if is_avoid_only:
        # For am-only positions: solved if final move is NOT in avoid_moves
        solved = last_move is not None and last_move not in position.avoid_moves
        return SolveResult(
            solved=solved,
            move_found=last_move,
            solve_time=elapsed if solved else None,
            final_depth=final_depth,
            score=final_score,
            score_valid=score_valid,
            timed_out=False  # am positions always run to completion
        )
    else:
        # For bm positions: solved if we found the expected move
        return SolveResult(
            solved=found_move is not None,
            move_found=found_move if found_move else last_move,
            solve_time=found_time,
            final_depth=final_depth,
            score=final_score,
            score_valid=score_valid,
            timed_out=found_move is None and elapsed >= timeout * 0.95
        )


def run_epd_solve_single(engine_names: list[str], engine_dir: Path, epd_file: Path,
                         position_selector: str, timeout: float, score_tolerance: int = 50):
    """
    Solve a single EPD position with verbose output showing all engine info lines.

    Args:
        engine_names: List of engine names to test
        engine_dir: Directory containing engine binaries
        epd_file: Path to EPD file
        position_selector: Position number (1-based) or position ID
        timeout: Maximum time in seconds
        score_tolerance: Acceptable difference in centipawns for score validation
    """
    # Load EPD positions
    positions = load_epd_for_solving(epd_file)
    if not positions:
        print(f"Error: No valid positions found in {epd_file}")
        sys.exit(1)

    # Filter to testable positions
    testable_positions = [p for p in positions if p.best_moves or p.avoid_moves]
    if not testable_positions:
        print(f"Error: No positions with 'bm' or 'am' operations found in {epd_file}")
        sys.exit(1)

    # Find the requested position
    target_position = None
    target_index = None

    # Try to match by position number first
    try:
        pos_num = int(position_selector)
        if 1 <= pos_num <= len(testable_positions):
            target_position = testable_positions[pos_num - 1]
            target_index = pos_num
    except ValueError:
        pass

    # If not found by number, try to match by ID
    if target_position is None:
        for idx, pos in enumerate(testable_positions):
            if pos.id == position_selector or position_selector in pos.id:
                target_position = pos
                target_index = idx + 1
                break

    if target_position is None:
        print(f"Error: Position '{position_selector}' not found in {epd_file}")
        print(f"Available positions: 1-{len(testable_positions)}")
        print("Position IDs:")
        for idx, pos in enumerate(testable_positions[:10]):
            print(f"  {idx + 1}: {pos.id}")
        if len(testable_positions) > 10:
            print(f"  ... and {len(testable_positions) - 10} more")
        sys.exit(1)

    # Determine test type
    is_avoid_only = not target_position.best_moves and target_position.avoid_moves
    test_type = 'am' if is_avoid_only else 'bm'
    expected_moves = target_position.avoid_moves if is_avoid_only else target_position.best_moves

    print(f"\n{'='*70}")
    print(f"SINGLE POSITION SOLVE: {target_position.id}")
    print(f"{'='*70}")
    print(f"FEN: {target_position.fen}")
    print(f"Expected: {test_type} {' '.join(expected_moves)}")
    if target_position.centipawn_eval is not None:
        print(f"Expected eval: {target_position.centipawn_eval:+d} cp")
    if target_position.direct_mate is not None:
        print(f"Direct mate: {target_position.direct_mate}")
    print(f"Timeout: {timeout}s")
    print(f"{'='*70}")

    board = chess.Board(target_position.fen)

    for engine_name in engine_names:
        print(f"\nEngine: {engine_name}")
        print("Searching with verbose output...\n")

        engine_path, engine_uci_options = get_engine_info(engine_name, engine_dir)
        cmd = engine_path if isinstance(engine_path, list) else str(engine_path)
        engine = chess.engine.SimpleEngine.popen_uci(cmd, stderr=subprocess.DEVNULL)

        if engine_uci_options:
            engine.configure(engine_uci_options)

        start_time = time.time()
        found_move = None
        found_time = None
        final_score = None
        final_depth = None
        last_move = None

        try:
            with engine.analysis(board, chess.engine.Limit(time=timeout)) as analysis:
                for info in analysis:
                    elapsed = time.time() - start_time

                    # Print verbose info line
                    parts = []
                    if "depth" in info:
                        parts.append(f"depth {info['depth']}")
                    if "seldepth" in info:
                        parts.append(f"seldepth {info['seldepth']}")
                    if "score" in info:
                        score = info["score"]
                        if score.is_mate():
                            parts.append(f"score mate {score.white().mate()}")
                        else:
                            cp = score.white().score(mate_score=10000)
                            parts.append(f"score cp {cp}")
                    if "nodes" in info:
                        parts.append(f"nodes {info['nodes']}")
                    if "nps" in info:
                        parts.append(f"nps {info['nps']}")
                    if "pv" in info and info["pv"]:
                        pv_moves = [board.san(m) for m in info["pv"][:8]]
                        parts.append(f"pv {' '.join(pv_moves)}")

                    if parts:
                        print(f"info {' '.join(parts)}")

                    # Check for solution
                    if "pv" in info and info["pv"]:
                        move = info["pv"][0]
                        move_san = board.san(move)
                        last_move = move_san

                        if "score" in info:
                            final_score = info["score"]
                        if "depth" in info:
                            final_depth = info["depth"]

                        if not is_avoid_only and move_san in expected_moves:
                            # Found best move - stop
                            found_move = move_san
                            found_time = elapsed
                            break

            # For avoid-move positions, check final result
            if is_avoid_only:
                if last_move and last_move not in expected_moves:
                    found_move = last_move
                    found_time = time.time() - start_time

        except Exception as e:
            print(f"\nError during analysis: {e}")

        engine.quit()

        # Print final result
        print(f"\n{'-'*70}")
        print("Final result:")
        if is_avoid_only:
            if found_move:
                print(f"  Move: {found_move} (avoided {', '.join(expected_moves)}) - PASS")
            else:
                print(f"  Move: {last_move} (should have avoided {', '.join(expected_moves)}) - FAIL")
        else:
            if found_move:
                print(f"  Move: {found_move} - PASS")
            else:
                print(f"  Move: {last_move or 'none'} (expected {', '.join(expected_moves)}) - FAIL")

        print(f"  Depth: {final_depth}")
        if final_score:
            if final_score.is_mate():
                print(f"  Score: Mate in {final_score.white().mate()}")
            else:
                cp = final_score.white().score(mate_score=10000)
                print(f"  Score: {cp:+d} cp")
        print(f"  Time: {found_time:.1f}s" if found_time else f"  Time: {time.time() - start_time:.1f}s (timeout)")
        print(f"  Status: {'SOLVED' if found_move else 'FAILED'}")
        print(f"{'='*70}")


def run_epd_solve(engine_names: list[str], engine_dir: Path, epd_file: Path,
                  timeout: float, results_dir: Path, score_tolerance: int = 50,
                  store_results: bool = True):
    """
    Run EPD solve test - measure time for engines to find correct moves.

    For each position:
    1. Start engine analysis
    2. Monitor PV until best move matches expected (or timeout)
    3. Record solve time and final evaluation
    4. Validate eval against expected 'ce' value if present

    Args:
        engine_names: List of engine names to test
        engine_dir: Directory containing engine binaries
        epd_file: Path to EPD file
        timeout: Maximum time per position in seconds
        results_dir: Directory for results (not used currently)
        score_tolerance: Acceptable difference in centipawns for score validation
        store_results: Whether to save results to database (default True)
    """
    # Load EPD positions
    positions = load_epd_for_solving(epd_file)
    if not positions:
        print(f"Error: No valid positions found in {epd_file}")
        sys.exit(1)

    # Filter positions that have best moves OR avoid moves defined
    testable_positions = [p for p in positions if p.best_moves or p.avoid_moves]
    if not testable_positions:
        print(f"Error: No positions with 'bm' or 'am' operations found in {epd_file}")
        print("EPD solve mode requires positions with expected best moves or avoid moves.")
        sys.exit(1)

    # Count position types
    bm_count = sum(1 for p in testable_positions if p.best_moves)
    am_only_count = sum(1 for p in testable_positions if not p.best_moves and p.avoid_moves)

    print(f"\n{'='*70}")
    print(f"EPD SOLVE TEST: {epd_file.name}")
    print(f"{'='*70}")
    print(f"Engines: {', '.join(engine_names)}")
    print(f"Positions: {len(testable_positions)} ({bm_count} with 'bm', {am_only_count} with 'am' only)")
    print(f"Timeout: {timeout}s/position")
    print(f"Score tolerance: +/-{score_tolerance} centipawns")
    print(f"{'='*70}")

    # Results per engine: {engine_name: [(position_data, result), ...]}
    # position_data = (index, position_id, fen, test_type, expected_moves)
    all_results = {}
    all_results_with_positions = {}

    for engine_name in engine_names:
        print(f"\n{'='*70}")
        print(f"Testing: {engine_name}")
        print(f"{'='*70}")

        engine_path, engine_uci_options = get_engine_info(engine_name, engine_dir)

        # Open engine
        cmd = engine_path if isinstance(engine_path, list) else str(engine_path)
        engine = chess.engine.SimpleEngine.popen_uci(cmd, stderr=subprocess.DEVNULL)

        if engine_uci_options:
            engine.configure(engine_uci_options)

        results = []
        results_with_positions = []
        total_solve_time = 0.0

        for idx, position in enumerate(testable_positions):
            pos_num = idx + 1
            is_avoid_only = not position.best_moves and position.avoid_moves
            print(f"\nPosition {pos_num}/{len(testable_positions)}: {position.id}")
            print(f"  FEN: {position.fen[:60]}...")

            # Determine test type and expected moves for storage
            if position.best_moves:
                test_type = 'bm'
                expected_moves = ','.join(position.best_moves)
            else:
                test_type = 'am'
                expected_moves = ','.join(position.avoid_moves)

            # Show expected
            expected_parts = []
            if position.best_moves:
                expected_parts.append(f"bm {' '.join(position.best_moves)}")
            if position.avoid_moves:
                expected_parts.append(f"am {' '.join(position.avoid_moves)}")
            if position.centipawn_eval is not None:
                expected_parts.append(f"ce {position.centipawn_eval:+d}")
            if position.direct_mate is not None:
                expected_parts.append(f"dm {position.direct_mate}")
            print(f"  Expected: {'; '.join(expected_parts)}")

            # Set up board
            board = chess.Board(position.fen)

            # Solve
            print(f"  Searching...", end="", flush=True)
            result = solve_position(engine, board, position, timeout, score_tolerance)
            results.append(result)

            # Store position data with result for database
            position_data = (pos_num, position.id, position.fen, test_type, expected_moves)
            results_with_positions.append((position_data, result))

            # Report result
            if result.solved:
                if is_avoid_only:
                    print(f" played {result.move_found} (avoided: {', '.join(position.avoid_moves)}) at {result.solve_time:.1f}s (depth {result.final_depth})")
                else:
                    print(f" found {result.move_found} at {result.solve_time:.1f}s (depth {result.final_depth})")
                total_solve_time += result.solve_time if result.solve_time else 0

                # Score validation
                if result.score:
                    if result.score.is_mate():
                        mate_moves = result.score.white().mate()
                        score_str = f"M{mate_moves}" if mate_moves else "M?"
                    else:
                        cp = result.score.white().score(mate_score=10000)
                        score_str = f"{cp:+d}" if cp is not None else "?"

                    if position.centipawn_eval is not None:
                        diff = None
                        if not result.score.is_mate():
                            cp = result.score.white().score(mate_score=10000)
                            if cp is not None:
                                diff = cp - position.centipawn_eval
                        valid_mark = "OK" if result.score_valid else "MISMATCH"
                        if diff is not None:
                            print(f"  Engine score: {score_str} (expected {position.centipawn_eval:+d}, diff: {diff:+d}) {valid_mark}")
                        else:
                            print(f"  Engine score: {score_str} (expected {position.centipawn_eval:+d}) {valid_mark}")

                print(f"  Status: SOLVED ({result.solve_time:.1f}s)")
            else:
                if is_avoid_only:
                    print(f" played {result.move_found} (depth {result.final_depth})")
                    print(f"  Should have avoided: {', '.join(position.avoid_moves)}")
                    print(f"  Status: FAILED (played avoid move)")
                elif result.timed_out:
                    print(f" timeout ({timeout}s)")
                    print(f"  Last move considered: {result.move_found or 'none'}")
                    print(f"  Status: FAILED (timeout)")
                else:
                    print(f" did not find expected move")
                    print(f"  Move found: {result.move_found or 'none'}")
                    print(f"  Status: FAILED (wrong move)")

        # Close engine
        engine.quit()

        # Store results
        all_results[engine_name] = results
        all_results_with_positions[engine_name] = results_with_positions

        # Print summary for this engine
        total = len(results)
        solved = sum(1 for r in results if r.solved)
        solved_with_correct_score = sum(1 for r in results if r.solved and r.score_valid is True)
        solved_with_wrong_score = sum(1 for r in results if r.solved and r.score_valid is False)
        solved_no_score_check = sum(1 for r in results if r.solved and r.score_valid is None)
        failed_wrong_move = sum(1 for r in results if not r.solved and not r.timed_out)
        failed_timeout = sum(1 for r in results if r.timed_out)
        positions_with_eval = solved_with_correct_score + solved_with_wrong_score

        avg_solve_time = total_solve_time / solved if solved > 0 else 0

        print(f"\n{'='*70}")
        print(f"RESULTS: {engine_name}")
        print(f"{'='*70}")
        print(f"Solved: {solved}/{total} ({100*solved/total:.1f}%)")
        if solved > 0:
            print(f"Average solve time: {avg_solve_time:.2f}s")
            print(f"Total solve time: {total_solve_time:.1f}s")
        print()
        print("Breakdown:")
        print(f"  - Found correct move with correct eval: {solved_with_correct_score}/{positions_with_eval}")
        print(f"  - Found correct move, wrong eval: {solved_with_wrong_score}/{positions_with_eval}")
        print(f"  - Found correct move, no eval to check: {solved_no_score_check}/{total}")
        print(f"  - Found wrong move: {failed_wrong_move}/{total}")
        print(f"  - Timed out: {failed_timeout}/{total}")
        print(f"{'='*70}")

    # If multiple engines, print comparison
    if len(engine_names) > 1:
        print(f"\n{'='*70}")
        print("ENGINE COMPARISON")
        print(f"{'='*70}")
        print(f"{'Engine':<30} {'Solved':>8} {'%':>8} {'Avg Time':>10}")
        print("-" * 58)

        for engine_name in engine_names:
            results = all_results[engine_name]
            solved = sum(1 for r in results if r.solved)
            total_time = sum(r.solve_time for r in results if r.solved and r.solve_time)
            avg_time = total_time / solved if solved > 0 else 0
            pct = 100 * solved / len(results)
            print(f"{engine_name:<30} {solved:>5}/{len(results):<2} {pct:>7.1f}% {avg_time:>9.2f}s")

        print(f"{'='*70}")

    # Save results to database
    if store_results:
        hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())
        save_epd_test_run(
            epd_file=epd_file.name,
            total_positions=len(testable_positions),
            timeout_seconds=timeout,
            score_tolerance=score_tolerance,
            hostname=hostname,
            engine_results=all_results_with_positions
        )
