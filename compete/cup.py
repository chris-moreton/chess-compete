"""
Knockout cup competition mode with seeded brackets and byes.
"""

import math
import os
import random
import socket
import sys
from datetime import datetime
from pathlib import Path

from compete.constants import PROVISIONAL_GAMES
from compete.database import load_elo_ratings, save_game_to_db, get_initial_elo
from compete.engine_manager import get_engine_info, ensure_engines_initialized
from compete.game import play_game
from compete.openings import OPENING_BOOK


def get_seeded_engines(engine_dir: Path, num_engines: int = None,
                       engine_type: str = None, include_inactive: bool = False) -> list[tuple[str, float]]:
    """
    Get engines sorted by Ordo rating for seeding.

    Args:
        engine_dir: Path to engines directory
        num_engines: Limit to top N engines (None = all)
        engine_type: Filter by engine type ('rusty' or 'stockfish', None = all)
        include_inactive: If True, include inactive engines

    Returns list of (engine_name, ordo_rating) tuples, best first.
    Uses Ordo rating if available, falls back to standard Elo.
    """
    from web.queries import get_engines_ranked_by_elo
    from web.app import create_app

    app = create_app()
    with app.app_context():
        # Get engines with all rating types
        active_only = not include_inactive
        ranked = get_engines_ranked_by_elo(active_only=active_only, engine_type=engine_type)

        # Extract name and Ordo (or Elo as fallback)
        seeded = []
        for r in ranked:
            engine = r['engine']
            # Use Ordo if available, otherwise Elo
            rating = r.get('ordo') or r.get('elo', 1500)
            seeded.append((engine.name, float(rating)))

        # Sort by rating descending (best first)
        seeded.sort(key=lambda x: -x[1])

        # Limit to num_engines if specified
        if num_engines:
            seeded = seeded[:num_engines]

        return seeded


def next_power_of_2(n: int) -> int:
    """Return the next power of 2 >= n."""
    return 1 << (n - 1).bit_length()


def generate_bracket(num_participants: int) -> list[tuple[int, int | None]]:
    """
    Generate seeded bracket matchups for a knockout tournament.

    Returns list of (seed1, seed2) tuples where:
    - seed1 is 1-indexed (1 = best seed)
    - seed2 is None for byes

    Standard seeding ensures 1 and 2 can only meet in the final:
    - Top half: 1v16, 8v9, 4v13, 5v12
    - Bottom half: 2v15, 7v10, 3v14, 6v11
    """
    bracket_size = next_power_of_2(num_participants)
    num_byes = bracket_size - num_participants

    # Generate standard seeded matchups for bracket size
    # This algorithm recursively builds the bracket so top seeds are spread out
    def get_matchups(size: int) -> list[tuple[int, int]]:
        if size == 2:
            return [(1, 2)]

        half = size // 2
        lower_matchups = get_matchups(half)

        matchups = []
        for m1, m2 in lower_matchups:
            # In a round of 'size', seed m1 plays seed (size+1 - m1)
            # and seed m2 plays seed (size+1 - m2)
            matchups.append((m1, size + 1 - m1))
            matchups.append((m2, size + 1 - m2))

        return matchups

    matchups = get_matchups(bracket_size)

    # Convert to handle byes (any seed > num_participants becomes a bye)
    result = []
    for s1, s2 in matchups:
        if s2 > num_participants:
            result.append((s1, None))  # Bye for s1
        else:
            result.append((s1, s2))

    return result


def get_round_name(remaining_participants: int) -> str:
    """Get the name for a round based on remaining participants."""
    names = {
        2: "Final",
        4: "Semifinals",
        8: "Quarterfinals",
        16: "Round of 16",
        32: "Round of 32",
        64: "Round of 64",
    }
    return names.get(remaining_participants, f"Round of {remaining_participants}")


def play_cup_match(engine1_name: str, engine2_name: str, engine_dir: Path,
                   games_per_match: int, time_per_move: float,
                   time_low: float = None, time_high: float = None,
                   cup_id: int = None) -> tuple[str, float, float, int, bool, bool]:
    """
    Play a cup match between two engines.

    Args:
        games_per_match: Number of game PAIRS (so total games = games_per_match * 2)

    Returns: (winner_name, engine1_points, engine2_points, games_played, is_tiebreaker, decided_by_coin_flip)

    Rules:
    1. Play games_per_match pairs (2 games each, alternating colors)
    2. Each game gets a random opening
    3. If tied: play tiebreaker pairs until someone wins (max 10 pairs = 20 games)
    4. If still tied after tiebreaker: coin flip decides
    """
    engine1_path, engine1_uci = get_engine_info(engine1_name, engine_dir)
    engine2_path, engine2_uci = get_engine_info(engine2_name, engine_dir)

    use_time_range = time_low is not None and time_high is not None

    engine1_points = 0.0
    engine2_points = 0.0
    games_played = 0
    is_tiebreaker = False
    decided_by_coin_flip = False

    def play_one_game(white_name: str, black_name: str, white_path, black_path,
                      white_uci, black_uci, game_time: float):
        """Play a single game with random opening and return the result."""
        nonlocal games_played
        games_played += 1

        # Random opening for each game
        opening_fen, opening_name = random.choice(OPENING_BOOK)

        result, game = play_game(white_path, black_path, white_name, black_name,
                                 game_time, opening_fen, opening_name,
                                 white_uci, black_uci)

        # Save to database
        hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())
        save_game_to_db(white_name, black_name, result, f"{game_time:.2f}s/move",
                        opening_name, opening_fen, str(game),
                        time_per_move_ms=int(game_time * 1000),
                        hostname=hostname)

        return result, opening_name

    # Play regular game pairs
    for pair_idx in range(games_per_match):
        # Select time for this pair
        if use_time_range:
            pair_time = random.uniform(time_low, time_high)
        else:
            pair_time = time_per_move

        # Game 1: engine1 as white
        result, opening = play_one_game(engine1_name, engine2_name,
                                        engine1_path, engine2_path,
                                        engine1_uci, engine2_uci, pair_time)

        if result == "1-0":
            engine1_points += 1
        elif result == "0-1":
            engine2_points += 1
        elif result == "1/2-1/2":
            engine1_points += 0.5
            engine2_points += 0.5

        print(f"  Game {games_played}: {engine1_name} vs {engine2_name} -> {result} [{opening}] "
              f"Score: {engine1_points:.1f} - {engine2_points:.1f}")

        # Game 2: engine2 as white (different random opening)
        result, opening = play_one_game(engine2_name, engine1_name,
                                        engine2_path, engine1_path,
                                        engine2_uci, engine1_uci, pair_time)

        if result == "1-0":
            engine2_points += 1
        elif result == "0-1":
            engine1_points += 1
        elif result == "1/2-1/2":
            engine1_points += 0.5
            engine2_points += 0.5

        print(f"  Game {games_played}: {engine2_name} vs {engine1_name} -> {result} [{opening}] "
              f"Score: {engine1_points:.1f} - {engine2_points:.1f}")

    # Tiebreaker if needed (play pairs until someone pulls ahead)
    max_tiebreaker_pairs = 10  # 20 games max
    tiebreaker_pairs = 0

    while engine1_points == engine2_points and tiebreaker_pairs < max_tiebreaker_pairs:
        is_tiebreaker = True
        tiebreaker_pairs += 1

        game_time = random.uniform(time_low, time_high) if use_time_range else time_per_move

        # Game 1 of tiebreaker pair: engine1 as white
        result, opening = play_one_game(engine1_name, engine2_name,
                                        engine1_path, engine2_path,
                                        engine1_uci, engine2_uci, game_time)
        if result == "1-0":
            engine1_points += 1
        elif result == "0-1":
            engine2_points += 1
        else:
            engine1_points += 0.5
            engine2_points += 0.5

        print(f"  Tiebreaker {games_played}: {engine1_name} vs {engine2_name} -> {result} [{opening}] "
              f"Score: {engine1_points:.1f} - {engine2_points:.1f}")

        # Check if decided after game 1
        if engine1_points != engine2_points:
            break

        # Game 2 of tiebreaker pair: engine2 as white
        result, opening = play_one_game(engine2_name, engine1_name,
                                        engine2_path, engine1_path,
                                        engine2_uci, engine1_uci, game_time)
        if result == "1-0":
            engine2_points += 1
        elif result == "0-1":
            engine1_points += 1
        else:
            engine1_points += 0.5
            engine2_points += 0.5

        print(f"  Tiebreaker {games_played}: {engine2_name} vs {engine1_name} -> {result} [{opening}] "
              f"Score: {engine1_points:.1f} - {engine2_points:.1f}")

    # Coin flip if still tied
    if engine1_points == engine2_points:
        decided_by_coin_flip = True
        winner_name = random.choice([engine1_name, engine2_name])
        print(f"  Match tied after {max_tiebreaker_pairs * 2} tiebreaker games!")
        print(f"  COIN FLIP: {winner_name} wins!")
    else:
        winner_name = engine1_name if engine1_points > engine2_points else engine2_name

    return winner_name, engine1_points, engine2_points, games_played, is_tiebreaker, decided_by_coin_flip


def run_cup(engine_dir: Path, num_engines: int = None, games_per_match: int = 10,
            time_per_move: float = 1.0, cup_name: str = None,
            time_low: float = None, time_high: float = None,
            engine_type: str = None, include_inactive: bool = False):
    """
    Run a complete knockout cup competition.

    Args:
        engine_dir: Path to engines directory
        num_engines: Limit to top N engines (None = all active)
        games_per_match: Number of game PAIRS per match (total games = pairs * 2)
        time_per_move: Fixed time per move (ignored if time_low/time_high set)
        cup_name: Optional custom cup name
        time_low: Minimum time per move for random range
        time_high: Maximum time per move for random range
        engine_type: Filter by engine type ('rusty' or 'stockfish', None = all)
        include_inactive: If True, include inactive engines in the cup
    """
    from web.database import db
    from web.models import Cup, CupRound, CupMatch, Engine
    from web.app import create_app

    use_time_range = time_low is not None and time_high is not None

    # Get seeded engines
    filter_desc = []
    if engine_type:
        filter_desc.append(f"type={engine_type}")
    if include_inactive:
        filter_desc.append("including inactive")
    filter_str = f" ({', '.join(filter_desc)})" if filter_desc else ""
    print(f"Getting seeded engines by Ordo rating{filter_str}...")
    seeded_engines = get_seeded_engines(engine_dir, num_engines, engine_type, include_inactive)

    if len(seeded_engines) < 2:
        print("Error: Need at least 2 active engines for a cup")
        sys.exit(1)

    num_participants = len(seeded_engines)

    # Ensure all engines are initialized
    engine_names = [e[0] for e in seeded_engines]
    if not ensure_engines_initialized(engine_names, engine_dir):
        print("Error: Failed to initialize some engines")
        sys.exit(1)

    # Generate bracket
    bracket = generate_bracket(num_participants)
    bracket_size = next_power_of_2(num_participants)
    num_rounds = int(math.log2(bracket_size))

    # Create cup name if not provided
    if not cup_name:
        cup_name = f"Cup {datetime.now().strftime('%Y-%m-%d %H:%M')}"

    hostname = os.environ.get("COMPUTER_NAME", socket.gethostname())

    print(f"\n{'='*70}")
    print(f"KNOCKOUT CUP: {cup_name}")
    print(f"{'='*70}")
    print(f"Participants: {num_participants}")
    print(f"Bracket size: {bracket_size}")
    print(f"Rounds: {num_rounds}")
    print(f"Games per match: {games_per_match}")
    if use_time_range:
        print(f"Time: {time_low}-{time_high}s/move (random)")
    else:
        print(f"Time: {time_per_move}s/move")
    print(f"{'='*70}")

    # Print seedings
    print("\nSeedings (by Ordo rating):")
    for i, (name, rating) in enumerate(seeded_engines, 1):
        print(f"  {i:2d}. {name} ({rating:.0f})")

    # Create cup in database
    app = create_app()
    with app.app_context():
        # Create cup record
        cup = Cup(
            name=cup_name,
            status='in_progress',
            num_participants=num_participants,
            games_per_match=games_per_match,
            time_per_move_ms=int(time_per_move * 1000) if not use_time_range else None,
            time_low_ms=int(time_low * 1000) if time_low else None,
            time_high_ms=int(time_high * 1000) if time_high else None,
            hostname=hostname
        )
        db.session.add(cup)
        db.session.flush()
        cup_id = cup.id

        # Create engine ID lookup
        engine_ids = {}
        for name, _ in seeded_engines:
            engine = Engine.query.filter_by(name=name).first()
            if engine:
                engine_ids[name] = engine.id

        # Create first round with initial matchups
        first_round = CupRound(
            cup_id=cup_id,
            round_number=1,
            round_name=get_round_name(bracket_size),
            status='pending'
        )
        db.session.add(first_round)
        db.session.flush()

        # Create matches for first round
        for match_order, (seed1, seed2) in enumerate(bracket, 1):
            engine1_name = seeded_engines[seed1 - 1][0]
            engine1_id = engine_ids[engine1_name]

            if seed2 is None:
                # Bye
                match = CupMatch(
                    round_id=first_round.id,
                    match_order=match_order,
                    engine1_id=engine1_id,
                    engine2_id=None,
                    engine1_seed=seed1,
                    engine2_seed=None,
                    winner_engine_id=engine1_id,
                    status='bye'
                )
            else:
                engine2_name = seeded_engines[seed2 - 1][0]
                engine2_id = engine_ids[engine2_name]
                match = CupMatch(
                    round_id=first_round.id,
                    match_order=match_order,
                    engine1_id=engine1_id,
                    engine2_id=engine2_id,
                    engine1_seed=seed1,
                    engine2_seed=seed2,
                    status='pending'
                )
            db.session.add(match)

        db.session.commit()

    # Process rounds
    current_round = 1

    while current_round <= num_rounds:
        app = create_app()

        # Get round info
        with app.app_context():
            cup_round = CupRound.query.filter_by(cup_id=cup_id, round_number=current_round).first()
            round_name = cup_round.round_name
            round_id = cup_round.id

            # Get match data for this round
            matches_data = []
            for match in CupMatch.query.filter_by(round_id=round_id).order_by(CupMatch.match_order).all():
                engine1 = Engine.query.get(match.engine1_id)
                engine2 = Engine.query.get(match.engine2_id) if match.engine2_id else None
                matches_data.append({
                    'id': match.id,
                    'match_order': match.match_order,
                    'status': match.status,
                    'engine1_name': engine1.name,
                    'engine2_name': engine2.name if engine2 else None,
                    'engine1_id': match.engine1_id,
                    'engine2_id': match.engine2_id,
                    'engine1_seed': match.engine1_seed,
                    'engine2_seed': match.engine2_seed,
                    'winner_engine_id': match.winner_engine_id,
                })

            # Mark round as in progress
            cup_round.status = 'in_progress'
            db.session.commit()

        print(f"\n{'='*70}")
        print(f"{round_name.upper()}")
        print(f"{'='*70}")

        # Process each match
        winners = []
        for match_data in matches_data:
            engine1_name = match_data['engine1_name']

            if match_data['status'] == 'bye':
                print(f"\nMatch {match_data['match_order']}: {engine1_name} (#{match_data['engine1_seed']}) - BYE")
                winners.append((match_data['winner_engine_id'], match_data['engine1_seed']))
                continue

            engine2_name = match_data['engine2_name']

            print(f"\nMatch {match_data['match_order']}: {engine1_name} (#{match_data['engine1_seed']}) vs {engine2_name} (#{match_data['engine2_seed']})")
            print("-" * 50)

            # Mark match as in progress
            with app.app_context():
                match = CupMatch.query.get(match_data['id'])
                match.status = 'in_progress'
                db.session.commit()

            # Play the match (outside app context to avoid long transaction)
            winner_name, e1_points, e2_points, games, is_tb, coin_flip = play_cup_match(
                engine1_name, engine2_name, engine_dir,
                games_per_match, time_per_move,
                time_low, time_high, cup_id
            )

            winner_seed = match_data['engine1_seed'] if winner_name == engine1_name else match_data['engine2_seed']

            print(f"\n  WINNER: {winner_name} ({e1_points:.1f} - {e2_points:.1f})")
            if coin_flip:
                print(f"  (Decided by coin flip)")
            elif is_tb:
                print(f"  (Decided in tiebreaker)")

            # Update match in database
            with app.app_context():
                match = CupMatch.query.get(match_data['id'])
                winner_engine = Engine.query.filter_by(name=winner_name).first()
                match.engine1_points = e1_points
                match.engine2_points = e2_points
                match.games_played = games
                match.winner_engine_id = winner_engine.id
                match.status = 'completed'
                match.is_tiebreaker = is_tb
                match.decided_by_coin_flip = coin_flip
                db.session.commit()

                winners.append((winner_engine.id, winner_seed))

        # Mark round as completed and create next round if needed
        with app.app_context():
            cup_round = CupRound.query.filter_by(cup_id=cup_id, round_number=current_round).first()
            cup_round.status = 'completed'

            if len(winners) > 1:
                # Create next round
                next_round_num = current_round + 1
                next_round = CupRound(
                    cup_id=cup_id,
                    round_number=next_round_num,
                    round_name=get_round_name(len(winners)),
                    status='pending'
                )
                db.session.add(next_round)
                db.session.flush()

                # Create matches for next round (pair up winners)
                for i in range(0, len(winners), 2):
                    engine1_id, seed1 = winners[i]
                    engine2_id, seed2 = winners[i + 1]

                    next_match = CupMatch(
                        round_id=next_round.id,
                        match_order=(i // 2) + 1,
                        engine1_id=engine1_id,
                        engine2_id=engine2_id,
                        engine1_seed=seed1,
                        engine2_seed=seed2,
                        status='pending'
                    )
                    db.session.add(next_match)
            else:
                # Cup complete!
                cup = Cup.query.get(cup_id)
                cup.status = 'completed'
                cup.winner_engine_id = winners[0][0]
                cup.completed_at = datetime.utcnow()

                winner = Engine.query.get(winners[0][0])
                print(f"\n{'='*70}")
                print(f"CUP WINNER: {winner.name}!")
                print(f"{'='*70}\n")

            db.session.commit()

        current_round += 1

    return cup_id
