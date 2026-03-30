#!/usr/bin/env python3
"""
Generate NNUE training data via engine self-play.

Plays games between the engine and itself, recording each position with:
- FEN (board position)
- Search score (centipawns, white-relative)
- Game result (1.0 = white win, 0.5 = draw, 0.0 = black win)

Output format: <FEN> | <score> | <result>
This is the text format accepted by bullet's data conversion tools.

Usage:
    python3 scripts/generate-nnue-data.py \
        --engine /path/to/rusty-rival \
        --depth 8 \
        --games 10000 \
        --output data/training.txt \
        --concurrency 16

    # On AWS (launched via scripts/launch-datagen.sh):
    python3 scripts/generate-nnue-data.py \
        --engine ~/engine \
        --depth 8 \
        --games 50000 \
        --output /tmp/training_data.txt \
        --concurrency 48 \
        --upload s3://chess-compete-builds/nnue-data/
"""

import argparse
import os
import re
import subprocess
import sys
import threading
import time
from collections import deque
from concurrent.futures import ThreadPoolExecutor, as_completed


class UCIEngine:
    """Manages a UCI chess engine process."""

    def __init__(self, path, hash_mb=16, eval_noise=15):
        self.process = subprocess.Popen(
            [path],
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.DEVNULL,
            text=True,
            bufsize=1,
        )
        self._send("uci")
        self._wait_for("uciok")
        self._send(f"setoption name Hash value {hash_mb}")
        self._send(f"setoption name Threads value 1")
        if eval_noise > 0:
            self._send(f"setoption name EvalNoise value {eval_noise}")
        self._send("isready")
        self._wait_for("readyok")

    def _send(self, cmd):
        self.process.stdin.write(cmd + "\n")
        self.process.stdin.flush()

    def _wait_for(self, token):
        while True:
            line = self.process.stdout.readline().strip()
            if token in line:
                return line

    def new_game(self):
        self._send("ucinewgame")
        self._send("isready")
        self._wait_for("readyok")

    def search(self, fen, depth):
        """Search a position to given depth. Returns (best_move, score_cp, is_mate)."""
        self._send(f"position fen {fen}")
        self._send(f"go depth {depth}")

        best_move = None
        score_cp = 0
        is_mate = False

        while True:
            line = self.process.stdout.readline().strip()
            if line.startswith("info") and "score" in line:
                # Parse score from info line
                if " score cp " in line:
                    m = re.search(r"score cp (-?\d+)", line)
                    if m:
                        score_cp = int(m.group(1))
                        is_mate = False
                elif " score mate " in line:
                    m = re.search(r"score mate (-?\d+)", line)
                    if m:
                        mate_in = int(m.group(1))
                        score_cp = 30000 - abs(mate_in) if mate_in > 0 else -(30000 - abs(mate_in))
                        is_mate = True
            elif line.startswith("bestmove"):
                best_move = line.split()[1]
                break

        return best_move, score_cp, is_mate

    def quit(self):
        try:
            self._send("quit")
            self.process.wait(timeout=5)
        except Exception:
            self.process.kill()


def make_move_fen(engine, fen, move):
    """Apply a move and get the resulting FEN by asking the engine."""
    engine._send(f"position fen {fen} moves {move}")
    engine._send("d")
    # We can't use 'd' on all engines. Instead, do a depth-1 search from the new position
    # and parse the FEN from the position setup.
    # Actually, simpler: use python-chess for this.
    return None  # Will use python-chess instead


def load_openings(book_path):
    """Load opening positions from a PGN book file. Returns list of FENs."""
    import chess.pgn
    import io

    fens = []
    with open(book_path) as f:
        while True:
            game = chess.pgn.read_game(f)
            if game is None:
                break
            board = game.board()
            for move in game.mainline_moves():
                board.push(move)
            fens.append(board.fen())
    return fens


def play_game(engine_path, depth, hash_mb=16, eval_noise=15, opening_fen=None):
    """Play one self-play game and return list of (fen, score_cp, result) tuples.

    Returns list of positions from the game, with result filled in after game ends.
    Scores are white-relative. Depth is randomized ±2 per move for variety.
    """
    import chess
    import random

    engine = UCIEngine(engine_path, hash_mb=hash_mb, eval_noise=eval_noise)
    engine.new_game()

    board = chess.Board(opening_fen) if opening_fen else chess.Board()
    positions = []  # (fen, white_relative_score)
    move_count = 0
    max_moves = 300  # Prevent infinite games
    min_depth = max(4, depth - 2)
    max_depth = depth + 2

    while not board.is_game_over(claim_draw=True) and move_count < max_moves:
        fen = board.fen()
        move_depth = random.randint(min_depth, max_depth)
        best_move, score_cp, is_mate = engine.search(fen, move_depth)

        if best_move is None or best_move == "(none)":
            break

        # Convert score to white-relative
        white_score = score_cp if board.turn == chess.WHITE else -score_cp

        # Skip positions in the first 8 plies (opening book territory)
        # and positions where the score is a forced mate (not useful for eval training)
        if move_count >= 8 and not is_mate:
            positions.append((fen, white_score))

        # Apply move
        try:
            board.push_uci(best_move)
        except ValueError:
            break

        move_count += 1

        # Adjudicate: resign if score is too extreme
        if abs(score_cp) > 3000 and move_count > 20:
            break

    engine.quit()

    # Determine game result
    if board.is_game_over(claim_draw=True):
        result_obj = board.result(claim_draw=True)
        if result_obj == "1-0":
            result = 1.0
        elif result_obj == "0-1":
            result = 0.0
        else:
            result = 0.5
    elif move_count >= max_moves:
        result = 0.5  # Draw by length
    else:
        # Adjudicated by score
        # Use the last search score to determine winner
        if abs(score_cp) > 3000:
            # The side to move has the score
            if board.turn == chess.WHITE:
                result = 1.0 if score_cp > 0 else 0.0
            else:
                result = 0.0 if score_cp > 0 else 1.0
        else:
            result = 0.5

    # Return positions with result
    return [(fen, score, result) for fen, score in positions]


def worker(game_id, engine_path, depth, hash_mb, eval_noise, opening_fen):
    """Worker function for parallel game generation."""
    try:
        positions = play_game(engine_path, depth, hash_mb, eval_noise, opening_fen)
        return game_id, positions
    except Exception as e:
        print(f"Game {game_id} failed: {e}", file=sys.stderr)
        return game_id, []


def main():
    parser = argparse.ArgumentParser(description="Generate NNUE training data via self-play")
    parser.add_argument("--engine", required=True, help="Path to engine binary")
    parser.add_argument("--depth", type=int, default=8, help="Search depth per move (default: 8)")
    parser.add_argument("--games", type=int, default=10000, help="Number of games to play")
    parser.add_argument("--output", default="data/training.txt", help="Output file path")
    parser.add_argument("--concurrency", type=int, default=1, help="Parallel games")
    parser.add_argument("--hash", type=int, default=16, help="Hash per engine instance in MB (default: 16)")
    parser.add_argument("--eval-noise", type=int, default=15, help="Eval noise in cp (default: 15, 0=disabled)")
    parser.add_argument("--book", default="", help="Path to opening book PGN")
    parser.add_argument("--upload", default="", help="S3 path to upload results (e.g. s3://bucket/path/)")
    args = parser.parse_args()

    # Create output directory
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)

    import random

    # Load opening book
    openings = []
    if args.book and os.path.isfile(args.book):
        openings = load_openings(args.book)

    print(f"NNUE Data Generation")
    print(f"  Engine:      {args.engine}")
    print(f"  Depth:       {args.depth} (±2 randomized per move)")
    print(f"  Games:       {args.games}")
    print(f"  Concurrency: {args.concurrency}")
    print(f"  Hash/engine: {args.hash}MB")
    print(f"  Eval noise:  ±{args.eval_noise}cp")
    print(f"  Openings:    {len(openings)} positions" if openings else "  Openings:    none (starting position)")
    print(f"  Output:      {args.output}")
    print()

    total_positions = 0
    games_completed = 0
    start_time = time.time()

    with open(args.output, "w") as f:
        with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
            futures = {}
            next_game = 0

            # Submit initial batch
            for _ in range(min(args.concurrency, args.games)):
                opening = random.choice(openings) if openings else None
                future = executor.submit(worker, next_game, args.engine, args.depth, args.hash,
                                         args.eval_noise, opening)
                futures[future] = next_game
                next_game += 1

            while futures:
                for future in as_completed(futures):
                    game_id, positions = future.result()
                    del futures[future]

                    # Write positions
                    for fen, score, result in positions:
                        f.write(f"{fen} | {score} | {result}\n")
                    total_positions += len(positions)
                    games_completed += 1

                    # Progress
                    elapsed = time.time() - start_time
                    rate = games_completed / elapsed if elapsed > 0 else 0
                    pos_rate = total_positions / elapsed if elapsed > 0 else 0
                    if games_completed % 100 == 0 or games_completed == args.games:
                        eta = (args.games - games_completed) / rate if rate > 0 else 0
                        eta_str = f"{eta/3600:.1f}h" if eta > 3600 else f"{eta/60:.0f}m"
                        print(f"  Games: {games_completed}/{args.games} | "
                              f"Positions: {total_positions:,} | "
                              f"Rate: {rate:.1f} games/s, {pos_rate:.0f} pos/s | "
                              f"ETA: {eta_str} | "
                              f"Elapsed: {elapsed:.0f}s")

                    # Periodic S3 checkpoint every 10,000 games
                    if args.upload and games_completed % 10000 == 0 and games_completed > 0:
                        f.flush()
                        s3_path = args.upload.rstrip("/") + "/" + os.path.basename(args.output)
                        print(f"  [Checkpoint] Uploading {total_positions:,} positions to S3...")
                        subprocess.run(
                            ["aws", "s3", "cp", args.output, s3_path],
                            capture_output=True, text=True,
                        )

                    # Submit next game if more remain
                    if next_game < args.games:
                        opening = random.choice(openings) if openings else None
                        future = executor.submit(worker, next_game, args.engine, args.depth, args.hash,
                                                 args.eval_noise, opening)
                        futures[future] = next_game
                        next_game += 1

                    break  # Process one at a time to maintain ordering

    elapsed = time.time() - start_time
    print()
    print(f"Complete: {games_completed} games, {total_positions:,} positions in {elapsed:.0f}s")
    print(f"Output: {args.output} ({os.path.getsize(args.output) / 1024 / 1024:.1f} MB)")

    # Upload to S3 if requested
    if args.upload:
        s3_path = args.upload.rstrip("/") + "/" + os.path.basename(args.output)
        print(f"Uploading to {s3_path}...")
        result = subprocess.run(
            ["aws", "s3", "cp", args.output, s3_path],
            capture_output=True, text=True,
        )
        if result.returncode == 0:
            print("Upload complete.")
        else:
            print(f"Upload failed: {result.stderr}", file=sys.stderr)


if __name__ == "__main__":
    main()
