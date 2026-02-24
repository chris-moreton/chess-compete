"""
Game playing and Elo calculation.
"""

import math
import os
import socket
import subprocess
import traceback
from datetime import datetime
from pathlib import Path
from dataclasses import dataclass
from typing import Optional

import chess
import chess.engine
import chess.pgn


# NPS calibration constants
CALIBRATION_TARGET_NPS = 2_000_000
CALIBRATION_DEPTH = 15
CALIBRATION_MAX_TIMEMULT = 5.0


@dataclass
class GameConfig:
    """Configuration for a single game, used for parallel execution."""
    game_index: int
    white_name: str
    black_name: str
    white_path: str  # String path for pickling
    black_path: str
    white_uci_options: Optional[dict]
    black_uci_options: Optional[dict]
    time_per_move: float
    opening_fen: Optional[str]
    opening_name: Optional[str]
    is_engine1_white: bool  # Track which engine is white for result attribution


@dataclass
class GameResult:
    """Result of a single game, returned from parallel execution."""
    game_index: int
    result: str  # "1-0", "0-1", "1/2-1/2", "*"
    pgn: str
    is_engine1_white: bool
    white_name: str
    black_name: str
    time_per_move: float
    opening_name: Optional[str]
    opening_fen: Optional[str]
    white_nps: Optional[int] = None
    black_nps: Optional[int] = None
    timemult: Optional[float] = None


def calibrate_nps(engine_path: str) -> tuple[int, float]:
    """
    Calibrate NPS by searching startpos to a fixed depth.

    Returns (measured_nps, timemult) where timemult = CALIBRATION_TARGET_NPS / measured_nps,
    capped at CALIBRATION_MAX_TIMEMULT. Falls back to (0, 1.0) on failure.
    """
    try:
        cmd = engine_path if isinstance(engine_path, list) else str(engine_path)
        engine = chess.engine.SimpleEngine.popen_uci(cmd, stderr=subprocess.DEVNULL)
        board = chess.Board()
        info = engine.analyse(board, chess.engine.Limit(depth=CALIBRATION_DEPTH))
        engine.quit()

        nodes = info.get("nodes", 0)
        time_spent = info.get("time", 0.0)
        if time_spent > 0 and nodes > 0:
            nps = int(nodes / time_spent)
            timemult = CALIBRATION_TARGET_NPS / nps
            timemult = min(timemult, CALIBRATION_MAX_TIMEMULT)
            return nps, round(timemult, 3)
        else:
            print("  Calibration: no timing data, using timemult=1.0")
            return 0, 1.0
    except Exception as e:
        print(f"  Calibration failed: {e}, using timemult=1.0")
        return 0, 1.0


def play_game_from_config(config: GameConfig, on_move: callable = None,
                          calibration_engine_path: str = None) -> GameResult:
    """Play a game from a GameConfig - suitable for ThreadPoolExecutor."""
    timemult = None
    effective_config = config
    if calibration_engine_path:
        nps, tm = calibrate_nps(calibration_engine_path)
        timemult = tm
        adjusted_time = config.time_per_move * tm
        # Create a new config with adjusted time
        effective_config = GameConfig(
            game_index=config.game_index,
            white_name=config.white_name,
            black_name=config.black_name,
            white_path=config.white_path,
            black_path=config.black_path,
            white_uci_options=config.white_uci_options,
            black_uci_options=config.black_uci_options,
            time_per_move=adjusted_time,
            opening_fen=config.opening_fen,
            opening_name=config.opening_name,
            is_engine1_white=config.is_engine1_white,
        )

    result, game = play_game(
        engine1_cmd=effective_config.white_path,
        engine2_cmd=effective_config.black_path,
        engine1_name=effective_config.white_name,
        engine2_name=effective_config.black_name,
        time_per_move=effective_config.time_per_move,
        start_fen=effective_config.opening_fen,
        opening_name=effective_config.opening_name,
        engine1_uci_options=effective_config.white_uci_options,
        engine2_uci_options=effective_config.black_uci_options,
        on_move=on_move
    )

    # Extract NPS from game headers
    white_nps = int(game.headers.get("WhiteNPS", 0)) or None
    black_nps = int(game.headers.get("BlackNPS", 0)) or None

    return GameResult(
        game_index=config.game_index,
        result=result,
        pgn=str(game),
        is_engine1_white=config.is_engine1_white,
        white_name=config.white_name,
        black_name=config.black_name,
        time_per_move=effective_config.time_per_move,
        opening_name=config.opening_name,
        opening_fen=config.opening_fen,
        white_nps=white_nps,
        black_nps=black_nps,
        timemult=timemult
    )


def calculate_elo_difference(wins: int, losses: int, draws: int) -> tuple[float, float]:
    """
    Calculate Elo difference and error margin from match results.
    Returns (elo_diff, error_margin) where positive means engine1 is stronger.
    """
    total = wins + losses + draws
    if total == 0:
        return 0.0, 0.0

    # Score from engine1's perspective
    score = (wins + draws * 0.5) / total

    # Avoid division by zero at extremes
    if score <= 0.001:
        return -800.0, 100.0
    if score >= 0.999:
        return 800.0, 100.0

    # Elo difference formula
    elo_diff = -400 * math.log10(1 / score - 1)

    # Error margin (approximate 95% confidence interval)
    # Based on standard error of proportion
    std_error = math.sqrt(score * (1 - score) / total)

    # Convert to Elo error (derivative of Elo formula)
    if 0.01 < score < 0.99:
        elo_error = 400 * std_error / (score * (1 - score) * math.log(10))
        elo_error = min(elo_error, 200)  # Cap at reasonable value
    else:
        elo_error = 100

    return elo_diff, elo_error


def play_game(engine1_cmd: Path | list, engine2_cmd: Path | list,
              engine1_name: str, engine2_name: str,
              time_per_move: float,
              start_fen: str = None,
              opening_name: str = None,
              engine1_uci_options: dict = None,
              engine2_uci_options: dict = None,
              on_move: callable = None) -> tuple[str, chess.pgn.Game]:
    """Play a single game and return (result, pgn_game).

    engine1_cmd/engine2_cmd can be:
    - Path: for native executables
    - list: for Java engines ["java", "-jar", "path/to/engine.jar"]
    """
    if start_fen:
        board = chess.Board(start_fen)
    else:
        board = chess.Board()

    # popen_uci accepts either a string or a list of arguments
    cmd1 = engine1_cmd if isinstance(engine1_cmd, list) else str(engine1_cmd)
    cmd2 = engine2_cmd if isinstance(engine2_cmd, list) else str(engine2_cmd)
    engine1 = chess.engine.SimpleEngine.popen_uci(cmd1, stderr=subprocess.DEVNULL)
    engine2 = chess.engine.SimpleEngine.popen_uci(cmd2, stderr=subprocess.DEVNULL)

    # Configure UCI options if provided
    if engine1_uci_options:
        engine1.configure(engine1_uci_options)
    if engine2_uci_options:
        engine2.configure(engine2_uci_options)

    engines = [engine1, engine2]

    game = chess.pgn.Game()
    game.headers["Event"] = "Engine Match"
    game.headers["Site"] = os.environ.get("COMPUTER_NAME", socket.gethostname())
    game.headers["Date"] = datetime.now().strftime("%Y.%m.%d")
    game.headers["White"] = engine1_name
    game.headers["Black"] = engine2_name
    game.headers["TimeControl"] = f"{time_per_move:.2f}s/move"
    if start_fen:
        game.headers["FEN"] = start_fen
        game.headers["SetUp"] = "1"
    if opening_name:
        game.headers["Opening"] = opening_name

    # Set up the game from the FEN position
    if start_fen:
        game.setup(board)

    node = game

    # Track nodes and time for NPS calculation
    engine1_nodes = 0
    engine1_time = 0.0
    engine2_nodes = 0
    engine2_time = 0.0

    move_count = 0
    try:
        while not board.is_game_over():
            is_white = board.turn == chess.WHITE
            engine = engines[0] if is_white else engines[1]
            result = engine.play(board, chess.engine.Limit(time=time_per_move), info=chess.engine.INFO_ALL)
            board.push(result.move)
            node = node.add_variation(result.move)
            move_count += 1

            # Report move progress via callback
            if on_move:
                on_move(move_count)

            # Accumulate nodes and time from search info
            if result.info:
                nodes = result.info.get("nodes", 0)
                time_spent = result.info.get("time", 0.0)
                if is_white:
                    engine1_nodes += nodes
                    engine1_time += time_spent
                else:
                    engine2_nodes += nodes
                    engine2_time += time_spent

    except Exception as e:
        print(f"  Error during game: {e}")
        traceback.print_exc()
    finally:
        engine1.quit()
        engine2.quit()

    game.headers["Result"] = board.result()

    # Add NPS to headers
    if engine1_time > 0:
        engine1_nps = int(engine1_nodes / engine1_time)
        game.headers["WhiteNPS"] = str(engine1_nps)
    if engine2_time > 0:
        engine2_nps = int(engine2_nodes / engine2_time)
        game.headers["BlackNPS"] = str(engine2_nps)

    return board.result(), game
