"""
Parallel game execution with continuous pipeline.

Provides a shared runner that keeps workers busy by replenishing
games as they complete, with visual progress display.
"""

import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
from dataclasses import dataclass
from typing import Callable, Iterator, Optional

from compete.game import GameConfig, GameResult, play_game_from_config


# Typical chess game length for progress estimation
EXPECTED_MOVES = 80


@dataclass
class GameStatus:
    """Status of a single game for progress display."""
    game_index: int
    start_time: float
    move_count: int = 0
    nps: Optional[int] = None
    result: Optional[str] = None
    finished: bool = False


class ProgressDisplay:
    """
    Live progress display for parallel games.

    Shows a fixed-height display with:
    - Top section: active (in-progress) games
    - Bottom section: recently completed games
    - Summary line at the bottom

    Example with concurrency=4 (8 slots total):
      Game 101: ●●●●●●●●●●●●●●●●●●●●○○○○○○○○○○○○ 1.1M nps
      Game 102: ●●●●●●●●●●●●●●●○○○○○○○○○○○○○○○○○ 1.0M nps
      [empty]
      [empty]
      --------------------------------------------------
      Game  99: ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 1.1M (1-0)
      Game 100: ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 1.0M (0-1)
      [empty]
      [empty]
      [100 done: 45W-40L-15D | 2 active]
    """

    BAR_WIDTH = 32
    FILLED_CHAR = "●"
    EMPTY_CHAR = "○"

    def __init__(self, label: str = "Game", concurrency: int = 4):
        self.label = label
        self.concurrency = concurrency
        self.max_slots = concurrency * 2  # Fixed display height
        self.games: dict[int, GameStatus] = {}
        self.recently_completed: list[GameStatus] = []  # Most recent completions
        self.lock = threading.Lock()
        self.display_thread: Optional[threading.Thread] = None
        self.running = False
        self.lines_printed = 0
        self.completed_count = 0
        self.results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "err": 0}
        self.ansi_supported = self._check_ansi_support()

    def _check_ansi_support(self) -> bool:
        """Check if terminal supports ANSI escape codes."""
        if os.name == 'nt':
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                kernel32.SetConsoleMode(
                    kernel32.GetStdHandle(-11),
                    7
                )
                return True
            except Exception:
                return False
        return True

    def start_game(self, game_index: int):
        """Record that a game has started."""
        with self.lock:
            self.games[game_index] = GameStatus(
                game_index=game_index,
                start_time=time.time()
            )

    def update_moves(self, game_index: int, move_count: int):
        """Update the move count for a game."""
        with self.lock:
            if game_index in self.games:
                self.games[game_index].move_count = move_count

    def finish_game(self, game_index: int, result: str, nps: Optional[int] = None):
        """Record that a game has finished."""
        with self.lock:
            if game_index in self.games:
                game = self.games[game_index]
                game.finished = True
                game.result = result
                if nps:
                    game.nps = nps
                self.completed_count += 1
                if result in self.results:
                    self.results[result] += 1

                # Move to recently completed list (keep most recent)
                self.recently_completed.insert(0, game)
                if len(self.recently_completed) > self.concurrency:
                    self.recently_completed.pop()

                # Remove from active games
                del self.games[game_index]

    def _format_nps(self, nps: Optional[int]) -> str:
        """Format NPS for display."""
        if not nps:
            return ""
        if nps >= 1_000_000:
            return f"{nps/1_000_000:.1f}M"
        elif nps >= 1_000:
            return f"{nps/1_000:.0f}k"
        return str(nps)

    def _format_game_line(self, status: GameStatus) -> str:
        """Format a single game's status line."""
        if status.finished:
            progress = 1.0
            result_str = f" ({status.result})" if status.result else " (done)"
        else:
            progress = min(1.0, status.move_count / EXPECTED_MOVES)
            result_str = ""

        filled = int(progress * self.BAR_WIDTH)
        empty = self.BAR_WIDTH - filled
        bar = self.FILLED_CHAR * filled + self.EMPTY_CHAR * empty

        nps_str = self._format_nps(status.nps)
        if nps_str:
            nps_str = f" {nps_str}"

        game_num = f"{self.label} {status.game_index + 1:4d}"
        return f"  {game_num}: {bar}{nps_str}{result_str}"

    def _format_empty_slot(self) -> str:
        """Format an empty slot line."""
        empty_bar = self.EMPTY_CHAR * self.BAR_WIDTH
        return f"  {'[empty]':<{len(self.label) + 6}}: {empty_bar}"

    def _render(self) -> list[str]:
        """Render fixed-height display with active games on top, completed on bottom."""
        lines = []

        with self.lock:
            # Top section: active (in-progress) games
            active_games = sorted(self.games.values(), key=lambda g: g.game_index)
            active_slots = self.concurrency

            for i in range(active_slots):
                if i < len(active_games):
                    lines.append(self._format_game_line(active_games[i]))
                else:
                    lines.append(self._format_empty_slot())

            # Separator
            lines.append("  " + "-" * 50)

            # Bottom section: recently completed games
            completed_slots = self.concurrency

            for i in range(completed_slots):
                if i < len(self.recently_completed):
                    lines.append(self._format_game_line(self.recently_completed[i]))
                else:
                    lines.append(self._format_empty_slot())

            # Summary line
            active_count = len(self.games)
            w = self.results["1-0"]
            l = self.results["0-1"]
            d = self.results["1/2-1/2"]
            lines.append(f"  [{self.completed_count} done: {w}W-{l}L-{d}D | {active_count} active]")

        return lines

    def _clear_lines(self, count: int):
        """Clear the last N lines."""
        if not self.ansi_supported or count == 0:
            return
        for _ in range(count):
            sys.stdout.write('\033[F')
            sys.stdout.write('\033[K')
        sys.stdout.flush()

    def _display_loop(self):
        """Background thread that updates the display."""
        while self.running:
            self._update_display()
            time.sleep(10)

    def _update_display(self):
        """Update the display (called from background thread)."""
        if not self.ansi_supported:
            return

        lines = self._render()
        if not lines:
            return

        self._clear_lines(self.lines_printed)

        for line in lines:
            print(line)

        self.lines_printed = len(lines)
        sys.stdout.flush()

    def start(self):
        """Start the display update thread."""
        if not self.ansi_supported:
            print(f"  (Running games...)")
            return

        self.running = True
        self.display_thread = threading.Thread(target=self._display_loop, daemon=True)
        self.display_thread.start()

    def stop(self):
        """Stop the display and show final state."""
        self.running = False
        if self.display_thread:
            self.display_thread.join(timeout=1.0)

        if self.ansi_supported:
            self._clear_lines(self.lines_printed)
            self.lines_printed = 0


def run_games_parallel(
    game_configs: list[GameConfig],
    concurrency: int,
    on_game_complete: Callable[[GameConfig, GameResult], None],
    label: str = "Game"
) -> dict:
    """
    Run games in parallel with continuous pipeline.

    Keeps `concurrency` games running at all times by starting new games
    as others complete. Shows visual progress with move-based progress bars.

    Args:
        game_configs: List of game configurations to run
        concurrency: Number of parallel games to maintain
        on_game_complete: Callback called for each completed game with (config, result)
        label: Label for progress display (default "Game")

    Returns:
        dict with summary: {'completed': N, 'errors': N, 'results': {'1-0': N, ...}}
    """
    if not game_configs:
        return {'completed': 0, 'errors': 0, 'results': {"1-0": 0, "0-1": 0, "1/2-1/2": 0}}

    total_games = len(game_configs)
    completed = 0
    errors = 0
    results = {"1-0": 0, "0-1": 0, "1/2-1/2": 0, "*": 0}

    if concurrency <= 1:
        # Sequential execution
        for config in game_configs:
            try:
                result = play_game_from_config(config)
                on_game_complete(config, result)
                completed += 1
                if result.result in results:
                    results[result.result] += 1
            except Exception as e:
                print(f"  Error in game: {e}")
                errors += 1
        return {'completed': completed, 'errors': errors, 'results': results}

    # Parallel execution with continuous pipeline
    progress = ProgressDisplay(label=label, concurrency=concurrency)
    progress.start()

    def make_move_callback(game_idx: int):
        """Create a callback that updates progress for a specific game."""
        return lambda move_count: progress.update_moves(game_idx, move_count)

    config_iter = iter(enumerate(game_configs))

    with ThreadPoolExecutor(max_workers=concurrency) as executor:
        pending_futures = {}  # future -> (config, game_index)

        # Submit initial batch up to concurrency limit
        for _ in range(min(concurrency, total_games)):
            try:
                idx, config = next(config_iter)
                callback = make_move_callback(idx)
                future = executor.submit(play_game_from_config, config, callback)
                pending_futures[future] = (config, idx)
                progress.start_game(idx)
            except StopIteration:
                break

        # Process completions and submit new games
        while pending_futures:
            done, _ = wait(pending_futures.keys(), return_when=FIRST_COMPLETED)

            for future in done:
                config, game_idx = pending_futures.pop(future)
                try:
                    result = future.result()

                    # Get NPS for display
                    nps = None
                    if result.white_nps and result.black_nps:
                        nps = (result.white_nps + result.black_nps) // 2
                    elif result.white_nps:
                        nps = result.white_nps
                    elif result.black_nps:
                        nps = result.black_nps

                    progress.finish_game(game_idx, result.result, nps)
                    on_game_complete(config, result)
                    completed += 1
                    if result.result in results:
                        results[result.result] += 1

                except Exception as e:
                    progress.finish_game(game_idx, "err")
                    print(f"\n  Error in game {game_idx}: {e}")
                    errors += 1

                # Submit next game if available
                try:
                    idx, next_config = next(config_iter)
                    callback = make_move_callback(idx)
                    new_future = executor.submit(play_game_from_config, next_config, callback)
                    pending_futures[new_future] = (next_config, idx)
                    progress.start_game(idx)
                except StopIteration:
                    pass  # No more games to submit

    progress.stop()
    return {'completed': completed, 'errors': errors, 'results': results}
