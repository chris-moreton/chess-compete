"""
Live progress display for SPSA games.

Shows individual game progress with visual bars and NPS.
"""

import os
import sys
import threading
import time
from dataclasses import dataclass, field
from typing import Optional


# Typical chess game length for progress estimation
EXPECTED_MOVES = 80  # Full moves (plies / 2)


@dataclass
class GameStatus:
    """Status of a single game."""
    game_index: int
    start_time: float
    game_type: str = "spsa"  # "spsa" or "ref"
    move_count: int = 0  # Current move count (plies)
    nps: Optional[int] = None
    result: Optional[str] = None  # "1-0", "0-1", "1/2-1/2"
    finished: bool = False


class ProgressDisplay:
    """
    Live progress display for parallel games.

    Shows a visual representation of each active game's progress:
      Game  1: ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 1.2M nps
      Game  2: ●●●●●●●●●●●●●●●●●●●●○○○○○○○○○○○○ 1.1M nps
      Game  3: ●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●●● 980k nps (1-0)
      Game  4: ●●●●●●●●●●●●●●●○○○○○○○○○○○○○○○○○ 1.0M nps
      [Completed: 15/100 | Active: 4]
    """

    BAR_WIDTH = 32
    FILLED_CHAR = "●"
    EMPTY_CHAR = "○"

    def __init__(self, label: str = "Game"):
        self.label = label
        self.games: dict[int, GameStatus] = {}
        self.lock = threading.Lock()
        self.display_thread: Optional[threading.Thread] = None
        self.running = False
        self.lines_printed = 0
        self.completed_count = 0
        # Track results separately for SPSA and reference games
        # SPSA: plus_win, minus_win, draw (from plus engine's perspective)
        # Ref: win, loss, draw (from base engine's perspective)
        self.spsa_results = {"plus_win": 0, "minus_win": 0, "draw": 0, "err": 0}
        self.ref_results = {"win": 0, "loss": 0, "draw": 0, "err": 0}

        # Check if terminal supports ANSI (Windows 10+ does, older doesn't)
        self.ansi_supported = self._check_ansi_support()

    def _check_ansi_support(self) -> bool:
        """Check if terminal supports ANSI escape codes."""
        if os.name == 'nt':
            # Enable ANSI on Windows 10+
            try:
                import ctypes
                kernel32 = ctypes.windll.kernel32
                # Enable virtual terminal processing
                kernel32.SetConsoleMode(
                    kernel32.GetStdHandle(-11),  # STD_OUTPUT_HANDLE
                    7  # ENABLE_PROCESSED_OUTPUT | ENABLE_WRAP_AT_EOL_OUTPUT | ENABLE_VIRTUAL_TERMINAL_PROCESSING
                )
                return True
            except Exception:
                return False
        return True  # Unix-like systems support ANSI

    def start_game(self, game_index: int, game_type: str = "spsa"):
        """Record that a game has started."""
        with self.lock:
            self.games[game_index] = GameStatus(
                game_index=game_index,
                start_time=time.time(),
                game_type=game_type
            )

    def update_moves(self, game_index: int, move_count: int):
        """Update the move count for a game."""
        with self.lock:
            if game_index in self.games:
                self.games[game_index].move_count = move_count

    def finish_game(self, game_index: int, result: str, nps: Optional[int] = None,
                    outcome: Optional[str] = None):
        """Record that a game has finished.

        Args:
            game_index: The game index
            result: Raw game result for display ("1-0", "0-1", "1/2-1/2", "err")
            nps: Optional NPS to display
            outcome: Interpreted outcome for stats tracking:
                     SPSA: "plus_win", "minus_win", "draw"
                     Ref: "win", "loss", "draw"
        """
        with self.lock:
            if game_index in self.games:
                game = self.games[game_index]
                game.finished = True
                game.result = result
                if nps:
                    game.nps = nps
                self.completed_count += 1
                # Track interpreted outcome by game type
                if outcome:
                    results_dict = self.ref_results if game.game_type == "ref" else self.spsa_results
                    if outcome in results_dict:
                        results_dict[outcome] += 1

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
            # Show full bar with result
            progress = 1.0
            result_str = f" ({status.result})" if status.result else " (done)"
        else:
            # Show progress based on move count (typical game ~80 plies)
            progress = min(1.0, status.move_count / EXPECTED_MOVES)
            result_str = ""

        # Build progress bar
        filled = int(progress * self.BAR_WIDTH)
        empty = self.BAR_WIDTH - filled
        bar = self.FILLED_CHAR * filled + self.EMPTY_CHAR * empty

        # Format NPS
        nps_str = self._format_nps(status.nps)
        if nps_str:
            nps_str = f" {nps_str}"

        # Game number with padding (handle up to 9999 games)
        game_num = f"{self.label} {status.game_index + 1:4d}"

        return f"  {game_num}: {bar}{nps_str}{result_str}"

    def _render(self) -> list[str]:
        """Render all game lines."""
        lines = []

        with self.lock:
            # Sort by game index
            sorted_games = sorted(self.games.values(), key=lambda g: g.game_index)

            # Render each game
            for g in sorted_games:
                lines.append(self._format_game_line(g))

            # Add summary line with SPSA and ref results
            active_count = sum(1 for g in self.games.values() if not g.finished)
            # SPSA results (from plus engine's perspective)
            spsa_plus = self.spsa_results["plus_win"]
            spsa_minus = self.spsa_results["minus_win"]
            spsa_draw = self.spsa_results["draw"]
            spsa_str = f"+{spsa_plus}W-{spsa_minus}L-{spsa_draw}D"
            # Reference results (from base engine's perspective)
            ref_total = self.ref_results["win"] + self.ref_results["loss"] + self.ref_results["draw"]
            if ref_total > 0:
                ref_w = self.ref_results["win"]
                ref_l = self.ref_results["loss"]
                ref_d = self.ref_results["draw"]
                ref_str = f" | Ref: {ref_w}W-{ref_l}L-{ref_d}D"
            else:
                ref_str = ""
            lines.append(f"  [{self.completed_count} done: {spsa_str}{ref_str} | {active_count} active]")

        return lines

    def _clear_lines(self, count: int):
        """Clear the last N lines."""
        if not self.ansi_supported or count == 0:
            return
        # Move cursor up and clear each line
        for _ in range(count):
            sys.stdout.write('\033[F')  # Move up
            sys.stdout.write('\033[K')  # Clear line
        sys.stdout.flush()

    def _display_loop(self):
        """Background thread that updates the display."""
        while self.running:
            self._update_display()
            time.sleep(10)  # Update every 10 seconds to reduce flicker

    def _update_display(self):
        """Update the display (called from background thread)."""
        if not self.ansi_supported:
            return

        lines = self._render()
        if not lines:
            return

        # Clear previous output
        self._clear_lines(self.lines_printed)

        # Print new output
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
            # Clear the live display
            self._clear_lines(self.lines_printed)
            self.lines_printed = 0

    def get_summary(self) -> str:
        """Get a summary line for after completion."""
        with self.lock:
            return f"{self.completed_count} games"
