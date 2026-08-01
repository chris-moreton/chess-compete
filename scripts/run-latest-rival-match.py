#!/usr/bin/env python3
"""
Interactively pick 2+ Rusty Rival builds from the last 10 releases in engines/,
choose a randomized time control and concurrency, and run a competition
between them via `python -m compete`.

Usage:
    python scripts/run-latest-rival-match.py
    python scripts/run-latest-rival-match.py --dry-run
"""

import argparse
import os
import re
import subprocess
import sys
from pathlib import Path

try:
    import questionary
except ImportError:
    print("Error: the 'questionary' package is required for the interactive picker.")
    print("Install it with: pip install -r requirements.txt")
    sys.exit(1)

REPO_ROOT = Path(__file__).parent.parent

LAST_N_RELEASES = 10
DEFAULT_GAMES = 400
DEFAULT_TC_BASE_LOW = 30
DEFAULT_TC_BASE_HIGH = 60
DEFAULT_TC_INC_LOW = 0.5
DEFAULT_TC_INC_HIGH = 2
DEFAULT_CONCURRENCY = 12

VERSION_RE = re.compile(r"^v(\d+)\.(\d+)\.(\d+)(?:-rc(\d+))?$")


def get_engines_dir() -> Path:
    """Mirrors compete.engine_manager.get_engines_dir() without importing the
    compete package - that pulls in Flask/python-dotenv/etc. just to resolve
    a path, and would fail here if this script is run without .venv active."""
    if os.environ.get("ENGINES_DIR"):
        return Path(os.environ["ENGINES_DIR"])
    return REPO_ROOT / "engines"


def find_python_interpreter() -> str:
    """Prefer the repo's .venv interpreter so the match subprocess has
    requirements.txt installed, even if this script itself was launched with
    a bare system python (e.g. via its shebang, without .venv activated)."""
    if ".venv" in Path(sys.executable).parts:
        return sys.executable
    for candidate in (REPO_ROOT / ".venv" / "bin" / "python", REPO_ROOT / ".venv" / "Scripts" / "python.exe"):
        if candidate.exists():
            return str(candidate)
    return sys.executable


def version_sort_key(tag: str):
    """Sort key for tags like v1.0.49, v1.0.44-rc7, v1.0.6 - numeric, and a
    final release sorts after its own -rcN builds. Unparseable names (e.g. old
    SPSA dev builds like v001-baseline) sort first, oldest, via a zero key."""
    m = VERSION_RE.match(tag)
    if not m:
        return (0, 0, 0, 0, 0, tag)
    major, minor, patch, rc = m.groups()
    is_final = 0 if rc is not None else 1
    rc_num = int(rc) if rc is not None else 0
    return (int(major), int(minor), int(patch), is_final, rc_num, tag)


def find_rusty_rival_builds(engine_dir: Path) -> list[str]:
    """Rusty Rival build directories are named v<something-starting-with-a-digit>."""
    if not engine_dir.exists():
        return []
    names = [
        d.name for d in engine_dir.iterdir()
        if d.is_dir() and d.name.startswith("v") and len(d.name) > 1 and d.name[1].isdigit()
    ]
    return sorted(names, key=version_sort_key)  # oldest first, newest last


def ask_float(message: str, default: float) -> float:
    raw = questionary.text(
        message, default=str(default),
        validate=lambda text: True if _is_float(text) else "Enter a number",
    ).ask()
    if raw is None:
        sys.exit(0)  # Ctrl-C
    return float(raw)


def ask_int(message: str, default: int) -> int:
    raw = questionary.text(
        message, default=str(default),
        validate=lambda text: True if text.strip().isdigit() else "Enter a whole number",
    ).ask()
    if raw is None:
        sys.exit(0)
    return int(raw)


def _is_float(text: str) -> bool:
    try:
        float(text)
        return True
    except ValueError:
        return False


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true",
                        help="Show the command that would run, but don't launch it")
    args = parser.parse_args()

    engine_dir = get_engines_dir()
    builds = find_rusty_rival_builds(engine_dir)

    if len(builds) < 2:
        print(f"Error: found {len(builds)} Rusty Rival build(s) in {engine_dir}, need at least 2.")
        print("Download some with, e.g.: python scripts/download-all-rival-versions.py")
        sys.exit(1)

    last_n = builds[-LAST_N_RELEASES:]

    selected = questionary.checkbox(
        f"Select 2+ engines for the competition (last {len(last_n)} release(s), newest first):",
        choices=[questionary.Choice(title=name, value=name) for name in reversed(last_n)],
    ).ask()

    if selected is None:
        sys.exit(0)  # Ctrl-C
    if len(selected) < 2:
        print("Error: need at least 2 engines selected.")
        sys.exit(1)

    tc_base_low = ask_float("Base time low (seconds):", DEFAULT_TC_BASE_LOW)
    tc_base_high = ask_float("Base time high (seconds):", DEFAULT_TC_BASE_HIGH)
    if tc_base_low > tc_base_high:
        print("Error: base time low must be <= base time high.")
        sys.exit(1)

    tc_inc_low = ask_float("Increment low (seconds):", DEFAULT_TC_INC_LOW)
    tc_inc_high = ask_float("Increment high (seconds):", DEFAULT_TC_INC_HIGH)
    if tc_inc_low > tc_inc_high:
        print("Error: increment low must be <= increment high.")
        sys.exit(1)

    games = ask_int("Number of games:", DEFAULT_GAMES)
    concurrency = ask_int("Concurrency (games in parallel):", DEFAULT_CONCURRENCY)

    cmd = [
        find_python_interpreter(), "-m", "compete",
        *selected,
        "--games", str(games),
        "--tc-base-low", str(tc_base_low), "--tc-base-high", str(tc_base_high),
        "--tc-inc-low", str(tc_inc_low), "--tc-inc-high", str(tc_inc_high),
        "--concurrency", str(concurrency),
    ]
    print(f"\n$ {' '.join(cmd)}\n")

    if args.dry_run:
        print("(dry run - not launching)")
        return

    if not questionary.confirm(f"Launch {games} games between {', '.join(selected)}?", default=True).ask():
        print("Cancelled.")
        return

    # Not capturing output: compete's own live progress display (and per-game
    # prints at concurrency=1) stream straight to this terminal as games finish.
    result = subprocess.run(cmd, cwd=REPO_ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
