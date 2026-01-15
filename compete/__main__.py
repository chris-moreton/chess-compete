"""
Entry point for running the compete package as a module.

Usage:
    python -m compete --help
    python -m compete v1 v2 --games 100 --time 1.0
"""

from compete.cli import main

if __name__ == "__main__":
    main()
