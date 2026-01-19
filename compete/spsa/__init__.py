"""
SPSA (Simultaneous Perturbation Stochastic Approximation) tuning support.

This module provides distributed SPSA parameter tuning for Rusty Rival:

- Master controller: Orchestrates tuning, generates perturbations, calculates gradients
- Worker: Polls database, builds engines, runs games between perturbed pairs

Usage:
    # Run master controller (generates iterations, waits for results)
    python -m compete.spsa.master

    # Run worker (polls for work, builds engines, runs games)
    python -m compete --spsa -c 6

Configuration:
    - config.toml: SPSA hyperparameters, time control, build settings
    - params.toml: Current parameter values being tuned

The rusty-rival source path is configurable in config.toml (default: ../rusty-rival).
"""

from compete.spsa.worker import run_spsa_worker
from compete.spsa.master import run_master
from compete.spsa.build import build_engine, build_spsa_engines, get_rusty_rival_path

__all__ = [
    'run_spsa_worker',
    'run_master',
    'build_engine',
    'build_spsa_engines',
    'get_rusty_rival_path',
]
