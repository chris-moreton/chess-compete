"""
SPSA (Simultaneous Perturbation Stochastic Approximation) tuning support.

This module provides distributed SPSA parameter tuning for Rusty Rival:

- Master controller: Orchestrates tuning, generates perturbations, calculates gradients
- HTTP worker: Polls API for work, builds engines locally from parameter JSON, runs games

Usage:
    # Run master controller (generates iterations, waits for results)
    python -m compete.spsa.master

    # Run HTTP worker (polls for work, builds engines, runs games)
    python -m compete --spsa-http -c 6

Configuration:
    - config.toml: SPSA hyperparameters, time control, build settings
    - Database spsa_params table: Current parameter values being tuned

The rusty-rival source path is configurable in config.toml (default: ../rusty-rival).
"""

from compete.spsa.master import run_master
from compete.spsa.build import build_engine, build_spsa_engines, get_rusty_rival_path

__all__ = [
    'run_master',
    'build_engine',
    'build_spsa_engines',
    'get_rusty_rival_path',
]
