"""
SPSA (Simultaneous Perturbation Stochastic Approximation) tuning support.

This module provides worker functionality for distributed SPSA parameter tuning.
Workers poll the database for pending iterations and run games between
perturbed engine pairs.
"""

from compete.spsa.worker import run_spsa_worker

__all__ = ['run_spsa_worker']
