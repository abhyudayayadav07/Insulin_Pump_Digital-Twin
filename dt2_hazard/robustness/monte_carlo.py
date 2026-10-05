"""Monte Carlo robustness analysis."""
from dataclasses import dataclass
from typing import Callable, Dict, Optional
import numpy as np


@dataclass(frozen=True)
class MonteCarloResult:
    n_runs: int
    success_count: int
    failure_count: int
    success_rate: float
    mean_score: float
    std_score: float
    worst_score: float


def run_monte_carlo(
    evaluator: Callable[[np.random.Generator], float],
    n_runs: int = 100,
    seed: int = 42,
) -> MonteCarloResult:
    if n_runs <= 0:
        raise ValueError("n_runs must be positive")

    rng = np.random.default_rng(seed)
    scores = np.asarray(
        [float(evaluator(rng)) for _ in range(n_runs)],
        dtype=float,
    )
    success = scores >= 0.5

    return MonteCarloResult(
        n_runs=n_runs,
        success_count=int(success.sum()),
        failure_count=int((~success).sum()),
        success_rate=float(success.mean()),
        mean_score=float(scores.mean()),
        std_score=float(scores.std()),
        worst_score=float(scores.min()),
    )
