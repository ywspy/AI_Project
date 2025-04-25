"""
Multi–objective scorer used by the Sandbox.

It returns a dict with four finite metrics:
    performance  – average fitness returned by `program`
    runtime      – average wall-clock time (seconds)
    cc           – mean cyclomatic complexity
    composite    – weighted sum ALPHA – BETA – GAMMA  (higher is better)
"""

from __future__ import annotations
import inspect
import statistics
import time
from typing import Any, Callable, Dict, List

import numpy as np
from radon.complexity import cc_visit


# ────────────────────────────────────────── weights
ALPHA = 1.0      # reward for performance
BETA  = 0.1      # penalty per second
GAMMA = 0.05     # penalty per CC point
# ───────────────────────────────────────────────────


# ───────────────────────── helpers ─────────────────────────
def _avg_cyclomatic_complexity(src: str) -> float:
    """Mean cyclomatic complexity over all code blocks (min 1.0)."""
    blocks = cc_visit(src)
    return max(statistics.mean(b.complexity for b in blocks) if blocks else 1.0, 1.0)


def _time_func(func: Callable, arg: Any, runs: int = 3) -> float:
    """
    Average wall-clock time to run *func(arg)*, never returning 0.
    `arg` is passed verbatim (no **kw unpacking!).
    """
    elapsed: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        func(arg)
        elapsed.append(time.perf_counter() - start)

    mean_rt = statistics.mean(elapsed)
    return max(mean_rt, 1e-6)         # avoid 0 → div/weight issues
# ────────────────────────────────────────────────────────────


# ───────────────────────── main entry ──────────────────────
def score(program: Callable, data: Any) -> Dict[str, float]:
    """
    Compute all four metrics in one call.

    *data* is the **entire dataset** (commonly a nested dict).
    The caller guarantees `program(data)` is the right way
    to evaluate fitness for this problem.
    """
    # 1. runtime  (average over a few runs)
    runtime = _time_func(program, data)

    # 2. performance  (single evaluation is enough)
    perf_val = program(data)
    if not isinstance(perf_val, (int, float)):
        raise ValueError("Program must return a numeric score (int|float).")

    # 3. cyclomatic complexity
    cc_val = _avg_cyclomatic_complexity(inspect.getsource(program))

    # 4. clamp to finite numbers
    if not np.isfinite(perf_val):
        perf_val = -1e9
    if not np.isfinite(runtime):
        runtime = 1e6
    if not np.isfinite(cc_val):
        cc_val = 1e6

    # 5. composite
    composite = ALPHA * perf_val - BETA * runtime - GAMMA * cc_val

    return {
        "performance": perf_val,
        "runtime": runtime,
        "cc": cc_val,
        "composite": composite,
    }
# ───────────────────────────────────────────────────────────
