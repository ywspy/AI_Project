import inspect
import statistics
import time
from typing import Callable, Any, Dict, List

from radon.complexity import cc_visit
import numpy as np

# ──────────────────────────────────────────────────────────────────────────────
# Weights for the composite score
ALPHA = 1.0      # performance weight
BETA  = 0.1      # runtime penalty
GAMMA = 0.05     # cyclomatic-complexity penalty
# ──────────────────────────────────────────────────────────────────────────────


# ──────────────────────────────  helpers  ─────────────────────────────────────
def _avg_cyclomatic_complexity(source: str) -> float:
    """Mean CC over all blocks returned by radon; at least 1.0."""
    blocks = cc_visit(source)
    if not blocks:
        return 1.0
    return max(statistics.mean(b.complexity for b in blocks), 1.0)


def _time_func(func: Callable, arg: Any, runs: int = 3) -> float:
    """
    Average wall-clock time to run *func(arg)*, never returning 0.
    We **do not unpack arg** – every instance is passed verbatim.
    """
    elapsed: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        func(arg)
        elapsed.append(time.perf_counter() - start)

    mean_rt = statistics.mean(elapsed)
    return max(mean_rt, 1e-6)        # avoid zero-runtime → 1/0
# ──────────────────────────────────────────────────────────────────────────────


def score(program: Callable, test_inputs: List[Any]) -> Dict[str, float]:
    """
    Multi-objective evaluation of *program* on *test_inputs*.

    Returns a dict with keys
        {performance | runtime | cc | composite}
    All values are guaranteed to be finite.
    """
    perf:  List[float] = []
    rts:   List[float] = []

    # ── run the program on every instance ────────────────────────────────────
    for instance in test_inputs:
        # 1) runtime
        rts.append(_time_func(program, instance))

        # 2) performance (program must return a number)
        result = program(instance)
        if not isinstance(result, (int, float)):
            raise ValueError("Program must return a numeric score.")
        perf.append(result)
    # ─────────────────────────────────────────────────────────────────────────

    performance = statistics.mean(perf)
    runtime     = statistics.mean(rts)

    # 3) cyclomatic complexity of the program source
    src = inspect.getsource(program)
    cc  = _avg_cyclomatic_complexity(src)

    # 4) safety clamp: all metrics must be finite
    if not np.isfinite(performance):
        performance = -1e9
    if not np.isfinite(runtime):
        runtime = 1e6
    if not np.isfinite(cc):
        cc = 1e6

    # 5) composite
    composite = ALPHA * performance - BETA * runtime - GAMMA * cc

    return {
        "performance": performance,
        "runtime": runtime,
        "cc": cc,
        "composite": composite,
    }
