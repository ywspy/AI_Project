import inspect, statistics, time, textwrap
from typing import Callable, Any, Dict, List

import numpy as np
from radon.complexity import cc_visit

ALPHA = 1.0      # performance weight
BETA  = 0.1      # runtime penalty
GAMMA = 0.05     # cyclomatic-complexity penalty
MIN_RT = 1e-6    # avoid div-by-zero in runtime

# ───────────────────────────────── helpers ────────────────────────────────────
def _avg_cyclomatic_complexity(source: str) -> float:
    blocks = cc_visit(source)
    return max(statistics.mean(b.complexity for b in blocks) if blocks else 1.0, 1.0)

def _time_func(func: Callable, arg: Any, runs: int = 3) -> float:
    t = []
    for _ in range(runs):
        start = time.perf_counter()
        func(arg)
        t.append(time.perf_counter() - start)
    return max(statistics.mean(t), MIN_RT)
# ──────────────────────────────────────────────────────────────────────────────


def score(program: Callable,
          test_inputs: List[Any],
          source_fallback: str | None = None) -> Dict[str, float]:
    """
    Multi-objective evaluation.

    Parameters
    ----------
    program : callable
    test_inputs : list of instances (each passed verbatim)
    source_fallback : str | None
        Source code to use if `inspect.getsource` fails (e.g. numba-wrapped).

    Returns
    -------
    dict with keys  {performance, runtime, cc, composite}  – all finite.
    """
    perf, rts = [], []
    for data in test_inputs:
        rts.append(_time_func(program, data))
        res = program(data)
        if not isinstance(res, (int, float)):
            raise ValueError("Program must return a numeric score.")
        perf.append(res)

    performance = statistics.mean(perf)
    runtime     = statistics.mean(rts)

    # ── cyclomatic complexity ────────────────────────────────────────────────
    try:
        src = inspect.getsource(program)
    except OSError:
        if source_fallback is None:
            raise
        src = textwrap.dedent(source_fallback)
    cc_val = _avg_cyclomatic_complexity(src)

    # ── composite ────────────────────────────────────────────────────────────
    composite = ALPHA*performance - BETA*runtime - GAMMA*cc_val

    # ensure finiteness
    if not np.isfinite(composite):
        composite = -1e20

    return {
        "performance": performance,
        "runtime": runtime,
        "cc": cc_val,
        "composite": composite,
    }
