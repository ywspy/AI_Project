import inspect, statistics, time, textwrap
from typing import Callable, Any, Dict, List
import numpy as np
from radon.complexity import cc_visit

# ───────── weights ──────────────────────────────────────────────────────────
ALPHA = 1.0          # + performance
BETA  = 0.1          # – runtime
GAMMA = 0.05         # – cyclomatic–complexity
_MIN_RT = 1e-6       # avoid div-by-zero
# ────────────────────────────────────────────────────────────────────────────


# ───────── helpers ──────────────────────────────────────────────────────────
def _avg_cc(src: str) -> float:
    """mean radon CC, but ≥ 1.0"""
    blocks = cc_visit(src)
    if not blocks:
        return 1.0
    return max(statistics.mean(b.complexity for b in blocks), 1.0)


def _time(func: Callable[[Any], Any], arg: Any, runs: int = 3) -> float:
    t: List[float] = []
    for _ in range(runs):
        beg = time.perf_counter()
        func(arg)
        t.append(time.perf_counter() - beg)
    return max(statistics.mean(t), _MIN_RT)
# ────────────────────────────────────────────────────────────────────────────


def score(
    program: Callable[[Dict[str, Any]], float],
    test_inputs: List[Dict[str, Any]],
    *,
    source_fallback: str | None = None,
) -> Dict[str, float]:
    """
    Multi-objective score on a list of *instances-dicts*.

    `test_inputs` **must be** `[ whole_dataset_dict , … ]`
    so that it matches the reference template’s signature.
    """
    perf, rts = [], []

    for dataset in test_inputs:            # one full dataset per run
        rts.append(_time(program, dataset))
        val = program(dataset)
        if not isinstance(val, (int, float)):
            raise ValueError("Heuristic must return numeric score")
        perf.append(val)

    performance = statistics.mean(perf)
    runtime     = statistics.mean(rts)

    # cyclomatic complexity
    try:
        src = inspect.getsource(program)
    except OSError:
        if source_fallback is None:
            raise
        src = textwrap.dedent(source_fallback)
    cc_val = _avg_cc(src)

    composite = ALPHA*performance - BETA*runtime - GAMMA*cc_val
    if not np.isfinite(composite):
        composite = -1e20

    return {
        "performance": performance,
        "runtime": runtime,
        "cc": cc_val,
        "composite": composite,
    }
