import inspect
import statistics
import time
import textwrap
from typing import Callable, Any, Dict, List

import numpy as np
from radon.complexity import cc_visit

# ───────── weights ──────────────────────────────────────────────────────────
ALPHA  = 1.0          # + performance
BETA   = 0.1          # – runtime
GAMMA  = 0.05         # – cyclomatic-complexity
_MIN_RT = 1e-6        # avoid div-by-zero
_WORST  = -1e20       # score used when a run fails
# ────────────────────────────────────────────────────────────────────────────


# ───────── helpers ──────────────────────────────────────────────────────────
def _avg_cc(src: str) -> float:
    """Mean Radon CC (≥ 1)."""
    blocks = cc_visit(src)
    if not blocks:
        return 1.0
    return max(statistics.mean(b.complexity for b in blocks), 1.0)


def _time(func: Callable[[Any], Any], arg: Any, runs: int = 3) -> float:
    """
    Average wall-clock runtime of *func(arg)*.

    If *func* raises, we immediately return **inf** so the candidate
    is punished but the search keeps running.
    """
    timings: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        try:
            func(arg)
        except Exception:        # noqa: BLE001 – anything from user code
            return float("inf")
        timings.append(time.perf_counter() - start)

    return max(statistics.mean(timings), _MIN_RT)
# ────────────────────────────────────────────────────────────────────────────


def score(
    program: Callable[[Dict[str, Any]], float],
    test_inputs: List[Dict[str, Any]],
    *,
    source_fallback: str | None = None,
) -> Dict[str, float]:
    """
    Multi-objective evaluation on *test_inputs*.

    `test_inputs` **must** be `[whole_dataset_dict, …]` to match the
    reference template’s signature.
    """
    perf: List[float] = []
    rts:  List[float] = []

    for dataset in test_inputs:          # one full dataset per run
        rt = _time(program, dataset)
        # If timing already failed → penalise and abort early
        if not np.isfinite(rt):
            return {
                "performance": _WORST,
                "runtime":     rt,
                "cc":          1e6,
                "composite":   _WORST,
            }

        try:
            val = program(dataset)
        except Exception:
            return {
                "performance": _WORST,
                "runtime":     rt,
                "cc":          1e6,
                "composite":   _WORST,
            }

        if not isinstance(val, (int, float)):
            return {
                "performance": _WORST,
                "runtime":     rt,
                "cc":          1e6,
                "composite":   _WORST,
            }

        perf.append(val)
        rts.append(rt)

    performance = statistics.mean(perf)
    runtime     = statistics.mean(rts)

    # ── cyclomatic complexity ────────────────────────────────────────────
    try:
        src = inspect.getsource(program)
    except OSError:
        if source_fallback is None:
            raise
        src = textwrap.dedent(source_fallback)
    cc_val = _avg_cc(src)

    # ── composite ────────────────────────────────────────────────────────
    composite = ALPHA * performance - BETA * runtime - GAMMA * cc_val
    if not np.isfinite(composite):
        composite = _WORST

    return {
        "performance": performance,
        "runtime":     runtime,
        "cc":          cc_val,
        "composite":   composite,
    }
