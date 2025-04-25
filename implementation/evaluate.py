import inspect
import statistics
import textwrap
import time
from typing import Any, Callable, Dict, Iterable, List

import numpy as np
from radon.complexity import cc_visit

# ───────────────────────────── weights ───────────────────────────────────────
ALPHA = 1.0        # performance weight
BETA  = 0.1        # runtime penalty
GAMMA = 0.05       # cyclomatic-complexity penalty
MIN_RT = 1e-6      # avoid zero runtime in mean()
# ─────────────────────────────────────────────────────────────────────────────


# ─────────────────────────── helper functions ────────────────────────────────
def _avg_cyclomatic_complexity(source: str) -> float:
    """Mean CC across all code blocks; never returns < 1.0."""
    blocks = cc_visit(source)
    return max(statistics.mean(b.complexity for b in blocks) if blocks else 1.0, 1.0)


def _time_func(func: Callable, arg: Any, runs: int = 3) -> float:
    """Average wall-clock runtime of *func(arg)* over *runs* executions."""
    times: List[float] = []
    for _ in range(runs):
        t0 = time.perf_counter()
        func(arg)
        times.append(time.perf_counter() - t0)
    return max(statistics.mean(times), MIN_RT)


def _iter_instances(test_inputs: Any) -> Iterable[Any]:
    """
    Guaranteed to yield **instances** (never keys).

    * dict  -> its .values()
    * list/tuple -> elements as-is
    * single object -> yields that object once
    """
    if isinstance(test_inputs, dict):
        return test_inputs.values()
    if isinstance(test_inputs, (list, tuple)):
        return test_inputs
    # single instance (e.g. numpy array)
    return (test_inputs,)
# ─────────────────────────────────────────────────────────────────────────────


def score(
    program: Callable,
    test_inputs: Any,
    *,
    source_fallback: str | None = None,
) -> Dict[str, float]:
    """
    Multi-objective evaluation of *program*.

    Parameters
    ----------
    program : Callable[[Any], float]
        Function returned by the template's `evaluate(...)`.
    test_inputs : Any
        List/tuple/dict of instances **or** a single instance.
    source_fallback : str | None, default None
        Source code to analyse if `inspect.getsource` fails (e.g., after
        numba-jit decoration).

    Returns
    -------
    dict
        Keys: {performance, runtime, cc, composite}.  All finite numbers.
    """
    perf: List[float] = []
    rts:  List[float] = []

    for instance in _iter_instances(test_inputs):
        rts.append(_time_func(program, instance))
        result = program(instance)
        if not isinstance(result, (int, float)):
            raise ValueError("Program must return a numeric score.")
        perf.append(result)

    performance = statistics.mean(perf)
    runtime     = statistics.mean(rts)

    # ── cyclomatic complexity ────────────────────────────────────────────────
    try:
        src = inspect.getsource(program)
    except OSError:
        if source_fallback is None:
            raise                     # propagate – caller should handle
        src = textwrap.dedent(source_fallback)

    cc_val = _avg_cyclomatic_complexity(src)

    # ── composite score ─────────────────────────────────────────────────────
    composite = ALPHA * performance - BETA * runtime - GAMMA * cc_val

    # Make absolutely sure everything is finite
    for k, v in (("P", performance), ("R", runtime), ("C", cc_val)):
        if not np.isfinite(v):
            composite = -1e20
            break
    if not np.isfinite(composite):
        composite = -1e20

    return {
        "performance": performance,
        "runtime": runtime,
        "cc": cc_val,
        "composite": composite,
    }
