import inspect
import statistics
import time
import textwrap
from typing import Callable, Any, Dict, List
import numpy as np
from radon.complexity import cc_visit

# ───────── weights and thresholds ──────────────────────────────────────────
ALPHA = 1.0  # + performance
BETA = 0  # - runtime (positive: punish slow; negative: reward complexity within bounds)
GAMMA = -0.08  # - cyclomatic-complexity (same logic)
_MIN_RT = 1e-6
_WORST = -1e20

RUNTIME_THRESHOLD = 2  # runtime threshold
COMPLEXITY_THRESHOLD = 15.0  # cyclomatic-complexity threshold


# ────────────────────────────────────────────────────────────────────────────

def _avg_cc(src: str) -> float:
    """Mean Radon CC (≥ 1)."""
    blocks = cc_visit(src)
    if not blocks:
        return 1.0
    return max(statistics.mean(b.complexity for b in blocks), 1.0)


def _time(func: Callable[[Any], Any], arg: Any, runs: int = 3) -> float:
    """Average wall-clock runtime of *func(arg)*."""
    timings: List[float] = []
    for _ in range(runs):
        start = time.perf_counter()
        try:
            func(arg)
        except Exception:
            return float("inf")
        timings.append(time.perf_counter() - start)
    return max(statistics.mean(timings), _MIN_RT)


def _penalize(value: float, weight: float, threshold: float) -> float:
    """
    Custom penalty or reward function.
    """
    if weight >= 0:
        # Normal mode: any deviation is bad
        return abs(value - threshold) * weight
    else:
        # Encouragement mode: within threshold is rewarded, beyond threshold is punished
        if value <= threshold:
            return -(threshold - value) * abs(weight)  # reward: closer to threshold better
        else:
            return (value - threshold) * abs(weight)  # penalty: worse when exceeding


def score(
        program: Callable[[Dict[str, Any]], float],
        test_inputs: List[Dict[str, Any]],
        *,
        source_fallback: str | None = None,
) -> Dict[str, float]:
    """
    Multi-objective evaluation on *test_inputs*.
    """
    perf: List[float] = []
    rts: List[float] = []

    for dataset in test_inputs:
        rt = _time(program, dataset)
        if not np.isfinite(rt):
            return {
                "performance": _WORST,
                "runtime": rt,
                "cc": 1e6,
                "composite": _WORST,
            }

        try:
            val = program(dataset)
        except Exception:
            return {
                "performance": _WORST,
                "runtime": rt,
                "cc": 1e6,
                "composite": _WORST,
            }

        if not isinstance(val, (int, float)):
            return {
                "performance": _WORST,
                "runtime": rt,
                "cc": 1e6,
                "composite": _WORST,
            }

        perf.append(val)
        rts.append(rt)

    performance = statistics.mean(perf)
    runtime = statistics.mean(rts)

    try:
        src = inspect.getsource(program)
    except OSError:
        if source_fallback is None:
            raise
        src = textwrap.dedent(source_fallback)
    cc_val = _avg_cc(src)

    # ── customized composite score ──
    composite = (
            ALPHA * performance
            - _penalize(runtime, BETA, RUNTIME_THRESHOLD)
            - _penalize(cc_val, GAMMA, COMPLEXITY_THRESHOLD)
    )

    if not np.isfinite(composite):
        composite = _WORST

    return {
        "performance": performance,
        "runtime": runtime,
        "cc": cc_val,
        "composite": composite,
    }
