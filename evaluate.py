# evaluate.py
import inspect
import statistics
import time
from radon.complexity import cc_visit

###############################################################################
# ① 你可以根据需要把这些权重调高 / 调低；alpha 越大越看重性能。
ALPHA = 1.0        # 性能权重（越大越好）
BETA  = 0.1        # 运行时间权重（越小越好）
GAMMA = 0.05       # 圈复杂度权重（越小越好）
###############################################################################


def _avg_cyclomatic_complexity(source: str) -> float:
    """平均 CC；若 radon 未找到块则返回 1."""
    blocks = cc_visit(source)
    if not blocks:
        return 1.0
    return statistics.mean(b.complexity for b in blocks)


def _time_func(func, arg, n: int = 3) -> float:
    """跑 n 次取平均运行时间(秒)。"""
    times = []
    for _ in range(n):
        start = time.perf_counter()  # 单调时钟，高精度 :contentReference[oaicite:1]{index=1}
        func(arg)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)


def score(program, test_inputs) -> dict:
    """
    返回三项原始指标 + 一个加权综合分：
        performance : float  # 原始 FunSearch 返回值 (越大越好)
        runtime     : float  # 平均秒 (越小越好)
        cc          : float  # 平均圈复杂度 (越小越好)
        composite   : float  # ALPHA*perf - BETA*time - GAMMA*cc
    """
    # --------- 1. 运行测试集合，统计性能 & 运行时间 ---------
    perf_scores = []
    runtimes = []
    for inp in test_inputs:
        runtime = _time_func(program, inp)
        result  = program(inp)
        if not isinstance(result, (int, float)):
            raise ValueError("program must return int or float performance.")
        perf_scores.append(result)
        runtimes.append(runtime)

    perf   = statistics.mean(perf_scores)
    t_mean = statistics.mean(runtimes)

    # --------- 2. 圈复杂度（可读性 proxy） ---------
    src  = inspect.getsource(program)
    cc   = _avg_cyclomatic_complexity(src)

    # --------- 3. 综合得分 ---------
    composite = ALPHA * perf - BETA * t_mean - GAMMA * cc
    return {
        "performance": perf,
        "runtime": t_mean,
        "cc": cc,
        "composite": composite,
    }
