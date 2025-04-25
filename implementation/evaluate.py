import inspect, time, statistics
from radon.complexity import cc_visit

ALPHA = 1.0
BETA  = 0.1
GAMMA = 0.05

def _avg_cyclomatic_complexity(source: str) -> float:
    blocks = cc_visit(source)
    return statistics.mean(b.complexity for b in blocks) if blocks else 1.0

def _time_func(func, arg, runs: int = 3) -> float:
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        # call with unpacking:
        if isinstance(arg, dict):
            func(**arg)
        elif isinstance(arg, (list, tuple)):
            func(*arg)
        else:
            func(arg)
        times.append(time.perf_counter() - start)
    return statistics.mean(times)

def score(program, test_inputs) -> dict:
    perf, runtimes = [], []
    for inp in test_inputs:
        runtimes.append(_time_func(program, inp))
        # same unpacking logic for measuring result:
        if isinstance(inp, dict):
            result = program(**inp)
        elif isinstance(inp, (list, tuple)):
            result = program(*inp)
        else:
            result = program(inp)

        if not isinstance(result, (int, float)):
            raise ValueError("Program must return a numeric score.")
        perf.append(result)

    performance = statistics.mean(perf)
    avg_runtime = statistics.mean(runtimes)

    src = inspect.getsource(program)
    cc  = _avg_cyclomatic_complexity(src)
    composite = ALPHA*performance - BETA*avg_runtime - GAMMA*cc

    return {
        "performance": performance,
        "runtime": avg_runtime,
        "cc": cc,
        "composite": composite,
    }
