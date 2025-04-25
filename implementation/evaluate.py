# evaluate.py
import inspect
import statistics
import time
from radon.complexity import cc_visit

# You can adjust these weights as needed; increasing ALPHA puts more emphasis on performance.
ALPHA = 1.0   # Weight for performance (higher is better)
BETA = 0.1    # Weight for runtime penalty (lower runtime is better)
GAMMA = 0.05  # Weight for cyclomatic complexity penalty (lower CC is better)

def _avg_cyclomatic_complexity(source: str) -> float:
    """
    Compute the average cyclomatic complexity of all code blocks in the source.
    If radon finds no blocks, returns a default of 1.0.
    """
    blocks = cc_visit(source)
    if not blocks:
        return 1.0
    return statistics.mean(block.complexity for block in blocks)

def _time_func(func, arg, runs: int = 3) -> float:
    """
    Measure the execution time of calling func(arg), averaged over a given number of runs.
    Returns the average duration in seconds.
    """
    times = []
    for _ in range(runs):
        start = time.perf_counter()
        if callable(func):
            func(arg)
        else:
            print(f"{func} is not callable")
        times.append(time.perf_counter() - start)
    return statistics.mean(times)

def score(program, test_inputs) -> dict:
    """
    Run the program across all test_inputs and compute four metrics:
      - performance: average return value of program(inp) (higher is better)
      - runtime: average execution time in seconds (lower is better)
      - cc: average cyclomatic complexity of the source (lower is better)
      - composite: combined score = ALPHA*performance - BETA*runtime - GAMMA*cc

    Returns a dict with keys: "performance", "runtime", "cc", "composite".
    """
    perf_scores = []
    runtimes = []
    # 1. Evaluate performance and runtime
    for inp in test_inputs:
        runtime = _time_func(program, inp)
        result = program(inp)
        if not isinstance(result, (int, float)):
            raise ValueError("program must return an int or float performance metric.")
        perf_scores.append(result)
        runtimes.append(runtime)

    performance = statistics.mean(perf_scores)
    avg_runtime = statistics.mean(runtimes)

    # 2. Compute average cyclomatic complexity of the source
    src = inspect.getsource(program)
    cc = _avg_cyclomatic_complexity(src)

    # 3. Compute the composite score
    composite = ALPHA * performance - BETA * avg_runtime - GAMMA * cc

    return {
        "performance": performance,
        "runtime": avg_runtime,
        "cc": cc,
        "composite": composite,
    }
