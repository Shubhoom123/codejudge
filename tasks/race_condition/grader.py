import json
import threading
import sys

try:
    from solution import Counter
except ImportError as e:
    print(json.dumps({"error": f"Could not import Counter: {e}"}))
    sys.exit(1)


def run_test(name: str, fn) -> dict:
    try:
        passed, detail = fn()
        return {"name": name, "passed": passed, "detail": detail}
    except Exception as e:
        return {"name": name, "passed": False, "detail": str(e)}


def test_basic():
    c = Counter()
    for _ in range(100):
        c.increment()
    passed = c.value == 100
    return passed, f"Expected 100, got {c.value}"


def test_two_threads():
    c = Counter()
    threads = [threading.Thread(target=lambda: [c.increment() for _ in range(1000)]) for _ in range(2)]
    for t in threads: t.start()
    for t in threads: t.join()
    passed = c.value == 2000
    return passed, f"Expected 2000, got {c.value}"


def test_high_contention():
    NUM_THREADS = 50
    INCREMENTS_EACH = 1000
    expected = NUM_THREADS * INCREMENTS_EACH

    c = Counter()
    threads = [
        threading.Thread(target=lambda: [c.increment() for _ in range(INCREMENTS_EACH)])
        for _ in range(NUM_THREADS)
    ]
    for t in threads: t.start()
    for t in threads: t.join()

    passed = c.value == expected
    return passed, f"Expected {expected}, got {c.value}"


def test_repeated_runs():
    failures = []
    for i in range(5):
        c = Counter()
        threads = [threading.Thread(target=lambda: [c.increment() for _ in range(500)]) for _ in range(20)]
        for t in threads: t.start()
        for t in threads: t.join()
        if c.value != 10000:
            failures.append(f"Run {i+1}: got {c.value}, expected 10000")

    passed = len(failures) == 0
    detail = "All runs correct" if passed else "; ".join(failures)
    return passed, detail


def test_counter_independence():
    c1, c2 = Counter(), Counter()
    for _ in range(50): c1.increment()
    for _ in range(30): c2.increment()
    passed = c1.value == 50 and c2.value == 30
    return passed, f"c1={c1.value} (want 50), c2={c2.value} (want 30)"


tests = [
    ("basic_single_thread", test_basic),
    ("two_threads", test_two_threads),
    ("high_contention_50_threads", test_high_contention),
    ("repeated_runs_stability", test_repeated_runs),
    ("counter_independence", test_counter_independence),
]

results = [run_test(name, fn) for name, fn in tests]
passed_count = sum(1 for r in results if r["passed"])

output = {
    "tests": results,
    "passed": passed_count,
    "total": len(tests),
    "score": round(passed_count / len(tests), 2),
}

print(json.dumps(output))