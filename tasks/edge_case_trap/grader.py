import json
import sys

try:
    from solution import most_frequent
except ImportError as e:
    print(json.dumps({"error": f"Could not import most_frequent: {e}"}))
    sys.exit(1)


def run_test(name, fn):
    try:
        passed, detail = fn()
        return {"name": name, "passed": passed, "detail": detail}
    except Exception as e:
        return {"name": name, "passed": False, "detail": str(e)}


def test_basic_int():
    r = most_frequent([1, 2, 2, 3])
    return r == 2, f"Got {r}"

def test_basic_str():
    r = most_frequent(["a", "b", "a"])
    return r == "a", f"Got {r}"

def test_single_element():
    r = most_frequent([7])
    return r == 7, f"Got {r}"

def test_all_same():
    r = most_frequent([5, 5, 5, 5])
    return r == 5, f"Got {r}"

def test_none_in_list():
    r = most_frequent([None, None, 1])
    return r is None, f"Got {r}"

def test_bool_vs_int():
    r = most_frequent([True, True, 1, 1, 1])
    return r in (True, 1), f"Got {r!r} (type={type(r).__name__})"

def test_large_list_performance():
    lst = [1] * 5000 + [2] * 3000 + [3] * 2000
    import random; random.shuffle(lst)
    r = most_frequent(lst)
    return r == 1, f"Got {r}"

def test_float_elements():
    r = most_frequent([1.1, 2.2, 1.1, 3.3])
    return abs(r - 1.1) < 1e-9, f"Got {r}"

def test_tie_returns_a_winner():
    r = most_frequent([1, 2])
    return r in (1, 2), f"Got {r}"

def test_negative_numbers():
    r = most_frequent([-1, -1, -2])
    return r == -1, f"Got {r}"


tests = [
    ("basic_int", test_basic_int),
    ("basic_str", test_basic_str),
    ("single_element", test_single_element),
    ("all_same", test_all_same),
    ("none_in_list", test_none_in_list),
    ("bool_vs_int_edge", test_bool_vs_int),
    ("large_list", test_large_list_performance),
    ("float_elements", test_float_elements),
    ("tie_returns_winner", test_tie_returns_a_winner),
    ("negative_numbers", test_negative_numbers),
]

results = [run_test(name, fn) for name, fn in tests]
passed_count = sum(1 for r in results if r["passed"])

print(json.dumps({
    "tests": results,
    "passed": passed_count,
    "total": len(tests),
    "score": round(passed_count / len(tests), 2),
}))