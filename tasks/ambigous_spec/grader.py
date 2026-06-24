import json
import sys

try:
    from solution import parse_amount
except ImportError as e:
    print(json.dumps({"error": f"Could not import parse_amount: {e}"}))
    sys.exit(1)


def run_test(name, fn):
    try:
        passed, detail = fn()
        return {"name": name, "passed": passed, "detail": detail}
    except Exception as e:
        return {"name": name, "passed": False, "detail": str(e)}


def test_dollar_sign():
    r = parse_amount("$12.50")
    return abs(r - 12.5) < 0.001, f"Got {r}"

def test_word_dollars():
    r = parse_amount("100 dollars")
    return abs(r - 100.0) < 0.001, f"Got {r}"

def test_written_number():
    r = parse_amount("fifty")
    return abs(r - 50.0) < 0.001, f"Got {r}"

def test_comma_thousands():
    r = parse_amount("$1,000")
    return abs(r - 1000.0) < 0.001, f"Got {r}"

def test_negative():
    try:
        r = parse_amount("-$50")
        return abs(r - (-50.0)) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_zero():
    r = parse_amount("$0")
    return abs(r - 0.0) < 0.001, f"Got {r}"

def test_whitespace():
    r = parse_amount("  $42  ")
    return abs(r - 42.0) < 0.001, f"Got {r}"

def test_invalid_raises():
    try:
        parse_amount("banana")
        return False, "Should have raised ValueError"
    except ValueError:
        return True, "Correctly raised ValueError"
    except Exception as e:
        return False, f"Raised wrong exception: {type(e).__name__}: {e}"

def test_empty_string_raises():
    try:
        parse_amount("")
        return False, "Should have raised ValueError"
    except ValueError:
        return True, "Correctly raised ValueError"
    except Exception as e:
        return False, f"Raised wrong exception: {type(e).__name__}: {e}"

def test_large_written():
    try:
        r = parse_amount("one hundred")
        return abs(r - 100.0) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_decimal_cents():
    try:
        r = parse_amount("$0.99")
        return abs(r - 0.99) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_written_thousand():
    try:
        r = parse_amount("one thousand")
        return abs(r - 1000.0) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_gibberish_raises():
    try:
        parse_amount("xyz abc")
        return False, "Should have raised ValueError"
    except ValueError:
        return True, "Correctly raised ValueError"
    except Exception as e:
        return False, f"Wrong exception: {type(e).__name__}: {e}"

def test_large_number():
    try:
        r = parse_amount("$1234567.89")
        return abs(r - 1234567.89) < 0.01, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_large_written_million():
    try:
        r = parse_amount("one million")
        return abs(r - 1000000.0) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_leading_zeroes():
    try:
        r = parse_amount("007")
        return abs(r - 7.0) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_decimal_only():
    try:
        r = parse_amount(".50")
        return abs(r - 0.5) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_whitespace_before():
    try:
        r = parse_amount("   100")
        return abs(r - 100.0) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_whitespace_after():
    try:
        r = parse_amount("100   ")
        return abs(r - 100.0) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_negative_plain():
    try:
        r = parse_amount("-50")
        return abs(r - (-50.0)) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_singular_dollar():
    try:
        r = parse_amount("68 dollar")
        return abs(r - 68.0) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_tab_whitespace():
    try:
        r = parse_amount("\t$100\t")
        return abs(r - 100.0) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_comma_decimal():
    try:
        r = parse_amount("$1,234.56")
        return abs(r - 1234.56) < 0.001, f"Got {r}"
    except Exception as e:
        return False, f"Raised {e}"

def test_number_with_letters_raises():
    try:
        parse_amount("123abc456")
        return False, "Should have raised ValueError"
    except ValueError:
        return True, "Correctly raised ValueError"
    except Exception as e:
        return False, f"Wrong exception: {type(e).__name__}: {e}"


tests = [
    ("dollar_sign",               test_dollar_sign),
    ("word_dollars",              test_word_dollars),
    ("written_number_single",     test_written_number),
    ("comma_thousands",           test_comma_thousands),
    ("negative_amount",           test_negative),
    ("zero",                      test_zero),
    ("extra_whitespace",          test_whitespace),
    ("invalid_raises_valueerror", test_invalid_raises),
    ("empty_string_raises",       test_empty_string_raises),
    ("compound_written_number",   test_large_written),
    ("decimal_cents",             test_decimal_cents),
    ("written_thousand",          test_written_thousand),
    ("gibberish_raises",          test_gibberish_raises),
    ("large_number",              test_large_number),
    ("large_written_million",     test_large_written_million),
    ("leading_zeroes",            test_leading_zeroes),
    ("decimal_only",              test_decimal_only),
    ("whitespace_before",         test_whitespace_before),
    ("whitespace_after",          test_whitespace_after),
    ("negative_plain",            test_negative_plain),
    ("singular_dollar",           test_singular_dollar),
    ("tab_whitespace",            test_tab_whitespace),
    ("comma_decimal",             test_comma_decimal),
    ("number_with_letters_raises",test_number_with_letters_raises),
]

results = [run_test(name, fn) for name, fn in tests]
passed_count = sum(1 for r in results if r["passed"])

print(json.dumps({
    "tests": results,
    "passed": passed_count,
    "total": len(tests),
    "score": round(passed_count / len(tests), 2),
}))