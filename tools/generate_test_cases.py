"""
Generates a massive test suite for parse_amount programmatically.
Covers all combinations of formats, numbers, and edge cases.

Usage:
    python tools/generate_test_cases_bulk.py --output data/test_cases.json
"""

import json
import argparse
import random
from pathlib import Path
from itertools import product

# ─── Reference solution ───────────────────────────────────────────────────────

def parse_amount(text: str) -> float:
    text = text.strip().lower()
    if not text:
        raise ValueError("Empty input")

    text = text.replace("$", "").replace(",", "")
    if "dollars" in text:
        text = text.replace("dollars", "").strip()
    if "dollar" in text:
        text = text.replace("dollar", "").strip()
    if "usd" in text:
        text = text.replace("usd", "").strip()

    text = text.strip()

    try:
        return float(text)
    except ValueError:
        pass

    number_words = {
        "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
        "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
        "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
        "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
        "eighteen": 18, "nineteen": 19, "twenty": 20, "thirty": 30,
        "forty": 40, "fifty": 50, "sixty": 60, "seventy": 70,
        "eighty": 80, "ninety": 90,
    }

    words = text.replace(" and ", " ").split()
    total = 0
    current = 0
    for word in words:
        if word in number_words:
            current += number_words[word]
        elif word == "hundred":
            current = current * 100 if current else 100
        elif word == "thousand":
            total += (current or 1) * 1000
            current = 0
        elif word == "million":
            total += (current or 1) * 1000000
            current = 0
        else:
            raise ValueError(f"Cannot parse: {word!r}")

    result = total + current
    if result == 0 and text not in ("zero", "0", "$0", "0.0"):
        raise ValueError(f"Cannot parse: {text!r}")
    return float(result)


# ─── Generators ───────────────────────────────────────────────────────────────

def gen_dollar_sign(numbers):
    """$12.50, $0, $1000, etc."""
    cases = []
    for n in numbers:
        text = f"${n}"
        try:
            expected = parse_amount(text)
            cases.append({"input": text, "expected": expected, "category": "dollar_sign"})
        except:
            pass
    return cases


def gen_dollar_with_commas(numbers):
    """$1,000  $1,000,000 etc."""
    cases = []
    for n in numbers:
        if n >= 1000:
            text = f"${n:,.0f}"
            try:
                expected = parse_amount(text)
                cases.append({"input": text, "expected": expected, "category": "comma_thousands"})
            except:
                pass
    return cases


def gen_word_dollars(numbers):
    """100 dollars, 50 dollar, etc."""
    cases = []
    for n in numbers:
        for suffix in ["dollars", "dollar"]:
            text = f"{n} {suffix}"
            try:
                expected = parse_amount(text)
                cases.append({"input": text, "expected": expected, "category": "word_dollars"})
            except:
                pass
    return cases


def gen_plain_numbers(numbers):
    """Just a number: 42, 3.14, 0, etc."""
    cases = []
    for n in numbers:
        text = str(n)
        try:
            expected = parse_amount(text)
            cases.append({"input": text, "expected": expected, "category": "plain_number"})
        except:
            pass
    return cases


def gen_whitespace_variants(numbers):
    """   42  , \t100\t, etc."""
    cases = []
    for n in random.sample(numbers, min(500, len(numbers))):
        for prefix, suffix in [("  ", "  "), ("\t", "\t"), ("   ", ""), ("", "   ")]:
            text = f"{prefix}${n}{suffix}"
            try:
                expected = parse_amount(text)
                cases.append({"input": text, "expected": expected, "category": "whitespace"})
            except:
                pass
    return cases


def gen_written_numbers():
    """fifty, one hundred, twenty three, etc."""
    ones = ["zero","one","two","three","four","five","six","seven","eight","nine",
            "ten","eleven","twelve","thirteen","fourteen","fifteen","sixteen",
            "seventeen","eighteen","nineteen"]
    tens = ["twenty","thirty","forty","fifty","sixty","seventy","eighty","ninety"]

    cases = []

    # Single words
    for word in ones + tens:
        try:
            expected = parse_amount(word)
            cases.append({"input": word, "expected": expected, "category": "written_single"})
        except:
            pass

    # Tens + ones: "twenty one", "forty five", etc.
    for ten, one in product(tens, ones[1:]):
        text = f"{ten} {one}"
        try:
            expected = parse_amount(text)
            cases.append({"input": text, "expected": expected, "category": "written_compound"})
        except:
            pass

    # Hundreds: "one hundred", "two hundred fifty", etc.
    for h in ones[1:10]:
        text = f"{h} hundred"
        try:
            expected = parse_amount(text)
            cases.append({"input": text, "expected": expected, "category": "written_hundred"})
        except:
            pass
        for ten in tens:
            text = f"{h} hundred {ten}"
            try:
                expected = parse_amount(text)
                cases.append({"input": text, "expected": expected, "category": "written_hundred_tens"})
            except:
                pass

    # Thousands
    for h in ones[1:10]:
        text = f"{h} thousand"
        try:
            expected = parse_amount(text)
            cases.append({"input": text, "expected": expected, "category": "written_thousand"})
        except:
            pass

    return cases


def gen_negative_numbers(numbers):
    """-$50, -100, etc."""
    cases = []
    for n in numbers:
        if n > 0:
            for fmt in [f"-${n}", f"-{n}", f"- {n}"]:
                try:
                    expected = parse_amount(fmt)
                    cases.append({"input": fmt, "expected": expected, "category": "negative"})
                except:
                    pass
    return cases


def gen_decimals():
    """0.99, .50, 1.5, 100.00, etc."""
    cases = []
    decimal_values = [
        0.01, 0.05, 0.10, 0.25, 0.50, 0.75, 0.99,
        1.5, 2.25, 10.99, 19.95, 99.99, 100.00,
        0.001, 1234.56, 9999.99,
    ]
    for v in decimal_values:
        for fmt in [str(v), f"${v}", f"{v} dollars"]:
            try:
                expected = parse_amount(fmt)
                cases.append({"input": fmt, "expected": expected, "category": "decimal"})
            except:
                pass
    return cases


def gen_invalid_inputs():
    """Inputs that should raise ValueError."""
    invalid = [
        "banana", "xyz", "abc123", "hello world",
        "twelve dollars fifty cents",
        "$$100", "€100", "£50",
        "one two three four five",
        "NaN", "inf", "-inf",
        "1e10",  # scientific notation - edge case
        "   ",  # only whitespace
        "!@#$%",
        "hundred hundred",
        "million billion",
        "one.two",
        "12/50",
    ]
    cases = []
    for text in invalid:
        try:
            parse_amount(text)
            # If it didn't raise, it's not invalid for our reference — skip
        except ValueError:
            cases.append({"input": text, "expected": "raises_ValueError", "category": "invalid"})
        except:
            pass
    return cases


def gen_zero_variants():
    """0, $0, zero, 0.0, 0.00, etc."""
    cases = []
    for fmt in ["0", "$0", "zero", "0.0", "0.00", "  0  ", "$0.00", "0 dollars"]:
        try:
            expected = parse_amount(fmt)
            cases.append({"input": fmt, "expected": expected, "category": "zero"})
        except:
            pass
    return cases


def gen_large_numbers():
    """Very large amounts."""
    cases = []
    large = [1_000_000, 5_000_000, 10_000_000, 999_999_999]
    for n in large:
        for fmt in [f"${n:,}", str(n), f"{n} dollars"]:
            try:
                expected = parse_amount(fmt)
                cases.append({"input": fmt, "expected": expected, "category": "large"})
            except:
                pass
    return cases


# ─── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default="data/test_cases.json")
    parser.add_argument("--limit", type=int, default=100_000)
    args = parser.parse_args()

    print("Generating test cases programmatically...")

    # Base number pool
    integers = list(range(0, 10_000))          # 0–9999
    large_ints = list(range(10_000, 1_000_000, 1_000))  # 10k–1M step 1k
    floats = [round(i * 0.01, 2) for i in range(1, 10_000)]  # 0.01–99.99
    all_numbers = integers + large_ints + floats

    all_cases = []

    generators = [
        ("dollar_sign",       lambda: gen_dollar_sign(all_numbers)),
        ("dollar_commas",     lambda: gen_dollar_with_commas(integers + large_ints)),
        ("word_dollars",      lambda: gen_word_dollars(integers[:500])),
        ("plain_numbers",     lambda: gen_plain_numbers(all_numbers)),
        ("whitespace",        lambda: gen_whitespace_variants(integers)),
        ("written_numbers",   gen_written_numbers),
        ("negative",          lambda: gen_negative_numbers(integers[:500])),
        ("decimals",          gen_decimals),
        ("invalid",           gen_invalid_inputs),
        ("zero_variants",     gen_zero_variants),
        ("large_numbers",     gen_large_numbers),
    ]

    for name, fn in generators:
        cases = fn()
        print(f"  {name}: {len(cases)} cases")
        all_cases.extend(cases)

    # Deduplicate by input
    seen = set()
    unique_cases = []
    for c in all_cases:
        if c["input"] not in seen:
            seen.add(c["input"])
            unique_cases.append(c)

    # Shuffle and limit
    random.shuffle(unique_cases)
    final_cases = unique_cases[:args.limit]

    # Stats
    categories = {}
    for c in final_cases:
        cat = c.get("category", "unknown")
        categories[cat] = categories.get(cat, 0) + 1

    print(f"\nTotal unique cases: {len(final_cases)}")
    print("By category:")
    for cat, count in sorted(categories.items(), key=lambda x: -x[1]):
        print(f"  {cat}: {count}")

    # Save
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(final_cases, indent=2))
    print(f"\nSaved to {args.output}")


if __name__ == "__main__":
    main()