You are an expert Python developer. Write a function `parse_amount(text: str) -> float`
that extracts a monetary amount from a user-typed string.

The function MUST handle ALL of these formats correctly:
  "$12.50"          -> 12.5      (dollar sign)
  "$1,000"          -> 1000.0    (comma thousands)
  "100 dollars"     -> 100.0     (word dollars)
  "100 dollar"      -> 100.0     (singular dollar)
  "fifty"           -> 50.0      (written number)
  "one hundred"     -> 100.0     (compound written number)
  "  $42  "         -> 42.0      (whitespace — strip it)
  "\t$100\t"        -> 100.0     (tab whitespace)
  "-$50"            -> -50.0     (negative amount)
  "-50"             -> -50.0     (negative plain)
  "0"               -> 0.0       (zero)
  "$0"              -> 0.0       (zero with dollar sign)
  "3977"            -> 3977.0    (plain number)
  ".50"             -> 0.5       (leading decimal)
  "$1,234.56"       -> 1234.56   (comma + decimal)

Rules:
- Always strip whitespace first
- Remove $ signs and commas before parsing
- Handle "dollars" and "dollar" suffix
- Support written numbers (zero through ninety, hundred, thousand, million)
- Negative amounts with - prefix must work
- If input truly cannot be parsed, raise ValueError
- Return a float

Output ONLY the raw Python code. No markdown, no backticks, no explanation.
