# Task: Parse User Input

## Problem

Write a function `parse_amount(text: str) -> float` that extracts a monetary amount from a user-typed string.

### Examples

```
"$12.50"     → 12.5
"100 dollars" → 100.0
"fifty"      → 50.0
"$1,000"     → 1000.0
```

### Notes
- Handle common formats users actually type.
- Return a `float`.
- If the input cannot be parsed, raise a `ValueError`.