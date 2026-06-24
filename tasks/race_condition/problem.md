# Task: Thread-Safe Counter

## Problem

Implement a `Counter` class in Python that is **safe to use across multiple threads**.

### Requirements

- `Counter` must have an `increment()` method that adds 1 to an internal count.
- `Counter` must have a `value` property that returns the current count.
- The counter must produce **correct results** when `increment()` is called concurrently from many threads.

### Starter

```python
class Counter:
    def __init__(self):
        self.count = 0

    def increment(self):
        self.count += 1

    @property
    def value(self):
        return self.count
```

The starter above has a race condition. Fix it.

### Notes
- Do NOT use multiprocessing — use `threading` only.
- Do not add any sleeps or artificial delays.
- Your solution file must define a class named `Counter`.