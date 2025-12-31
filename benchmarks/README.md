# Performance Benchmarks

This directory contains the unified benchmark suite for `datawarden`.

## Unified Benchmark Suite (`benchmark.py`)

All benchmarks have been consolidated into a single, robust script that measures decorator overhead, validator stacking, data scaling, and DataFrame patterns.

### Usage

Run the full benchmark suite:
```bash
python benchmarks/benchmark.py
```

Run as a CI performance gatekeeper (strict overhead check):
```bash
python benchmarks/benchmark.py --ci
```

### Design Principles

1. **Min Time:** Uses the minimum execution time across trials to represent the true cost without OS interference.
2. **Warmup:** Includes warmup phases to reach steady-state execution.
3. **Statistical Significance:** The CI mode uses a 95% confidence interval to distinguish true regressions from environment noise.
4. **Side-by-Side Comparison:** Compares implementations in the same process to reduce variance.

### Results Summary (Typical)

| Scenario | skip_validation=True Overhead | skip_validation=False Overhead |
|----------|------------------------|----------------------|
| Small Series (100 elements) | **~0.5µs/call** | ~200µs/call |
| Large Series (10,000 elements) | **~0.7µs/call** | ~220µs/call |
| Multiple Validators (2) | **~0.3µs/call** | ~280µs/call |

### Implementation Details

The core optimization is a "Fast Path" that avoids `inspect.signature().bind()` when validation is skipped:

```python
def wrapper(*args, **kwargs):
    # Fast path: check for skip flag directly in kwargs
    skip = kwargs.get("skip_validation", False)
    if skip:
        return func(*args, **kwargs)

    # Slow path: only execute when validation is active
    # ...
```
