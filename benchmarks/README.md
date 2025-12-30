# Benchmark Results

## Decorator Performance

Benchmark measuring the overhead of the `@validate` decorator with 10,000 iterations:

### Results Summary (After Optimization)

| Scenario | skip_validation=True Overhead | skip_validation=False Overhead |
|----------|------------------------|----------------------|
| Small Series (100 elements) | **0.41µs/call** | 210.22µs/call |
| Large Series (10,000 elements) | **0.75µs/call** | 226.42µs/call |
| Multiple Validators (2) | **0.30µs/call** | 285.67µs/call |
| Index Validators (3) | **0.60µs/call** | 236.42µs/call |

### Key Findings

1. **`skip_validation=True` overhead is essentially zero (~0.5µs per call)**, achieved by:
   - Checking `kwargs.get("skip_validation", ...)` directly
   - Only using `signature.bind()` when validation is actually needed

2. **`skip_validation=False` overhead scales with**:
   - Number of validators (more validators = more time)
   - Validator complexity (Index validators do more work)
   - Data size (slightly, since validation needs to check all elements)

3. **Performance improvement**:
   - **~50x faster** with `skip_validation=True` compared to previous implementation
   - Previous: ~25µs overhead (from `signature.bind()` on every call)
   - Current: ~0.5µs overhead (negligible)

### Recommendations

- **Development/Testing**: Use `skip_validation=False` (default) to catch data issues early
- **Production hot paths**: Use `skip_validation=True` for virtually zero overhead
- **One-time operations**: The overhead is negligible - validation is worth it
- **Tight loops**: Consider validating once before the loop, then use `skip_validation=True` inside

### Implementation Details

The optimization avoids `signature.bind()` when `skip_validation=True` by:
```python
def wrapper(*args, **kwargs):
    # (Simplified example of the fast path)
    skip = kwargs.get("skip_validation", False)

    if not skip:
        # Only bind args when we need to validate
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        # ... validate ...
        return func(*bound_args.args, **bound_args.kwargs)

    # Fast path: direct call
    return func(*args, **kwargs)
```

### Running the Benchmarks

```bash
# Run decorator overhead benchmark
uv run python benchmarks/benchmark_decorator.py

# Run data scaling benchmark
uv run python benchmarks/bench_scaling.py
```
