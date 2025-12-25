# Benchmark Results

## Decorator Performance

Benchmark measuring the overhead of the `@validated` decorator with 10,000 iterations:

### Results Summary (After Optimization)

| Scenario | validate=False Overhead | validate=True Overhead |
|----------|------------------------|----------------------|
| Small Series (100 elements) | **0.41µs/call** | 210.22µs/call |
| Large Series (10,000 elements) | **0.75µs/call** | 226.42µs/call |
| Multiple Validators (2) | **0.30µs/call** | 285.67µs/call |
| Index Validators (3) | **0.60µs/call** | 236.42µs/call |

### Key Findings

1. **`validate=False` overhead is essentially zero (~0.5µs per call)**, achieved by:
   - Checking `kwargs.get("validate", True)` directly
   - Only using `signature.bind()` when validation is actually needed

2. **`validate=True` overhead scales with**:
   - Number of validators (more validators = more time)
   - Validator complexity (Index validators do more work)
   - Data size (slightly, since validation needs to check all elements)

3. **Performance improvement**:
   - **~50x faster** with `validate=False` compared to previous implementation
   - Previous: ~25µs overhead (from `signature.bind()` on every call)
   - Current: ~0.5µs overhead (negligible)

### Recommendations

- **Development/Testing**: Use `validate=True` (default) to catch data issues early
- **Production hot paths**: Use `validate=False` for virtually zero overhead
- **One-time operations**: The overhead is negligible - validation is worth it
- **Tight loops**: Consider validating once before the loop, then use `validate=False` inside

### Implementation Details

The optimization avoids `signature.bind()` when `validate=False` by:
```python
def wrapper(*args, **kwargs):
    should_validate = kwargs.get("validate", True)
    
    if should_validate:
        # Only bind args when we need to validate
        sig = inspect.signature(func)
        bound_args = sig.bind(*args, **kwargs)
        # ... validate ...
        return func(*bound_args.args, **bound_args.kwargs)
    
    # Fast path: direct call
    return func(*args, **kwargs)
```

### Running the Benchmark

```bash
uv run python benchmarks/benchmark_decorator.py
```
