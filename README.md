# datawarden

![CI](https://github.com/sencer/datawarden/actions/workflows/ci.yml/badge.svg)
![codecov](https://codecov.io/gh/sencer/datawarden/branch/master/graph/badge.svg)

**High-performance Pandas validation using Annotated types and decorators**

`datawarden` is a lightweight, high-performance Python library for validating
pandas DataFrames and Series using Python's `Annotated` types and decorators. It
provides a clean, type-safe way to express data validation constraints directly
in function signatures, accelerated by **Numba JIT** and **multi-threaded
execution**.

## Features

- 🎯 **Type-safe validation** - Uses Python's `Annotated` types for inline
  constraints
- 🐼 **Pandas-focused** - Built specifically for pandas DataFrames and Series
- ⚡ **Numba-Accelerated** - Complex logic is fused and compiled to machine code
  for near-zero overhead on large datasets
- 🧵 **Multi-threaded** - Automatically validates multiple arguments in parallel
- 📦 **Memory Efficient** - Supports chunked validation to keep memory usage low
- 🔧 **Composable** - Use `&`, `|`, and `~` operators to combine validators
- 🚀 **Zero runtime overhead** - Validation can be globally or locally disabled

## Installation

```bash
pip install datawarden
```

Or with uv:
```bash
uv add datawarden
```

## Quick Start

```python
import pandas as pd
import numpy as np
from datawarden import validate, Validated, Finite, NotEmpty

@validate
def calculate_returns(
    prices: Validated[pd.Series, Finite, NotEmpty],
) -> pd.Series:
    """Calculate percentage returns from prices.

    Data is explicitly checked for:
    - Not empty (NotEmpty)
    - Finite values (No NaN, No Inf)
    """
    return prices.pct_change()

# Valid data passes through
prices = pd.Series([100.0, 102.0, 101.0, 103.0])
returns = calculate_returns(prices)

# Invalid data raises ValidationError
bad_prices = pd.Series([100.0, np.inf, 101.0])
# Raises: ValidationError: Data must be finite
calculate_returns(bad_prices)
```

## Available Validators

### Value Validators (Series/Index)

- **`Finite`** - Ensures no Inf AND no NaN values (uses atomic `np.isfinite()`)
- **`NotNaN`** - Ensures no NaN values (allows Inf)
- **`IsNaN`** - Ensures values are NaN
- **`Infinite`** - Ensures values are infinite
- **`NonNegative`** - Ensures all values >= 0
- **`Positive`** - Ensures all values > 0
- **`Negative`** - Ensures all values < 0
- **`NonPositive`** - Ensures all values <= 0
- **`Between(lower, upper)`** - Ensures values in range [lower, upper]
- **`Outside(lower, upper)`** - Ensures values are outside the range
  [lower, upper]
- **`NotEmpty`** - Ensures data is not empty
- **`Unique`** - Ensures all values are unique
- **`MonoUp`** - Ensures values are monotonically increasing
- **`MonoDown`** - Ensures values are monotonically decreasing
- **`Datetime`** - Ensures data is a DatetimeIndex
- **`OneOf(*values)`** - Ensures values are in allowed set (categorical)
- **`Dtype(dtype)`** - Ensures data has specific dtype (e.g. `int64`, `float64`)

### Logical Composition

Validators can be combined using standard Python operators:

```python
from datawarden import Validated, Ge, Le, IsNaN, Or

# Allow positive values OR NaN
data: Validated[pd.Series, Ge(0) | IsNaN()]

# Values must be in [0, 10] or exactly 100, and not NaN
data: Validated[pd.Series, ((Ge(0) & Le(10)) | Eq(100)) & ~IsNaN()]
```

When using `|` or `&`, `datawarden` uses **Numba JIT** to fuse these checks into
a single pass over the data, avoiding intermediate boolean array allocations and
providing massive speedups.

### Mixed-type Handling

The `Finite` validator handles mixed-type data (e.g., DataFrames with both
numeric and string columns) automatically:

- **DataFrames**: Automatically selects and validates **only numeric columns**.
  Non-numeric columns are ignored.
- **Series/Index**: Requires a numeric dtype. Applying it to a string Series
  will raise a `TypeError`.

### Shape Validators

- **`Shape(rows=10, cols=5)`** - Exact shape
- **`Shape(rows=None, cols=5)`** - Only check columns
- **`Shape(100)`** - For Series: exactly 100 rows

### Index Wrapper

The `Index()` wrapper allows you to apply any Series/Index validator to the
index:

- **`Index(Datetime)`** - Ensures index is a DatetimeIndex
- **`Index(MonoUp)`** - Ensures index is monotonically increasing
- **`Index(Unique)`** - Ensures index values are unique

### DataFrame Column Validators

- **`Column("col", Validator, ...)`** - Apply validators to a specific column
- **`Columns(["a", "b"], Validator, ...)`** - Apply validators to multiple
  columns
- **`Ge("high", "low")`** - Ensures column "high" >= column "low" (also `Gt`,
  `Le`, `Lt`)

### Lambda Validators

- **`Is(predicate, name=None)`** - Element-wise predicate validation
- **`Rows(predicate, name=None)`** - Row-wise predicate validation for
  DataFrames

```python
# Row-wise: check each row satisfies condition
@validate
def process_ohlc(
    data: Validated[pd.DataFrame, Rows(lambda row: row["high"] >= row["low"])],
) -> pd.DataFrame:
    return data
```

### Gap & Sequence Validators

- **`NoTimeGaps(freq)`** - Ensures no gaps in datetime values (e.g., freq="1H")
- **`MaxGap(duration)`** - Ensures maximum gap between datetime values
- **`MaxDiff(value)`** - Ensures maximum absolute difference between consecutive
  values
- **`MinDiff(value)`** - Ensures minimum absolute difference between consecutive
  values

## Performance & Optimization

### Numba Acceleration

`datawarden` uses Numba to compile validation logic into highly optimized
machine code. This is particularly effective for complex logical chains where
multiple boolean operations are fused into a single loop, bypassing Python
overhead and temporary array allocations.

### Multi-threaded Execution

When a function accepts multiple validated arguments, they are validated in
parallel using a thread pool. This provides significant speedups for
multi-argument functions on large datasets.

### Configuration & Memory Efficiency

Use `Overrides` to temporarily change settings:

```python
from datawarden import Overrides

# Process a massive dataset in chunks to save memory
with Overrides(chunk_size_rows=100_000):
    process_huge_df(df)

# Disable validation globally for a hot loop
with Overrides(skip_validation=True):
    for _ in range(1000):
        hot_function(df)
```

| Option | Default | Description |
| --- | --- | --- |
| `skip_validation` | `False` | Skip all validation globally |
| `warn_only` | `False` | Warn instead of raising on validation failures |
| `chunk_size_rows` | `None` | Process data in chunks (O(1) memory) |
| `use_numba` | `True` | Enable Numba acceleration |
| `parallel_threshold_rows` | `100,000` | Min rows to trigger parallel validation |

## License

MIT License - see LICENSE file for details.
