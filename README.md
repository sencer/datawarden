# datawarden

![CI](https://github.com/sencer/datawarden/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/sencer/datawarden/branch/master/graph/badge.svg)](https://app.codecov.io/github/sencer/datawarden)

**Pandas validation using Annotated types and decorators**

`datawarden` is a lightweight Python library for validating pandas DataFrames and Series using Python's `Annotated` types and decorators. It provides a clean, type-safe way to express data validation constraints directly in function signatures.

## Features

- 🎯 **Type-safe validation** - Uses Python's `Annotated` types for inline constraints
- 🐼 **Pandas-focused** - Built specifically for pandas DataFrames and Series
- ⚡ **Decorator-based** - Simple `@validate` decorator for automatic validation
- 🔧 **Composable validators** - Chain multiple validators together
- 🎨 **Clean syntax** - Validation rules live in your type annotations
- 🚀 **Zero runtime overhead** - Optional validation can be disabled

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
from datawarden import validate, Validated, Finite, NotNaN, NotEmpty

@validate
def calculate_returns(
    prices: Validated[pd.Series, Finite, NotNaN, NotEmpty],
) -> pd.Series:
    """Calculate percentage returns from prices.

    Data is explicitly checked for:
    - Not empty (NotEmpty)
    - No NaN values (NotNaN)
    - No infinite values (Finite)
    """
    return prices.pct_change()

# Valid data passes through
prices = pd.Series([100.0, 102.0, 101.0, 103.0])
returns = calculate_returns(prices)

# Invalid data raises ValueError
import numpy as np
bad_prices = pd.Series([100.0, np.inf, 101.0])
# Raises: ValueError: Data must be finite (contains Inf)
calculate_returns(bad_prices)
```

## Available Validators

### Value Validators (Series/Index)

- **`Finite`** - Ensures no Inf values (allows NaN)
- **`StrictFinite`** - Ensures no Inf AND no NaN values (uses atomic `np.isfinite()`)
- **`NotNaN`** - Ensures no NaN values (allows Inf)
- **`NonNegative`** - Ensures all values >= 0
- **`Positive`** - Ensures all values > 0
- **`Negative`** - Ensures all values < 0
- **`NonPositive`** - Ensures all values <= 0
- **`Between(lower, upper)`** - Ensures values in range [lower, upper]
- **`NotEmpty`** - Ensures data is not empty
- **`Unique`** - Ensures all values are unique
- **`MonoUp`** - Ensures values are monotonically increasing
- **`MonoDown`** - Ensures values are monotonically decreasing
- **`Datetime`** - Ensures data is a DatetimeIndex
- **`OneOf("a", "b", "c")`** - Ensures values are in allowed set (categorical)
- **`IsDtype(dtype)`** - Ensures data has specific dtype (e.g. `int64`, `float`)

### Mixed-type Handling

The `Finite` and `StrictFinite` validators handle mixed-type data (e.g., DataFrames with both numeric and string columns) automatically:

- **DataFrames**: They automatically select and validate **only numeric columns**. Non-numeric columns (strings, objects, datetimes) are silently ignored. This allows applying these validators to entire DataFrames conveniently.
- **Series/Index**: They require a numeric dtype. Applying them to a string Series will raise a `TypeError`.

```python
# DataFrame: Passes (ignores string column 'label')
df = pd.DataFrame({"value": [1.0, 2.0], "label": ["a", "b"]})
Finite().validate(df)

# Series: Raises TypeError (requires numeric data)
s = pd.Series(["a", "b"])
Finite().validate(s) # Raises TypeError
```

### NaN-Tolerant Validation with IgnoringNaNs

The `IgnoringNaNs` wrapper allows validators to skip NaN values during validation. It can be used in two ways:

**1. Explicit wrapping** - wrap specific validators:

```python
from datawarden import validate, Validated, IgnoringNaNs, Ge, Lt
import pandas as pd

@validate
def process(
    data: Validated[pd.Series, IgnoringNaNs(Ge(0)), Lt(10)],
) -> pd.Series:
    # Ge(0) ignores NaNs: values >= 0 OR NaN are valid
    # Lt(10) still rejects NaNs (default behavior)
    return data
```

**2. Marker mode** - apply to all validators with `IgnoringNaNs()`:

```python
@validate
def process(
    data: Validated[pd.Series, Ge(0), Lt(100), IgnoringNaNs()],
) -> pd.Series:
    # Equivalent to: IgnoringNaNs(Ge(0)), IgnoringNaNs(Lt(100))
    # All validators now ignore NaN values
    return data

# NaN values pass through, non-NaN values are validated
import numpy as np
data = pd.Series([10.0, np.nan, 50.0, np.nan, 90.0])
result = process(data)  # Works! NaNs are ignored
```

Works with: `Ge`, `Le`, `Gt`, `Lt`, `Positive`, `NonNegative`, `Finite`, and any value validator.

### Constraint Relaxation (Local Overrides)

Sometimes you want a global constraint (like `NotNaN` or `StrictFinite`) but need to allow exceptions for specific columns. Use `AllowNaN` or `AllowInf` to override global constraints locally.

```python
from datawarden import validate, Validated, NotNaN, HasColumn, AllowNaN
import pandas as pd

@validate
def process_data(
    # Global NotNaN applies to all columns by default
    df: Validated[pd.DataFrame, NotNaN, HasColumn("optional_col", AllowNaN)],
) -> pd.DataFrame:
    """Process data where 'optional_col' is allowed to have NaNs."""
    return df

df = pd.DataFrame({
    "required": [1.0, 2.0],        # Must not have NaNs (Global NotNaN)
    "optional_col": [1.0, float("nan")] # NaNs allowed here (Local AllowNaN)
})
process_data(df) # Passes!
```

### Shape Validators

- **`Shape(10, 5)`** - Exact shape (10 rows, 5 columns)
- **`Shape(Ge(10), Any)`** - At least 10 rows, any columns
- **`Shape(Any, Le(5))`** - Any rows, at most 5 columns
- **`Shape(Gt(0), Lt(100))`** - More than 0 rows, less than 100 columns
- **`Shape(100)`** - For Series: exactly 100 rows



### Index Wrapper

The `Index()` wrapper allows you to apply any Series/Index validator to the index of a Series or DataFrame:

- **`Index(Datetime)`** - Ensures index is a DatetimeIndex
- **`Index(MonoUp)`** - Ensures index is monotonically increasing
- **`Index(Unique)`** - Ensures index values are unique
- **`Index(Datetime, MonoUp, Unique)`** - Combine multiple validators

### DataFrame Column Validators

- **`HasColumns("col1", "col2")`** - Ensures specified columns exist
- **`Ge("high", "low")`** - Ensures one column >= another column
- **`Le("low", "high")`** - Ensures one column <= another column
- **`Gt("high", "low")`** - Ensures one column > another column
- **`Lt("low", "high")`** - Ensures one column < another column

### Column-Specific Validators

- **`HasColumn("col")`** - Check that DataFrame has column
- **`HasColumn("col", Validator, ...)`** - Check column exists and apply Series validators

### Lambda Validators

- **`Is(predicate, name=None)`** - Element-wise predicate validation
- **`Rows(predicate, name=None)`** - Row-wise predicate validation for DataFrames

```python
from datawarden import validate, Validated, Is, Rows, HasColumn
import pandas as pd

# Element-wise: check all values satisfy condition
@validate
def process_values(
    data: Validated[pd.Series, Is(lambda x: (x >= 0) & (x <= 100))],
) -> pd.Series:
    return data

# Column-specific with Is
@validate
def process_roots(
    data: Validated[pd.DataFrame, HasColumn("root", Is(lambda x: x**2 < 2))],
) -> pd.DataFrame:
    return data

# Row-wise: check each row satisfies condition
@validate
def process_ohlc(
    data: Validated[pd.DataFrame, Rows(lambda row: row["high"] >= row["low"])],
) -> pd.DataFrame:
    return data

# With descriptive error name
@validate
def process_budget(
    data: Validated[pd.DataFrame, Rows(lambda row: row.sum() < 100, name="row sum must be < 100")],
) -> pd.DataFrame:
    return data
```

### Gap Validators (Time Series)

- **`NoTimeGaps`** - Ensures no gaps in datetime values/index
- **`MaxGap(timedelta)`** - Ensures maximum gap between datetime values
- **`MaxDiff(value)`** - Ensures maximum difference between consecutive values.
  By default, rejects NaNs. When used with `IgnoringNaNs`, it validates jumps
  between the remaining values after dropping NaNs.

## Examples

### Basic Series Validation

```python
from datawarden import validate, Validated, Positive, NotNaN
import numpy as np
import pandas as pd

@validate
def calculate_log_returns(
    prices: Validated[pd.Series, Positive, NotNaN],
) -> pd.Series:
    """Calculate log returns - prices must be positive and not NaN."""
    return np.log(prices / prices.shift(1))

prices = pd.Series([100.0, 102.0, 101.0, 103.0])
log_returns = calculate_log_returns(prices)
```

### DataFrame Column Validation

```python
from datawarden import validate, Validated, HasColumns, Ge, NotNaN
import pandas as pd

@validate
def calculate_true_range(
    data: Validated[pd.DataFrame, HasColumns("high", "low", "close"), Ge("high", "low"), NotNaN],
) -> pd.Series:
    """Calculate True Range - requires OHLC data."""
    hl = data["high"] - data["low"]
    hc = abs(data["high"] - data["close"].shift(1))
    lc = abs(data["low"] - data["close"].shift(1))
    return pd.concat([hl, hc, lc], axis=1).max(axis=1)

# Valid OHLC data
ohlc = pd.DataFrame({
    "high": [102, 105, 104],
    "low": [100, 103, 101],
    "close": [101, 104, 102]
})
tr = calculate_true_range(ohlc)

# Missing column raises error
bad_data = pd.DataFrame({"high": [102], "close": [101]})
# Raises: ValueError: Missing columns: ['low']
calculate_true_range(bad_data)
```

### Time Series Validation with Index

```python
from datawarden import validate, Validated, Index, Datetime, MonoUp, Finite
import pandas as pd

@validate
def resample_ohlc(
    data: Validated[pd.DataFrame, Index(Datetime, MonoUp), Finite],
    freq: str = "1D",
) -> pd.DataFrame:
    """Resample OHLC data to different frequency."""
    return data.resample(freq).agg({
        "open": "first",
        "high": "max",
        "low": "min",
        "close": "last"
    })

# Valid time series
dates = pd.date_range("2024-01-01", periods=10, freq="1h")
data = pd.DataFrame({
    "open": range(100, 110),
    "high": range(101, 111),
    "low": range(99, 109),
    "close": range(100, 110)
}, index=dates)
daily = resample_ohlc(data)

# Non-datetime index raises error
bad_data = data.copy()
bad_data.index = range(len(bad_data))
# Raises: ValueError: Index must be DatetimeIndex
resample_ohlc(bad_data)
```

### Unique Values Validation

```python
from datawarden import validate, Validated, Index, Unique
import pandas as pd

@validate
def process_unique_ids(
    data: Validated[pd.DataFrame, Index(Unique)],
) -> pd.DataFrame:
    """Process data with unique index values."""
    return data.sort_index()

# Valid unique index
df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "y", "z"])
result = process_unique_ids(df)

# Duplicate index values raise error
bad_df = pd.DataFrame({"a": [1, 2, 3]}, index=["x", "x", "z"])
# Raises: ValueError: Values must be unique
process_unique_ids(bad_df)
```

### Categorical Values Validation

```python
from typing import Literal
from datawarden import validate, Validated, OneOf, HasColumn
import pandas as pd

@validate
def process_orders(
    data: Validated[pd.DataFrame, HasColumn("status", OneOf("pending", "shipped", "delivered"))],
) -> pd.DataFrame:
    """Process orders with validated status column."""
    return data[data["status"] != "pending"]

# Valid data
orders = pd.DataFrame({
    "order_id": [1, 2, 3],
    "status": ["pending", "shipped", "delivered"]
})
result = process_orders(orders)

# Invalid status raises error
bad_orders = pd.DataFrame({
    "order_id": [1, 2],
    "status": ["pending", "cancelled"]  # "cancelled" not in allowed values
})
# Raises: ValueError: Values must be one of {'pending', 'shipped', 'delivered'}, got invalid: {'cancelled'}
process_orders(bad_orders)
```

### Monotonic Value Validation

```python
from datawarden import validate, Validated, MonoUp, MonoDown
import pandas as pd

@validate
def calculate_cumulative_returns(
    prices: Validated[pd.Series, MonoUp],
) -> pd.Series:
    """Calculate cumulative returns - prices must be monotonically increasing."""
    return (prices / prices.iloc[0]) - 1

@validate
def track_drawdown(
    equity: Validated[pd.Series, MonoDown],
) -> pd.Series:
    """Track drawdown - equity must be monotonically decreasing."""
    return (equity / equity.iloc[0]) - 1
```

### Sign Validators (Positive/Negative)

Complete set of sign validators for numerical constraints:

```python
from datawarden import validate, Validated, Positive, NonNegative, Negative, NonPositive
import pandas as pd

# Positive: values > 0
@validate
def calculate_log(data: Validated[pd.Series, Positive]) -> pd.Series:
    """Log requires strictly positive values."""
    import numpy as np
    return np.log(data)

# NonNegative: values >= 0 (zero allowed)
@validate
def calculate_sqrt(data: Validated[pd.Series, NonNegative]) -> pd.Series:
    """Square root allows zero."""
    import numpy as np
    return np.sqrt(data)

# Negative: values < 0
@validate
def process_losses(data: Validated[pd.Series, Negative]) -> pd.Series:
    """Process losses (must be negative)."""
    return data.abs()

# NonPositive: values <= 0 (zero allowed)
@validate
def process_debits(data: Validated[pd.Series, NonPositive]) -> pd.Series:
    """Process debits (must be zero or negative)."""
    return data
```

### Shape Validation

```python
from typing import Any
from datawarden import validate, Validated, Shape, Ge, Le
import pandas as pd

@validate
def process_batch(
    data: Validated[pd.DataFrame, Shape(Ge(10), Any)],
) -> pd.DataFrame:
    """Process data batch - must have at least 10 rows."""
    return data.describe()

# Valid data (10+ rows)
df = pd.DataFrame({"a": range(20), "b": range(20)})
result = process_batch(df)

# Too few rows raises error
small_df = pd.DataFrame({"a": [1, 2, 3]})
# Raises: ValueError: DataFrame must have >= 10 rows, got 3
process_batch(small_df)

# Constrain both dimensions
@validate
def process_matrix(
    data: Validated[pd.DataFrame, Shape(Ge(5), Le(10))],
) -> pd.DataFrame:
    """Process matrix - 5+ rows, max 10 columns."""
    return data

# Exact shape for Series
@validate
def process_vector(
    data: Validated[pd.Series, Shape(100)],
) -> pd.Series:
    """Process vector - must have exactly 100 elements."""
    return data
```

### Between Range Validation

```python
from datawarden import validate, Validated, Between
import pandas as pd

@validate
def normalize_percentage(
    data: Validated[pd.Series, Between(0, 100)],
) -> pd.Series:
    """Normalize percentage data to [0, 1] range."""
    return data / 100

# Valid data
pct = pd.Series([0, 25, 50, 75, 100])
result = normalize_percentage(pct)

# Out of range raises error
bad_pct = pd.Series([0, 50, 150])  # 150 > 100
# Raises: ValueError: Data must be <= 100

# Exclusive bounds with inclusive=(False, True)
@validate
def process_probability(
    data: Validated[pd.Series, Between(0, 1, inclusive=(False, True))],
) -> pd.Series:
    """Process probabilities in range (0, 1]."""
    return data

# 0 is excluded, 1 is included
process_probability(pd.Series([0.5, 1.0]))  # OK
# process_probability(pd.Series([0.0, 0.5]))  # Fails: 0 not > 0
```

### Column-Specific Validation with HasColumn

```python
from datawarden import validate, Validated, HasColumn, Finite, Positive, MonoUp
import pandas as pd

@validate
def process_trading_data(
    data: Validated[
        pd.DataFrame,
        HasColumn("price", Finite, Positive),
        HasColumn("volume", Finite, Positive),
        HasColumn("timestamp", MonoUp),
    ],
) -> pd.DataFrame:
    """Process trading data with column-specific validation.

    - price: must exist, be finite and positive
    - volume: must exist, be finite and positive
    - timestamp: must exist and be monotonically increasing
    """
    return data.assign(
        notional=data["price"] * data["volume"]
    )

# Or just check column presence:
@validate
def simple_check(
    data: Validated[pd.DataFrame, HasColumn("price"), HasColumn("volume")],
) -> float:
    """Just check columns exist."""
    return (data["price"] * data["volume"]).sum()
```

### State Independence

When using stateful validators (like `MonoUp`, `NoTimeGaps`, or `MaxDiff`) across multiple columns via `HasColumns`, `datawarden` ensures that each column maintains its own independent state. This is handled automatically by cloning validator instances for each column, preventing state leakage and ensuring reliable validation.

```python
# 'a' and 'b' will each have their own independent MonoUp state
@validate
def process(df: Validated[pd.DataFrame, HasColumns(["a", "b"], MonoUp)]):
    ...
```

### Chaining Multiple Index Validators

```python
from datawarden import validate, Validated, Index, Datetime, MonoUp, Unique, Finite, Positive
import pandas as pd

@validate
def calculate_volume_profile(
    volume: Validated[pd.Series, Index(Datetime, MonoUp, Unique), Finite, Positive],
) -> pd.Series:
    """Calculate volume profile - must be datetime-indexed, monotonic, unique, finite, positive."""
    return volume.groupby(volume.index.hour).sum()
```

### Optional Validation

Use `skip_validation` to disable validation for performance:

```python
# Validation enabled (default)
result = calculate_returns(prices)

# Validation disabled for performance
result = calculate_returns(prices, skip_validation=True)
```

### Custom Validators

Create your own validators by subclassing `Validator`:

```python
from datawarden import Validator, validate, Validated
import pandas as pd

class InRange(Validator):
    """Validator for values within a specific range."""

    # Priority 20: More complex than simple vector checks (10), but faster than Python loops (100)
    priority = 20

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    @property
    def is_chunkable(self) -> bool:
        """Can validate chunks independently."""
        return True

    def validate(self, data):
        if not isinstance(data, (pd.Series, pd.DataFrame)):
            raise TypeError("InRange validator requires pandas Series or DataFrame")

        # Use numpy for performance
        vals = data.values
        if (vals < self.min_val).any() or (vals > self.max_val).any():
            raise ValueError(f"Data must be in range [{self.min_val}, {self.max_val}]")

@validate
def normalize_percentage(
    data: Validated[pd.Series, InRange(0, 100)],
) -> pd.Series:
    """Normalize percentage data to [0, 1] range."""
    return data / 100
```

**Custom Validator Properties:**
- **`priority` (int):** Controls execution order (default 50). Use lower values (0-20) for fast checks, 100 for slow checks.
- **`is_chunkable` (bool):** Set to `False` if your validator requires the entire dataset at once (e.g., uniqueness checks). Default is `True`.

## Performance & Optimization

`datawarden` is designed for high-performance data pipelines:

### Parallel Validation

When a function accepts multiple validated arguments, `datawarden` automatically validates them in parallel using a thread pool. This leverages the release of the GIL during pandas/numpy operations, providing significant speedups for large datasets.

```python
@validate
def process_large_data(
    source: Validated[pd.DataFrame, Finite, NotNaN],
    target: Validated[pd.DataFrame, Finite, NotNaN],
) -> pd.DataFrame:
    # Both 'source' and 'target' are validated concurrently
    return pd.merge(source, target, on="id")
```

### Zero-Overhead Production Mode

For maximum performance in production critical paths, you can disable validation globally or per-call:

- **Per-call:** `func(data, skip_validation=True)`
- **Defaults:** Use `@validate(skip_validation_by_default=True)` for functions that should only be validated during development or debugging.

### Cached Validator Compilation

Validation logic is pre-compiled at import time (when the decorator runs). The runtime overhead is minimal, consisting only of the necessary numpy/pandas checks.

### Validator Execution Order (Fast-Fail)
To optimize performance, `datawarden` sorts validators by priority to execute cheap checks before expensive ones. This ensures immediate failure for obvious issues (like wrong shape) before attempting costly scans.

**Execution Phases:**
1. **Structural Checks (Priority 0):** `Shape`, `HasColumns`, `IsDtype`, `NotEmpty`. (O(1))
2. **Fast Holistic Checks:** Fast global checks.
3. **Column-wise Validation:** All column-specific validators.
4. **Slow Holistic Checks (Priority 100):** Comparison of all rows, `Rows(lambda)`, etc.

**Interleaved Execution:**
For DataFrames, validation is interleaved. Fast global checks run first, then column checks, and only if all pass, slow global checks run.

## Configuration & Memory Efficiency

### Global Configuration

Access the global configuration via `datawarden.config`:

```python
from datawarden.config import get_config, overrides, reset_config

# View current settings
config = get_config()
print(config.skip_validation)     # False
print(config.warn_only)           # False
print(config.chunk_size_rows)     # None
print(config.parallel_threshold_rows)  # 50000
print(config.max_workers)         # 4
```

**Available Configuration Options:**

| Option | Type | Default | Description |
|--------|------|---------|-------------|
| `skip_validation` | `bool` | `False` | Skip all validation globally |
| `warn_only` | `bool` | `False` | Warn instead of raising on validation failures |
| `chunk_size_rows` | `int \| None` | `None` | Process data in chunks of N rows (memory optimization) |
| `parallel_threshold_rows` | `int` | `50000` | Min rows to trigger parallel validation |
| `max_workers` | `int` | `4` | Max threads for parallel validation |

### Config Overrides (Context Manager)

Temporarily change settings using `overrides()`:

```python
from datawarden.config import overrides
from datawarden import validate, Validated, Finite
import pandas as pd

@validate
def heavy_process(data: Validated[pd.DataFrame, Finite]):
    ...

# Skip validation for this block
with overrides(skip_validation=True):
    heavy_process(large_df)

# Warn instead of raise (useful for debugging)
with overrides(warn_only=True):
    heavy_process(dirty_data)

# Combine overrides for debugging large datasets
with overrides(warn_only=True, chunk_size_rows=50_000):
    # Validates in chunks, warns on failures instead of raising
    heavy_process(huge_df)

### Memory-Efficient Chunking

For very large DataFrames that fit in memory but whose validation might spike memory usage (e.g., creating large boolean masks), you can enable **Chunked Validation**.

When `chunk_size_rows` is set, `datawarden` splits the DataFrame/Series into smaller chunks and validates them sequentially.

```python
# Process in chunks of 100,000 rows
with overrides(chunk_size_rows=100_000):
    heavy_process(huge_df)
```

**Supported Validators:**
- **Stateless Validators:** `Ge`, `Le`, `NotNaN`, `Finite`, `Is`, `Rows`, etc. (Validated per chunk independently).
- **Stateful Validators:** `MonoUp`, `MonoDown`, `NoTimeGaps`, `MaxGap`, `MaxDiff`. (State is preserved across chunks to ensure global correctness).
- **Index Validators:** `Index(MonoUp)`, etc.

*Note: Global property validators like `Unique` or `Shape` (exact rows) are **NOT** chunkable and will still require the full dataset to be validated at once. The library automatically handles this distinction.*

## Type Checking

`datawarden` includes a `py.typed` marker for full type checker support. Your IDE and type checkers (mypy, pyright, basedpyright) will understand the validation annotations.

### How Type Checkers Handle `Validated`

According to PEP 593, `Annotated[T, metadata]` (which `Validated` is an alias for) is treated as **equivalent to `T`** for type checking purposes. This means:

```python
@validate
def process(data: Validated[pd.Series, Finite]) -> float:
    return data.sum()

# Type checkers understand that pd.Series is compatible with Validated[pd.Series, ...]
series = pd.Series([1, 2, 3])
result = process(series)  # ✓ Type checker is happy!
```

The validation metadata is:
- **Preserved at runtime** - Used by the `@validate` decorator for validation
- **Ignored by type checkers** - `Validated[pd.Series, Finite]` is treated as `pd.Series`

This gives you the best of both worlds: clean type checking and runtime validation.

## Opt-in Strictness

Validation in `datawarden` is opt-in. By default, arguments wrapped in `Validated[...]` are only checked for type compatibility (via other tools) unless validators are provided.

To enforce strict checks like "no NaNs" or "not empty", you must explicitly add the corresponding validators:

```python
from datawarden import validate, Validated, NotNaN, NotEmpty
import pandas as pd

@validate
def process_flexible(data: Validated[pd.Series, None]) -> float:
    """Accepts any Series (NaNs and empty allowed)."""
    if data.empty:
        return 0.0
    return data.sum()

@validate
def process_strict(data: Validated[pd.Series, NotNaN, NotEmpty]) -> float:
    """Rejects NaNs and empty data."""
    return data.sum()
```

## Comparison with Pandera

While [Pandera](https://pandera.readthedocs.io/) is excellent for comprehensive schema validation, `datawarden` offers a lighter-weight alternative focused on:

- **Inline validation** - Constraints live in function signatures
- **Decorator simplicity** - Single `@validate` decorator
- **Type annotation syntax** - Uses Python's native `Annotated` types
- **Minimal overhead** - Lightweight with no heavy dependencies

Use `datawarden` when you want simple, inline validation. Use Pandera when you need comprehensive schema management, complex validation logic, or data contracts.

## Performance

`datawarden` is designed to be lightweight with minimal overhead:

- Validation checks are only performed when `skip_validation=False` (default)
- No schema compilation or complex preprocessing
- Direct numpy/pandas operations for validation
- Optional validation can be disabled for production performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Why datawarden?

**Problem:** When building data analysis pipelines with pandas, you often need to validate:
- Data has no NaN or Inf values
- DataFrames have required columns
- Values are in expected ranges
- Indices are properly formatted

**Traditional approach:** Add manual validation checks at the start of each function.

**With datawarden:** Express validation constraints directly in type annotations using `Validated[Type, Validator, ...]` and get automatic validation with the `@validate` decorator.
