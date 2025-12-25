# validated

![CI](https://github.com/sencer/validated/actions/workflows/ci.yml/badge.svg)
[![codecov](https://codecov.io/gh/sencer/validated/branch/master/graph/badge.svg)](https://app.codecov.io/github/sencer/validated)

**Pandas validation using Annotated types and decorators**

`validated` is a lightweight Python library for validating pandas DataFrames and Series using Python's `Annotated` types and decorators. It provides a clean, type-safe way to express data validation constraints directly in function signatures.

## Features

- 🎯 **Type-safe validation** - Uses Python's `Annotated` types for inline constraints
- 🐼 **Pandas-focused** - Built specifically for pandas DataFrames and Series
- ⚡ **Decorator-based** - Simple `@validated` decorator for automatic validation
- 🔧 **Composable validators** - Chain multiple validators together
- 🎨 **Clean syntax** - Validation rules live in your type annotations
- 🚀 **Zero runtime overhead** - Optional validation can be disabled

## Installation

```bash
pip install validated
```

Or with uv:
```bash
uv add validated
```

## Quick Start

```python
import pandas as pd
from validated import validated, Validated, Finite, NonNaN, NonEmpty

@validated
def calculate_returns(
    prices: Validated[pd.Series, Finite, NonNaN, NonEmpty],
) -> pd.Series:
    """Calculate percentage returns from prices.
    
    Data is explicitly checked for:
    - Not empty (NonEmpty)
    - No NaN values (NonNaN)
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
- **`StrictFinite`** - Ensures no Inf AND no NaN values
- **`NonNaN`** - Ensures no NaN values (allows Inf)
- **`NonNegative`** - Ensures all values >= 0
- **`Positive`** - Ensures all values > 0
- **`NonEmpty`** - Ensures data is not empty
- **`Unique`** - Ensures all values are unique
- **`MonoUp`** - Ensures values are monotonically increasing
- **`MonoDown`** - Ensures values are monotonically decreasing
- **`Datetime`** - Ensures data is a DatetimeIndex
- **`OneOf["a", "b", "c"]`** - Ensures values are in allowed set (categorical)


### Shape Validators

- **`Shape[10, 5]`** - Exact shape (10 rows, 5 columns)
- **`Shape[Ge[10], Any]`** - At least 10 rows, any columns
- **`Shape[Any, Le[5]]`** - Any rows, at most 5 columns
- **`Shape[Gt[0], Lt[100]]`** - More than 0 rows, less than 100 columns
- **`Shape[100]`** - For Series: exactly 100 rows



### Index Wrapper

The `Index[]` wrapper allows you to apply any Series/Index validator to the index of a Series or DataFrame:

- **`Index[Datetime]`** - Ensures index is a DatetimeIndex
- **`Index[MonoUp]`** - Ensures index is monotonically increasing
- **`Index[Unique]`** - Ensures index values are unique
- **`Index[Datetime, MonoUp, Unique]`** - Combine multiple validators

### DataFrame Column Validators

- **`HasColumns["col1", "col2"]`** - Ensures specified columns exist
- **`Ge["high", "low"]`** - Ensures one column >= another column
- **`Le["low", "high"]`** - Ensures one column <= another column
- **`Gt["high", "low"]`** - Ensures one column > another column
- **`Lt["low", "high"]`** - Ensures one column < another column

### Column-Specific Validators

- **`HasColumn["col"]`** - Check that DataFrame has column
- **`HasColumn["col", Validator, ...]`** - Check column exists and apply Series validators

### Gap Validators (Time Series)

- **`NoTimeGaps`** - Ensures no gaps in datetime values/index
- **`MaxGap[timedelta]`** - Ensures maximum gap between datetime values
- **`MaxDiff[value]`** - Ensures maximum difference between consecutive values

## Examples

### Basic Series Validation

```python
from validated import validated, Validated, Positive, NonNaN
import numpy as np
import pandas as pd

@validated
def calculate_log_returns(
    prices: Validated[pd.Series, Positive, NonNaN],
) -> pd.Series:
    """Calculate log returns - prices must be positive and not NaN."""
    return np.log(prices / prices.shift(1))

prices = pd.Series([100.0, 102.0, 101.0, 103.0])
log_returns = calculate_log_returns(prices)
```

### DataFrame Column Validation

```python
from validated import validated, Validated, HasColumns, Ge, NonNaN
import pandas as pd

@validated
def calculate_true_range(
    data: Validated[pd.DataFrame, HasColumns["high", "low", "close"], Ge["high", "low"], NonNaN],
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
from validated import validated, Validated, Index, Datetime, MonoUp, Finite
import pandas as pd

@validated
def resample_ohlc(
    data: Validated[pd.DataFrame, Index[Datetime, MonoUp], Finite],
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
from validated import validated, Validated, Index, Unique
import pandas as pd

@validated
def process_unique_ids(
    data: Validated[pd.DataFrame, Index[Unique]],
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
from validated import validated, Validated, OneOf, HasColumn
import pandas as pd

@validated
def process_orders(
    data: Validated[pd.DataFrame, HasColumn["status", OneOf["pending", "shipped", "delivered"]]],
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

# Also works with Literal type syntax
@validated
def process_with_literal(
    data: Validated[pd.Series, OneOf[Literal["a", "b", "c"]]],
) -> pd.Series:
    return data
```

### Monotonic Value Validation

```python
from validated import validated, Validated, MonoUp, MonoDown
import pandas as pd

@validated
def calculate_cumulative_returns(
    prices: Validated[pd.Series, MonoUp],
) -> pd.Series:
    """Calculate cumulative returns - prices must be monotonically increasing."""
    return (prices / prices.iloc[0]) - 1

@validated
def track_drawdown(
    equity: Validated[pd.Series, MonoDown],
) -> pd.Series:
    """Track drawdown - equity must be monotonically decreasing."""
    return (equity / equity.iloc[0]) - 1
```

### Shape Validation

```python
from typing import Any
from validated import validated, Validated, Shape, Ge, Le
import pandas as pd

@validated
def process_batch(
    data: Validated[pd.DataFrame, Shape[Ge[10], Any]],
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
@validated
def process_matrix(
    data: Validated[pd.DataFrame, Shape[Ge[5], Le[10]]],
) -> pd.DataFrame:
    """Process matrix - 5+ rows, max 10 columns."""
    return data

# Exact shape for Series
@validated
def process_vector(
    data: Validated[pd.Series, Shape[100]],
) -> pd.Series:
    """Process vector - must have exactly 100 elements."""
    return data
```

### Column-Specific Validation with HasColumn

```python
from validated import validated, Validated, HasColumn, Finite, Positive, MonoUp
import pandas as pd

@validated
def process_trading_data(
    data: Validated[
        pd.DataFrame,
        HasColumn["price", Finite, Positive],
        HasColumn["volume", Finite, Positive],
        HasColumn["timestamp", MonoUp],
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
@validated
def simple_check(
    data: Validated[pd.DataFrame, HasColumn["price"], HasColumn["volume"]],
) -> float:
    """Just check columns exist."""
    return (data["price"] * data["volume"]).sum()
```

### Chaining Multiple Index Validators

```python
from validated import validated, Validated, Index, Datetime, MonoUp, Unique, Finite, Positive
import pandas as pd

@validated
def calculate_volume_profile(
    volume: Validated[pd.Series, Index[Datetime, MonoUp, Unique], Finite, Positive],
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
from validated import Validator, validated, Validated
import pandas as pd

class InRange(Validator):
    """Validator for values within a specific range."""

    def __init__(self, min_val: float, max_val: float):
        self.min_val = min_val
        self.max_val = max_val

    def validate(self, data):
        if isinstance(data, (pd.Series, pd.DataFrame)):
            if (data < self.min_val).any() or (data > self.max_val).any():
                raise ValueError(f"Data must be in range [{self.min_val}, {self.max_val}]")
        return data

@validated
def normalize_percentage(
    data: Validated[pd.Series, InRange(0, 100)],
) -> pd.Series:
    """Normalize percentage data to [0, 1] range."""
    return data / 100
```

## Performance & Optimization

`validated` is designed for high-performance data pipelines:

### Parallel Validation

When a function accepts multiple validated arguments, `validated` automatically validates them in parallel using a thread pool. This leverages the release of the GIL during pandas/numpy operations, providing significant speedups for large datasets.

```python
@validated
def process_large_data(
    source: Validated[pd.DataFrame, Finite, NonNaN],
    target: Validated[pd.DataFrame, Finite, NonNaN],
) -> pd.DataFrame:
    # Both 'source' and 'target' are validated concurrently
    return pd.merge(source, target, on="id")
```

### Zero-Overhead Production Mode

For maximum performance in production critical paths, you can disable validation globally or per-call:

- **Per-call:** `func(data, skip_validation=True)`
- **Defaults:** Use `@validated(skip_validation_by_default=True)` for functions that should only be validated during development or debugging.

### Cached Validator Compilation

Validation logic is pre-compiled at import time (when the decorator runs). The runtime overhead is minimal, consisting only of the necessary numpy/pandas checks.

## Type Checking

`validated` includes a `py.typed` marker for full type checker support. Your IDE and type checkers (mypy, pyright, basedpyright) will understand the validation annotations.

### How Type Checkers Handle `Validated`

According to PEP 593, `Annotated[T, metadata]` (which `Validated` is an alias for) is treated as **equivalent to `T`** for type checking purposes. This means:

```python
@validated
def process(data: Validated[pd.Series, Finite]) -> float:
    return data.sum()

# Type checkers understand that pd.Series is compatible with Validated[pd.Series, ...]
series = pd.Series([1, 2, 3])
result = process(series)  # ✓ Type checker is happy!
```

The validation metadata is:
- **Preserved at runtime** - Used by the `@validated` decorator for validation
- **Ignored by type checkers** - `Validated[pd.Series, Finite]` is treated as `pd.Series`

This gives you the best of both worlds: clean type checking and runtime validation.

## Opt-in Strictness

In Validated

Pandas validation using Annotated types and decorators.

## Features

- **Standard Python Typing**: Use `Annotated[pd.Series, Validator]` syntax
- **Decorator-based**: `@validated` handles validation automatically
- **Performance**: Validation can be disabled globally or per-call
- **Clean API**: No strict schema objects required

## Installation

```bash
pip install validated
```

## Usage

```python
from validated import validated, Validated
from validated.validators import Finite, Ge
import pandas as pd

@validated
def process_data(
    data: Validated[pd.Series, Finite, Ge(0)]
):
    return data.mean()
```
validation is opt-in. By default, arguments wrapped in `Validated[...]` are only checked for type compatibility (via other tools) unless validators are provided.

To enforce strict checks like "no NaNs" or "not empty", you must explicitly add the corresponding validators:

```python
from validated import validated, Validated, NonNaN, NonEmpty
import pandas as pd

@validated
def process_flexible(data: Validated[pd.Series, None]) -> float:
    """Accepts any Series (NaNs and empty allowed)."""
    if data.empty:
        return 0.0
    return data.sum()

@validated
def process_strict(data: Validated[pd.Series, NonNaN, NonEmpty]) -> float:
    """Rejects NaNs and empty data."""
    return data.sum()
```

## Comparison with Pandera

While [Pandera](https://pandera.readthedocs.io/) is excellent for comprehensive schema validation, `validated` offers a lighter-weight alternative focused on:

- **Inline validation** - Constraints live in function signatures
- **Decorator simplicity** - Single `@validated` decorator
- **Type annotation syntax** - Uses Python's native `Annotated` types
- **Minimal overhead** - Lightweight with no heavy dependencies

Use `validated` when you want simple, inline validation. Use Pandera when you need comprehensive schema management, complex validation logic, or data contracts.

## Performance

`validated` is designed to be lightweight with minimal overhead:

- Validation checks are only performed when `skip_validation=False` (default)
- No schema compilation or complex preprocessing
- Direct numpy/pandas operations for validation
- Optional validation can be disabled for production performance

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

MIT License - see LICENSE file for details.

## Why validated?

**Problem:** When building data analysis pipelines with pandas, you often need to validate:
- Data has no NaN or Inf values
- DataFrames have required columns
- Values are in expected ranges
- Indices are properly formatted

**Traditional approach:** Add manual validation checks at the start of each function.

**With validated:** Express validation constraints directly in type annotations using `Validated[Type, Validator, ...]` and get automatic validation with the `@validated` decorator.
