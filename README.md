# datawarden

![CI](https://github.com/sencer/datawarden/actions/workflows/ci.yml/badge.svg)
![codecov](https://codecov.io/gh/sencer/datawarden/branch/master/graph/badge.svg)

**High-performance, JIT-accelerated data validation for Pandas.**

`datawarden` is a high-performance validation library that provides a clean, type-safe way to express data validation constraints directly in function signatures. It utilizes Python type hints to declare validation rules, which are then compiled into optimized machine code using **Numba JIT** for near-zero runtime overhead.

---

## 🚀 Why Datawarden?

*   🎯 **Type-Safe Declarations**: Use `Annotated` types (`Validated[T, ...]`) to define constraints directly in your function signatures.
*   ⚡ **Numba JIT Acceleration**: Complex logical chains are fused and compiled, achieving up to **75x speedups** over vectorized NumPy/Pandas for certain operations.
*   🧵 **Parallel Execution**: Automatically validates multiple function arguments in parallel using a thread pool.
*   📦 **Memory Efficient**: Supports chunked validation, allowing you to validate datasets larger than your RAM with O(1) memory overhead.
*   🔧 **N-ary Comparisons**: Compare multiple columns (e.g., `Ge('high', 'low', 'open')`) with zero-copy JIT execution.
*   🔄 **Cross-Chunk Continuity**: Built-in support for stateful sequence validation (e.g., monotonicity across streaming data chunks).

---

## 📦 Installation

```bash
pip install datawarden
```

Or with `uv`:
```bash
uv add datawarden
```

---

## 🛠️ Quick Start

```python
import pandas as pd
import numpy as np
from datawarden import validate, Validated, Gt, Finite, NotEmpty

@validate
def calculate_returns(
    prices: Validated[pd.Series, NotEmpty, Finite, Gt(0)],
    threshold: float = 0.01
) -> pd.Series:
    """
    prices is validated to be NotEmpty, have only Finite values, and all values > 0.
    threshold is a regular float.
    """
    return prices.pct_change()

# Valid data passes through
prices = pd.Series([100.0, 102.0, 101.0, 103.0])
returns = calculate_returns(prices)

# Invalid data raises ValidationError with a detailed report
bad_prices = pd.Series([100.0, np.nan, 102.0])
# Raises: ValidationError: Argument 'prices' failed validation:
#  - Finite (1/3 rows failed)
calculate_returns(bad_prices)
```

---

## 💎 Advanced Features

### 🔗 Logical Composition
Combine validators using standard Python logical operators. `datawarden` will fuse these into a single optimized pass.

```python
from datawarden import Ge, Le, IsNaN

# Value must be between 0 and 1, or can be NaN
UnitValue = Validated[pd.Series, (Ge(0) & Le(1)) | IsNaN()]
```

### 📊 N-ary Column Comparisons
Validate relationships across multiple columns in a DataFrame without manual iteration or heavy Pandas operations.

```python
from datawarden import Ge

# Validates that 'max' >= 'min' AND 'min' >= 'base' for all rows
@validate
def check_bounds(df: Validated[pd.DataFrame, Ge('max', 'min', 'base')]):
    ...
```

### 📈 Stateful Sequence Validation
Maintain validation state across data chunks – essential for streaming pipelines.

```python
from datawarden import MonoUp, NoTimeGaps, Index

# Ensure timestamps are increasing and have no gaps across all chunks
@validate
def ingest_stream(chunk: Validated[pd.DataFrame, Index(MonoUp(), NoTimeGaps("1min"))]):
    ...
```

---

## ⚡ Performance Benchmarks

`datawarden` is built for speed. By fusing operations and avoiding intermediate allocations, it significantly outperforms standard approaches on large datasets (~10M+ rows).

| Operation | Pandas/NumPy | Datawarden (JIT) | Improvement |
| :--- | :--- | :--- | :--- |
| `Ge(0) & Le(1)` | ~15ms | **~0.2ms** | **75x** |
| `MonoUp` (Monotonic) | ~24ms | **~8ms** | **3x** |
| Multi-column `Ge` | ~45ms | **~0.5ms** | **90x** |

> [!NOTE]
> Benchmarks performed on a modern CPU with 10M rows. Numba fusion provides the biggest gains for complex logical chains.

---

## 🛠️ Configuration

Fine-tune the behavior of `datawarden` using the `Overrides` context manager or global config.

```python
from datawarden import Overrides

# Process a massive dataset in chunks to save memory
with Overrides(chunk_size_rows=100_000, use_numba=True):
    my_heavy_function(massive_df)

# Disable validation for an entire module during import to avoid redundant checks
with Overrides(skip_validation=True):
    import sensitive_library_already_validated
```

> [!NOTE]
> `Overrides(skip_validation=True)` is particularly useful when importing a library that uses `datawarden` internally, but you've already validated the data upstream or want to disable validation for performance in a production environment.

| Option | Default | Description |
| :--- | :--- | :--- |
| `skip_validation` | `False` | Globally disable validation for production hot-loops. |
| `warn_only` | `False` | Log a warning instead of raising `ValidationError`. |
| `chunk_size_rows` | `None` | Automatically split large data into chunks for memory efficiency. |
| `use_numba` | `True` | Enable/Disable JIT compilation via Numba. |
| `parallel_threshold_rows` | `100,000` | Minimum row count to trigger parallel multi-argument validation. |

---

## 📖 Available Validators

### Structural
*   **`Index(*validators)`**: Apply validators to the data index.
*   **`Columns(names, *validators)`**: Apply validators to a set of columns.
*   **`Column(name, *validators)`**: Apply validators to a specific column.
*   **`Shape(rows, cols)`**: Validate container dimensions.
*   **`NotEmpty` / `Empty`**: Check for content existence.

### Numeric
*   **`Gt`, `Ge`, `Lt`, `Le`, `Eq`, `Ne`**: Standard comparisons (with multi-column support).
*   **`Finite`**: No `NaN` or `Inf`.
*   **`NotNaN` / `IsNaN`**: Null checks.
*   **`Positive` / `Negative`** / **`NonNegative`** / **`NonPositive`**: Sign checks.

### Sequence & Stateful
*   **`MonoUp` / `MonoDown`**: Monotonicity checks across chunks.
*   **`NoTimeGaps(freq)`**: Continuous time series check.
*   **`MaxGap(duration)`**: Maximum interval size check.
*   **`Unique`**: Ensure all values are unique.

### Value & Custom
*   **`Between(low, high)`** / **`Outside(low, high)`**: Range checks.
*   **`OneOf(*values)`**: Set membership.
*   **`Is(predicate)`**: Custom lambda/function element-wise check.
*   **`Rows(predicate)`**: Custom row-wise DataFrame check.

---

## 📜 License

MIT License. See [LICENSE](LICENSE) for more information.
