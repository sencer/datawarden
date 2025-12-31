# Code Review Action Items

## Fixed
- [x] **[LOW] Dead code removal.** Removed unused `_reset_validators` function from `src/datawarden/decorator.py`.
- [x] **[LOW] Reorganize tests.** Moved tests from `tests/test_failing_cases.py` to appropriate files (`tests/validators/test_finite.py`, `tests/validators/test_nan_handling.py`, `tests/test_decorator.py`).
- [x] **[CRITICAL] Incorrect extraction of control flags.**
    - *Decision:* Documented that `skip_validation` and `warn_only` are reserved keywords.
- [x] **[HIGH] Fast path allows unexpected keyword arguments.**
    - Added check for unexpected kwargs in `decorator.py` fast path.
- [x] **[LOW] Validation order should follow signature order.**
    - Updated `validation_order` logic in `decorator.py`.
- [x] **[MEDIUM] Suboptimal validator fusion logic.**
    - Updated `src/datawarden/plan.py` to correctly identify vectorizable validators by checking for method overrides.
