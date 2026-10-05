"""Filter known harmless library warnings so CLI and test logs stay clean.

The filters are centralized here because the warnings come from library version pairs, not from this
project's logic:
- scikit-learn 1.6 passes an `iprint` option to scipy 1.18, which dropped it, so every
  `LogisticRegression` fit prints `OptimizeWarning: Unknown solver options: iprint`.
- matplotlib 3.9 calls a deprecated pyparsing API, so drawing a figure prints
  `PyparsingDeprecationWarning`.
Both leave model parameters and metrics unchanged; results are identical with and without filtering.

`unittest`, when running `python -m unittest ...`, calls `warnings.simplefilter("default")` after test
modules are imported, so a filter set at import time is shadowed and the warning reappears. Test
modules therefore call `quiet_library_warnings()` again inside `setUpModule()`.

Only the exact known messages and owners are filtered; every other warning, including ones from this
project's own code, still shows. Remove this module once scikit-learn, scipy and matplotlib are
upgraded to a compatible set.
"""
from __future__ import annotations

import warnings

try:  # `OptimizeWarning` is a scipy warning
    from scipy.optimize import OptimizeWarning
except Exception:  # pragma: no cover - environment without scipy
    class OptimizeWarning(UserWarning):  # type: ignore[no-redef]
        """Fallback when scipy does not provide `OptimizeWarning`."""


#: Known harmless filters (kwargs matching `warnings.filterwarnings`).
LIBRARY_WARNING_FILTERS = (
    {"message": "Unknown solver options: iprint", "category": OptimizeWarning},
    {"category": DeprecationWarning, "module": r"matplotlib\..*"},
    {"category": DeprecationWarning, "module": r"pyparsing.*"},
)


def _pattern_text(value: object) -> str:
    """Pattern string of one filter (`warnings` stores a compiled regex or `None`)."""
    if value is None:
        return ""
    return getattr(value, "pattern", str(value))


def _filter_key(action: str, message: object, category: type, module: object) -> tuple:
    """Identity key of one filter (used to drop the old copy before re-inserting)."""
    return (action, _pattern_text(message), category, _pattern_text(module))


def quiet_library_warnings() -> None:
    """Move the harmless filters to the front of `warnings.filters`, keeping one copy each.

    Re-inserting at the front is required: `unittest` calls `warnings.simplefilter("default")` just
    before running tests, which pushes `default` to the front and shadows the older filter (still in
    the list), so the `iprint` warning prints again. This drops the stale copy and re-inserts it at
    the front, so repeated calls are safe and the list never grows.
    """
    for flt in LIBRARY_WARNING_FILTERS:
        key = _filter_key("ignore", flt.get("message", ""), flt["category"], flt.get("module", ""))
        warnings.filters[:] = [item for item in warnings.filters
                               if _filter_key(item[0], item[1], item[2], item[3]) != key]
        warnings.filterwarnings("ignore", **flt)
