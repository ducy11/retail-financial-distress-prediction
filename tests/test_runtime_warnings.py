"""Verify that `runtime_warnings.py` silences only the known harmless warnings.

The pinned versions emit `OptimizeWarning: Unknown solver options: iprint` on every logistic fit, and
`unittest` reinstalls a `default` filter after importing a test module, which would let that warning flood
the output again. The suite checks idempotency and that unrelated warnings stay visible.
"""
from __future__ import annotations

import pathlib
import sys
import unittest
import warnings

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from runtime_warnings import (LIBRARY_WARNING_FILTERS, OptimizeWarning,  # noqa: E402
                              _filter_key, quiet_library_warnings)


def _known_filter_count() -> int:
    """Count how many copies of the known filters are present in `warnings.filters`."""
    known = {_filter_key("ignore", flt.get("message", ""), flt["category"], flt.get("module", ""))
             for flt in LIBRARY_WARNING_FILTERS}
    return sum(1 for item in warnings.filters
               if _filter_key(item[0], item[1], item[2], item[3]) in known)


def setUpModule() -> None:  # noqa: D103 - `unittest` hook
    """Reapply the harmless-warning filters after `unittest` installs `simplefilter("default")`."""
    quiet_library_warnings()


class TestQuietLibraryWarnings(unittest.TestCase):
    """Library warning filtering mechanism."""

    def test_filter_list_is_narrow_and_documented(self):
        """The filter list holds only known warnings: the iprint solver warning and library deprecations."""
        self.assertEqual(len(LIBRARY_WARNING_FILTERS), 3)
        messages = [item.get("message", "") for item in LIBRARY_WARNING_FILTERS]
        modules = [item.get("module", "") for item in LIBRARY_WARNING_FILTERS]
        self.assertIn("Unknown solver options: iprint", messages)
        self.assertTrue(any(module.startswith("matplotlib") for module in modules))
        self.assertTrue(any(module.startswith("pyparsing") for module in modules))

    def test_idempotent_and_placed_at_front(self):
        """Repeated calls, as each `setUpModule` makes, keep one copy per filter at the front."""
        for _ in range(3):
            quiet_library_warnings()
        self.assertEqual(_known_filter_count(), len(LIBRARY_WARNING_FILTERS))
        head = {_filter_key(item[0], item[1], item[2], item[3]) for item in warnings.filters[:3]}
        self.assertEqual(len(head), len(LIBRARY_WARNING_FILTERS))  # our filters sit at the front

    def test_iprint_warning_is_hidden_even_after_simplefilter_default(self):
        """The iprint warning stays filtered even after `unittest` installs `simplefilter("default")`.

        This regression occurred before: appending a filter only when absent let the `default` filter that
        `unittest` pushes to the front win, so `OptimizeWarning` printed in the middle of the test output.
        """
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")      # what `unittest` does before running a test
            quiet_library_warnings()              # what `setUpModule` does
            warnings.warn("Unknown solver options: iprint", OptimizeWarning, stacklevel=1)
        self.assertEqual([str(item.message) for item in caught], [])
        self.assertEqual(_known_filter_count(), len(LIBRARY_WARNING_FILTERS))

    def test_unknown_warning_still_visible(self):
        """The filter must stay narrow, so unrelated warnings still surface instead of hiding real bugs."""
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("default")
            quiet_library_warnings()
            warnings.warn("test warning outside the known list", UserWarning, stacklevel=1)
        self.assertEqual([str(item.message) for item in caught],
                         ["test warning outside the known list"])


if __name__ == "__main__":  # pragma: no cover
    unittest.main(verbosity=2)
