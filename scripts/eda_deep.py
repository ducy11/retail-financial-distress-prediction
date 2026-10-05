"""Deep exploratory analysis of features, labels, correlations and drift.

Delegates every computation to `forecasting/eda.py`, whose routines take plain arrays or dicts and are
independently testable, and fits no model. Writes `reports/results/eda_deep.{json,md}` plus seven
figures under `reports/figures/eda_deep/`; run with `python -m scripts.eda_deep`.
"""
from __future__ import annotations

import argparse
import sys

from forecasting.config import ensure_utf8_stdio
from forecasting.eda import run


def main(argv=None) -> int:
    """Run the deep analysis and, by default, write the artifacts under `reports/`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true",
                        help="Print only; do not write reports/results/eda_deep.*")
    parser.add_argument("--no-figures", action="store_true",
                        help="Skip figure rendering (faster when only the numbers matter)")
    args = parser.parse_args(argv)
    print("=== Deep EDA: features / labels / correlations / drift ===")
    run(write=not args.no_write, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
