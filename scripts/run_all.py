"""Run the full pipeline in order and log the outcome, reproducing every artifact in one command.

Executes the stages listed in `STEPS` sequentially and captures each stage's stdout into
`reports/results/run_all.log`, continuing past failures and treating `train` and `evaluate` as core.
Run with `python -m scripts.run_all [--skip tune,analysis] [--only train,evaluate]`.
"""
from __future__ import annotations

import argparse
import importlib
import io
import sys
import traceback
import warnings
from contextlib import redirect_stderr, redirect_stdout
from typing import Any, Dict, List, Tuple

from forecasting.config import RESULTS_DIR, ensure_dirs, ensure_utf8_stdio

#: (label, module, callable, kwargs, is_core_step)
STEPS: List[Tuple[str, str, str, Dict[str, Any], bool]] = [
    ("data (rebuild the split from retail-expanded)", "forecasting.data", "run", {"force": False}, False),
    ("prepare_sec (ported ETL: rebuild the 16 indicators from the SEC snapshot and back-check)",
     "scripts.prepare_sec", "run", {}, False),
    ("provenance (verify the data against the SEC snapshot)",
     "scripts.verify_provenance", "run", {}, False),
    ("eda (EDA figures and overview tables)", "scripts.eda", "run", {}, False),
    ("eda_deep (features, labels, correlations and drift)", "scripts.eda_deep", "run", {}, False),
    ("train (four model families, selected by cross-company AP)", "forecasting.train", "run", {}, True),
    ("baselines (dummy / ticker prior / single feature / Altman rule)", "forecasting.baselines", "run", {}, False),
    ("validation (GroupKFold / LOCO / sample and cluster bootstrap / walk-forward)",
     "forecasting.validation", "run", {}, False),
    ("tuning (GridSearchCV split by company)", "forecasting.tuning", "run", {}, False),
    ("evaluate (final test scoring, run once)", "forecasting.evaluate", "run", {}, True),
    ("report (per-sample prediction table and histogram)", "forecasting.report", "run", {}, False),
    ("analyze (importance / ablation / errors / thresholds)", "scripts.analyze", "run", {}, False),
    ("prep_exp (winsorize x scaler on the real data)",
     "scripts.experiment_preprocessing", "run", {}, False),
    ("imbalance_real (class-imbalance techniques on the real data)",
     "scripts.experiment_imbalance_real", "run", {}, False),
    ("search (random search plus the runs.csv ledger)", "scripts.search", "run", {}, False),
    ("explain (SHAP/KernelSHAP and explanation figures)", "scripts.explain_model", "run", {}, False),
    ("significance (DeLong + paired bootstrap)", "scripts.significance", "run", {}, False),
    ("label_sensitivity (sensitivity across four label definitions)",
     "scripts.label_sensitivity", "run", {}, False),
    ("relabel (reproducible rule labels and comparison)", "scripts.relabel", "run", {}, False),
    ("predict (demo: score one sample with the frozen model)", "scripts.predict", "run",
     {"sample_id": "HD-2024Q2"}, False),
    ("make_report (docs/BAO-CAO.md + slide.md)", "scripts.make_report", "run", {}, False),
    ("export_office (Word .docx + Slide .pptx)", "scripts.export_office", "run", {}, False),
]


def _call(module_name: str, func_name: str, kwargs: Dict[str, Any]) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, func_name)(**kwargs)


def run(skip: List[str] | None = None, only: List[str] | None = None) -> Dict[str, Any]:
    """Run the pipeline and return a mapping of step key to status."""
    ensure_dirs()
    skip = skip or []
    only = only or []
    log_path = RESULTS_DIR / "run_all.log"
    lines: List[str] = []
    status: Dict[str, Any] = {}

    for label, module_name, func_name, kwargs, core in STEPS:
        key = module_name.split(".")[-1]
        if only and key not in only:
            continue
        if key in skip:
            lines.append(f"[SKIP]   {label}")
            status[key] = "skipped"
            continue
        buffer = io.StringIO()
        try:
            with warnings.catch_warnings(), redirect_stdout(buffer), redirect_stderr(buffer):
                warnings.simplefilter("ignore")  # keep the log clean; technical warnings are not results
                result = _call(module_name, func_name, kwargs)
            status[key] = "ok"
            lines.append(f"[OK]     {label}")
        except (ModuleNotFoundError, AttributeError) as e:
            status[key] = f"missing ({type(e).__name__})"
            lines.append(f"[MISSING] {label} — {e}")
        except Exception as e:  # noqa: BLE001 - capture the full traceback for debugging
            status[key] = f"failed: {type(e).__name__}"
            lines.append(f"[ERROR]  {label} — {type(e).__name__}: {e}")
            lines.append(traceback.format_exc())
        body = buffer.getvalue().strip()
        if body:
            lines.append("    " + body.replace("\n", "\n    "))
        _ = result

    header = "=== run_all: step status ==="
    text = "\n".join([header] + [f"  {k}: {v}" for k, v in status.items()] + ["", *lines])
    log_path.write_text(text + "\n", encoding="utf-8")
    print(text if len(text) < 6000 else text[:6000] + "\n... (see the full log in run_all.log)")
    return status


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", default="", help="Steps to skip, comma-separated.")
    parser.add_argument("--only", default="", help="Run only these steps, comma-separated.")
    args = parser.parse_args(argv)
    status = run(skip=[s for s in args.skip.split(",") if s],
                 only=[s for s in args.only.split(",") if s])
    failed = [k for k, v in status.items() if isinstance(v, str) and v.startswith("failed")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
