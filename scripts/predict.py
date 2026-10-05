"""Score a single new quarter or sample end to end, matching the training features exactly.

Loads the frozen pipeline, rebuilds the 47 features with `forecasting.features`, prints P(distress) with
the operating-threshold decision and a backtest warning, and optionally reports KernelSHAP values.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple

import joblib
import numpy as np

from forecasting.config import MODELS_DIR, RESULTS_DIR, TARGET, ensure_utf8_stdio
from forecasting.data_loader import load_prepared
from forecasting.explain import (DEFAULT_N_BACKGROUND, DEFAULT_N_COALITIONS,
                                 kernel_shap_values, pipeline_predict_fn)
from forecasting.features import build_feature_matrix, feature_names

#: Split lookup order, test first so the backtest warning is the most visible.
SPLITS: Tuple[str, ...] = ("test", "validation", "train", "purged")
#: Number of features shown when `--explain` is used.
DEFAULT_TOP_K = 6


def load_model(path: Optional[Path] = None) -> Dict[str, Any]:
    """Load `best.joblib` and fail with a clear message when training has not been run yet."""
    artifact_path = Path(path) if path else MODELS_DIR / "best.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"{artifact_path} is missing; run `python -m forecasting.train` first.")
    return joblib.load(artifact_path)


def operating_threshold(artifact: Dict[str, Any]) -> float:
    """Operating threshold, read from the artifact, then `summary.json`, then 0.5."""
    value = artifact.get("threshold")
    if value is None:
        summary_path = RESULTS_DIR / "summary.json"
        if summary_path.exists():
            value = json.loads(summary_path.read_text(encoding="utf-8")).get("best_threshold")
    return float(value) if value is not None else 0.5


def find_sample(sample_id: Optional[str] = None, ticker: Optional[str] = None,
                quarter: Optional[str] = None,
                splits: Sequence[str] = SPLITS) -> Tuple[Dict[str, Any], str]:
    """Look up one sample in `data/prepared/*.json` and return `(sample, split_name)`.

    Identifies the sample by `sample_id` such as `HD-2024Q2` or by the `(ticker, quarter)` pair, so
    `format_report` can warn when the sample belongs to validation or test and is therefore a backtest.
    """
    wanted_id = (sample_id or "").strip().upper()
    wanted_ticker = (ticker or "").strip().upper()
    wanted_quarter = (quarter or "").strip().upper()
    if not wanted_id and not (wanted_ticker and wanted_quarter):
        raise ValueError("Pass --sample-id, or both --ticker and --quarter.")
    for name in splits:
        for sample in load_prepared(name):
            current_id = str(sample.get("sample_id", "")).upper()
            if wanted_id and current_id != wanted_id:
                continue
            if wanted_ticker and str(sample.get("ticker", "")).upper() != wanted_ticker:
                continue
            if wanted_quarter and not current_id.endswith(wanted_quarter):
                continue
            return sample, name
    label = sample_id or f"{ticker}-{quarter}"
    raise KeyError(f"Sample {label!r} not found in splits {list(splits)}.")


def score_sample(sample: Dict[str, Any], artifact: Dict[str, Any], *,
                 explain: bool = False, top_k: int = DEFAULT_TOP_K,
                 n_coalitions: int = DEFAULT_N_COALITIONS,
                 n_background: int = DEFAULT_N_BACKGROUND) -> Dict[str, Any]:
    """Score one sample: probability, operating-threshold decision and optional top-K SHAP."""
    model = artifact["model"]
    names = list(artifact.get("features") or feature_names())
    matrix = build_feature_matrix([sample])
    if matrix.shape[1] != len(names):
        raise ValueError(
            f"Feature count does not match the artifact: {matrix.shape[1]} versus {len(names)}.")
    threshold = operating_threshold(artifact)
    probability = float(np.asarray(model.predict_proba(matrix))[0, 1])
    result: Dict[str, Any] = {
        "sample_id": sample.get("sample_id"),
        "ticker": sample.get("ticker"),
        "as_of": (sample.get("request") or {}).get("as_of"),
        "model": artifact.get("name"),
        "threshold": threshold,
        "probability": probability,
        "decision": int(probability >= threshold),
        "actual_label": sample.get(TARGET) if TARGET in sample else None,
        "n_features": int(matrix.shape[1]),
    }
    if explain:
        background = build_feature_matrix(load_prepared("train")[:max(2, int(n_background))])
        values = kernel_shap_values(pipeline_predict_fn(model), background, matrix[0],
                                    n_coalitions=n_coalitions,
                                    rng=np.random.default_rng(0))
        order = np.argsort(-np.abs(values["phi"]))[:top_k]
        result["explain"] = {
            "base_value": float(values["base_value"]),
            "prediction": float(values["prediction"]),
            "efficiency_gap": float(values["efficiency_gap"]),
            "n_coalitions": int(values["n_coalitions"]),
            "background_size": int(background.shape[0]),
            "contributions": [
                {"feature": names[int(i)], "phi": float(values["phi"][int(i)])} for i in order],
        }
    return result


def format_report(result: Dict[str, Any], split: Optional[str] = None) -> str:
    """Render the result for a human reader, with the backtest warning and the true-label comparison."""
    lines = [
        f"Sample         : {result['sample_id']} (ticker {result['ticker']}, as_of "
        f"{result['as_of']})",
        f"Model          : {result['model']}  ·  {result['n_features']} features",
        f"P(distress)    : {result['probability']:.4f}",
        f"Threshold      : {result['threshold']:.4f}",
        f"Decision       : {'DISTRESSED (1)' if result['decision'] else 'NOT distressed (0)'}",
    ]
    if result.get("actual_label") is not None:
        match = "match" if int(result["actual_label"]) == result["decision"] else "MISMATCH"
        lines.append(f"Actual label   : {int(result['actual_label'])} ({match} with the decision)")
    if split:
        lines.append(f"Split          : {split}")
        if split in ("test", "validation"):
            lines.append(
                "⚠  The sample belongs to a split used to SELECT the model and threshold, so this run is a "
                "backtest rather than a forecast for a future quarter.")
        elif split == "purged":
            lines.append("⚠  The sample falls in the purge band excluded from every split.")
    explain = result.get("explain")
    if explain:
        top_sum = sum(c["phi"] for c in explain["contributions"])
        lines += ["", f"Explanation (KernelSHAP, {explain['n_coalitions']} coalitions, background of "
                      f"{explain['background_size']} train samples):",
                  f"  base value E[f] = {explain['base_value']:.4f} · Σφ(top-K) = {top_sum:+.4f} · "
                  f"efficiency gap = {explain['efficiency_gap']:.2e}"]
        for item in explain["contributions"]:
            direction = "↑ increases risk" if item["phi"] > 0 else "↓ lowers risk"
            lines.append(f"  {item['feature']:<34} φ = {item['phi']:+.4f}  {direction}")
    return "\n".join(lines)


def run(sample_id: Optional[str] = None, ticker: Optional[str] = None,
        quarter: Optional[str] = None, input_path: Optional[str] = None,
        explain: bool = False, top_k: int = DEFAULT_TOP_K,
        as_json: bool = False) -> Dict[str, Any]:
    """Score one sample from prepared data or a JSON file and return the result mapping."""
    if input_path:
        sample = json.loads(Path(input_path).read_text(encoding="utf-8"))
        split = None
    else:
        sample, split = find_sample(sample_id=sample_id, ticker=ticker, quarter=quarter)
    artifact = load_model()
    result = score_sample(sample, artifact, explain=explain, top_k=top_k)
    if as_json:
        print(json.dumps({**result, "split": split}, ensure_ascii=False, indent=2, default=float))
    else:
        print(format_report(result, split))
    return {**result, "split": split}


def main(argv: Optional[Sequence[str]] = None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(
        description="Score a single new quarter or sample with the frozen model.",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-id", help="e.g. HD-2024Q2, looked up in data/prepared/*.json")
    parser.add_argument("--ticker", help="company ticker, used together with --quarter")
    parser.add_argument("--quarter", help="quarter, e.g. 2024Q3, used together with --ticker")
    parser.add_argument("--input", dest="input_path",
                        help="JSON file holding one sample in the prepared format")
    parser.add_argument("--explain", action="store_true",
                        help="also print the top-K KernelSHAP contributions")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="print the result as JSON for machine consumption")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(sample_id=args.sample_id, ticker=args.ticker, quarter=args.quarter,
        input_path=args.input_path, explain=args.explain, top_k=args.top_k,
        as_json=args.as_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
