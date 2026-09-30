"""Báo cáo kết quả: bảng điểm từng công ty + figure chính.

Lệnh: python -m forecasting.report

- Nạp test_evaluation.json (do forecasting.evaluate tạo).
- Xuất bảng xếp hạng distress theo công ty/quý (CSV) + biểu đồ histogram xác suất.
"""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from .config import FIGURES_DIR, MODELS_DIR, RESULTS_DIR, ensure_dirs, ensure_utf8_stdio
from .data_loader import load_prepared
from .features import build_feature_matrix
from .models import predict_proba


def load_threshold() -> float:
    """Threshold chọn trên validation (summaries/best_threshold)."""
    summary_path = RESULTS_DIR / "summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text(encoding="utf-8"))
        return float(summary.get("best_threshold", 0.5))
    return 0.5


def run() -> Dict[str, Any]:
    ensure_dirs()
    artifact = joblib.load(MODELS_DIR / "best.joblib")
    model, name = artifact["model"], artifact["name"]
    threshold = load_threshold()
    test_samples = load_prepared("test")
    proba = predict_proba(model, build_feature_matrix(test_samples))

    rows = [
        {
            "sample_id": s["sample_id"],
            "ticker": s["ticker"],
            "as_of": s["request"]["as_of"],
            "target_quarter": (s["request"]["target_period_start"] + " -> "
                               + s["request"]["target_period_end"]),
            "probability": float(p),
            "predicted": 1 if p >= threshold else 0,
            "actual": int(s["is_distressed"]),
        }
        for s, p in zip(test_samples, proba)
    ]
    rows.sort(key=lambda r: (-r["probability"], r["ticker"]))

    csv_path = RESULTS_DIR / "test_predictions.csv"
    with open(csv_path, "w", newline="", encoding="utf-8-sig") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    # Histogram xác suất theo thực tế
    fig, ax = plt.subplots(figsize=(6, 4))
    actuals = [r["actual"] for r in rows]
    probs = [r["probability"] for r in rows]
    ax.hist([p for p, a in zip(probs, actuals) if a == 1],
            bins=20, alpha=0.6, label="Distress (thực)", color="crimson")
    ax.hist([p for p, a in zip(probs, actuals) if a == 0],
            bins=20, alpha=0.5, label="Không distress", color="steelblue")
    ax.axvline(threshold, color="k", linestyle="--", label=f"threshold={threshold:.2f}")
    ax.set_xlabel("Xác suất suy giảm (model)")
    ax.set_ylabel("Số mẫu test")
    ax.set_title("Phân phối xác suất theo nhãn thực — test")
    ax.legend()
    ax.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "test_score_distribution.png", dpi=120)
    plt.close(fig)

    print(f"Đã xuất {csv_path} ({len(rows)} mẫu) và "
          f"{FIGURES_DIR / 'test_score_distribution.png'}")
    return {"rows": rows, "csv": str(csv_path), "model": name, "threshold": threshold}


def main(argv=None) -> int:
    ensure_utf8_stdio()
    _ = argv
    run()
    return 0


if __name__ == "__main__":
    sys.exit(main())
