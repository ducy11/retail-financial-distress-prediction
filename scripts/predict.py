"""Dự đoán cho MỘT quý/mẫu mới — bản demo end-to-end của đồ án.

Lệnh:

    python -m scripts.predict --sample-id HD-2024Q2
    python -m scripts.predict --ticker FIVE --quarter 2024Q3 --explain
    python -m scripts.predict --input mau-moi.json --explain --json

Vì sao cần: khi bảo vệ, hội đồng thường yêu cầu "chạy thử trên một mẫu" chứ không chỉ xem bảng có
sẵn. Script nạp pipeline ĐÃ FIT (`reports/models/best.joblib`, do `forecasting.train` sinh ra), dựng
lại **đúng 47 feature** bằng `forecasting.features` (nên không thể lệch với lúc huấn luyện), rồi in:

1. xác suất suy giảm P(distress) + quyết định tại **ngưỡng vận hành** (không phải 0,5);
2. cảnh báo nếu mẫu thuộc split đã dùng để chọn mô hình/ngưỡng (validation/test) — khi đó kết quả
   là **chạy lại lịch sử**, không phải dự báo tương lai;
3. tuỳ chọn `--explain`: top-K đóng góp KernelSHAP của chính mẫu đó (`forecasting.explain`), kèm
   đối chiếu nhãn thật khi mẫu nằm trong dữ liệu đã có.

Định dạng `--input` là **đúng một phần tử** của `data/prepared/*.json`:

    {"sample_id": "ABC-2025Q1", "ticker": "ABC",
     "request": {"history": [{"fiscal_year": 2024, "fiscal_quarter": 3,
                              "revenue_vnd": "...", "total_assets_vnd": "...", ...}, ...],
                 "as_of": "2024-11-01"},
     "is_distressed": 0, "label_available_on": "2025-02-01"}

với mọi chỉ tiêu trong `config.BASE_FIELDS` (hậu tố `_vnd`, chuỗi số nguyên) là tuỳ chọn — thiếu thì
feature tương ứng là NaN và được impute bằng median trong pipeline.
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

#: Thứ tự split dùng để tra mẫu: test trước (để cảnh báo backtest hiện rõ nhất).
SPLITS: Tuple[str, ...] = ("test", "validation", "train", "purged")
#: Số feature trình bày khi `--explain`.
DEFAULT_TOP_K = 6


def load_model(path: Optional[Path] = None) -> Dict[str, Any]:
    """Nạp `best.joblib` (dict: model, name, features, threshold); báo lỗi rõ nếu chưa huấn luyện."""
    artifact_path = Path(path) if path else MODELS_DIR / "best.joblib"
    if not artifact_path.exists():
        raise FileNotFoundError(
            f"Chưa có {artifact_path} — chạy `python -m forecasting.train` trước.")
    return joblib.load(artifact_path)


def operating_threshold(artifact: Dict[str, Any]) -> float:
    """Ngưỡng vận hành: ưu tiên trong artifact, rồi `summary.json`, cuối cùng là 0.5."""
    value = artifact.get("threshold")
    if value is None:
        summary_path = RESULTS_DIR / "summary.json"
        if summary_path.exists():
            value = json.loads(summary_path.read_text(encoding="utf-8")).get("best_threshold")
    return float(value) if value is not None else 0.5


def find_sample(sample_id: Optional[str] = None, ticker: Optional[str] = None,
                quarter: Optional[str] = None,
                splits: Sequence[str] = SPLITS) -> Tuple[Dict[str, Any], str]:
    """Tra một mẫu trong `data/prepared/*.json`; trả `(sample, tên split)`.

    Nhận diện theo `sample_id` (vd `HD-2024Q2`) hoặc cặp `(ticker, quarter)`. Trả kèm tên split để
    `format_report` cảnh báo khi mẫu thuộc validation/test (tức là backtest, không phải dự báo mới).
    """
    wanted_id = (sample_id or "").strip().upper()
    wanted_ticker = (ticker or "").strip().upper()
    wanted_quarter = (quarter or "").strip().upper()
    if not wanted_id and not (wanted_ticker and wanted_quarter):
        raise ValueError("Cần --sample-id, hoặc cả --ticker và --quarter.")
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
    raise KeyError(f"Không tìm thấy mẫu {label!r} trong các split {list(splits)}.")


def score_sample(sample: Dict[str, Any], artifact: Dict[str, Any], *,
                 explain: bool = False, top_k: int = DEFAULT_TOP_K,
                 n_coalitions: int = DEFAULT_N_COALITIONS,
                 n_background: int = DEFAULT_N_BACKGROUND) -> Dict[str, Any]:
    """Chấm 1 mẫu: xác suất, quyết định theo ngưỡng vận hành, (tuỳ chọn) top-K SHAP."""
    model = artifact["model"]
    names = list(artifact.get("features") or feature_names())
    matrix = build_feature_matrix([sample])
    if matrix.shape[1] != len(names):
        raise ValueError(
            f"Số feature không khớp artifact: {matrix.shape[1]} so với {len(names)}.")
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
    """In kết quả dạng người đọc được (kèm cảnh báo backtest và đối chiếu nhãn thật)."""
    lines = [
        f"Mẫu            : {result['sample_id']} (công ty {result['ticker']}, as_of "
        f"{result['as_of']})",
        f"Mô hình        : {result['model']}  ·  {result['n_features']} feature",
        f"P(distress)    : {result['probability']:.4f}",
        f"Ngưỡng vận hành: {result['threshold']:.4f}",
        f"Quyết định     : {'SUY GIẢM (1)' if result['decision'] else 'KHÔNG suy giảm (0)'}",
    ]
    if result.get("actual_label") is not None:
        match = "khớp" if int(result["actual_label"]) == result["decision"] else "LỆCH"
        lines.append(f"Nhãn thật      : {int(result['actual_label'])} ({match} với quyết định)")
    if split:
        lines.append(f"Split          : {split}")
        if split in ("test", "validation"):
            lines.append(
                "⚠  Mẫu nằm trong split đã dùng để CHỌN mô hình/ngưỡng ⇒ kết quả này là chạy lại "
                "lịch sử (backtest), không phải dự báo tương lai.")
        elif split == "purged":
            lines.append("⚠  Mẫu thuộc dải purge (bị loại khỏi cả train/validation/test).")
    explain = result.get("explain")
    if explain:
        top_sum = sum(c["phi"] for c in explain["contributions"])
        lines += ["", f"Giải thích (KernelSHAP, {explain['n_coalitions']} liên minh, nền "
                      f"{explain['background_size']} mẫu train):",
                  f"  giá trị nền E[f] = {explain['base_value']:.4f} · Σφ(top-K) = {top_sum:+.4f} · "
                  f"sai số efficiency = {explain['efficiency_gap']:.2e}"]
        for item in explain["contributions"]:
            direction = "↑ tăng rủi ro" if item["phi"] > 0 else "↓ giảm rủi ro"
            lines.append(f"  {item['feature']:<34} φ = {item['phi']:+.4f}  {direction}")
    return "\n".join(lines)


def run(sample_id: Optional[str] = None, ticker: Optional[str] = None,
        quarter: Optional[str] = None, input_path: Optional[str] = None,
        explain: bool = False, top_k: int = DEFAULT_TOP_K,
        as_json: bool = False) -> Dict[str, Any]:
    """Chấm một mẫu (từ prepared hoặc từ file JSON) và trả dict kết quả."""
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
        description="Dự đoán một quý/mẫu mới bằng mô hình đã chốt (demo end-to-end).",
        formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sample-id", help="vd HD-2024Q2 (tra trong data/prepared/*.json)")
    parser.add_argument("--ticker", help="mã công ty, dùng cùng --quarter")
    parser.add_argument("--quarter", help="quý, vd 2024Q3 (dùng cùng --ticker)")
    parser.add_argument("--input", dest="input_path",
                        help="file JSON chứa MỘT mẫu theo định dạng prepared")
    parser.add_argument("--explain", action="store_true",
                        help="in thêm top-K đóng góp KernelSHAP")
    parser.add_argument("--top-k", type=int, default=DEFAULT_TOP_K)
    parser.add_argument("--json", dest="as_json", action="store_true",
                        help="in kết quả dạng JSON (để máy đọc)")
    args = parser.parse_args(list(argv) if argv is not None else None)
    run(sample_id=args.sample_id, ticker=args.ticker, quarter=args.quarter,
        input_path=args.input_path, explain=args.explain, top_k=args.top_k,
        as_json=args.as_json)
    return 0


if __name__ == "__main__":
    sys.exit(main())
