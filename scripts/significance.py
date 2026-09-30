"""Kiểm định ý nghĩa thống kê: mô hình có THẬT SỰ hơn baseline trên test hay không?

Lệnh: python -m scripts.significance [--n-boot 2000] [--no-write]

So trên **cùng 64 mẫu test** (đúng một lần, không dùng để chọn cấu hình):
- `model[<tên>]` — mô hình đã chốt (`reports/models/best.joblib`);
- `ticker_prior` — baseline "nhớ mặt công ty" (tỉ lệ nhãn trung bình theo công ty trong train);
- `single_feature[debt_to_assets_latest]` — logistic một đặc trưng (đối chứng tối thiểu);
- `dummy_most_frequent` — lớp đa số.

Phép kiểm định: **DeLong (1988)** cho ΔAUROC + **paired bootstrap** cho ΔAP (CI 95% + p-value).
Đây là căn cứ trả lời câu hỏi khó nhất của đồ án: *"AUROC 0,98 có chứng minh năng lực dự báo?"*
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from forecasting.baselines import fit_dummy, fit_single_feature
from forecasting.config import RESULTS_DIR, TARGET, ensure_dirs, ensure_utf8_stdio
from forecasting.data_loader import load_prepared
from forecasting.eda import markdown_table
from forecasting.features import build_feature_matrix, extract_labels, feature_names
from forecasting.models import predict_proba
from forecasting.significance import compare_systems


def _ticker_prior_probabilities(train, test) -> np.ndarray:
    """Xác suất = tỉ lệ nhãn 1 của chính công ty trong TRAIN (baseline không dùng feature)."""
    prior: Dict[str, float] = {}
    for ticker in sorted({str(s["ticker"]) for s in train}):
        labels = [int(s[TARGET]) for s in train if str(s["ticker"]) == ticker]
        prior[ticker] = float(np.mean(labels))
    global_prior = float(np.mean([int(s[TARGET]) for s in train]))
    return np.array([prior.get(str(s["ticker"]), global_prior) for s in test])


def collect_systems() -> Dict[str, Any]:
    """Xác suất test của mọi hệ thống cần so sánh (mọi thứ học từ TRAIN)."""
    import joblib

    from forecasting.config import MODELS_DIR

    splits = {name: load_prepared(name) for name in ("train", "validation", "test")}
    train, test = splits["train"], splits["test"]
    y_train, y_test = extract_labels(train), extract_labels(test)
    X_test = build_feature_matrix(test)

    artifact = joblib.load(MODELS_DIR / "best.joblib")
    systems: Dict[str, np.ndarray] = {
        f"model[{artifact['name']}]": predict_proba(artifact["model"], X_test)}
    systems["ticker_prior"] = _ticker_prior_probabilities(train, test)
    single, index = fit_single_feature(train, y_train)
    # `fit_single_feature` chỉ dùng MỘT cột ⇒ phải truyền đúng cột đó (không truyền cả 47).
    systems[f"single_feature[{feature_names()[index]}]"] = predict_proba(single, X_test[:, [index]])
    systems["dummy_most_frequent"] = predict_proba(fit_dummy(train, y_train), X_test)
    return {"systems": systems, "y_test": y_test, "feature_used": feature_names()[index],
            "model_name": artifact["name"]}


def _fmt(value: Any) -> str:
    """Định dạng số gọn cho thông báo/log (None → '—')."""
    if value is None:
        return "—"
    try:
        return f"{float(value):.4g}"
    except (TypeError, ValueError):
        return str(value)


def run(write: bool = True, n_boot: int = 2000, out_dir: Path | None = None) -> Dict[str, Any]:
    """Chạy kiểm định, ghi `reports/results/significance.{json,md}`."""
    ensure_dirs()
    ensure_utf8_stdio()
    out = Path(out_dir) if out_dir else RESULTS_DIR
    collected = collect_systems()
    model_key = f"model[{collected['model_name']}]"
    result = compare_systems(collected["y_test"], collected["systems"], n_boot=n_boot,
                             baseline="ticker_prior")
    result["model_key"] = model_key
    result["feature_used_by_single"] = collected["feature_used"]
    result["pairs_vs_baseline"] = [p for p in result["pairs"] if "ticker_prior" in (p["a"], p["b"])]

    central = next((p for p in result["pairs"] if model_key in (p["a"], p["b"])
                    and "ticker_prior" in (p["a"], p["b"])), None)
    notes: List[str] = []
    if central:
        sign = 1 if central["a"] == model_key else -1
        delong, boot_ap = central["delong_auroc"], central["bootstrap_ap"]
        notes.append(
            f"**ΔAUROC (DeLong):** {model_key} − ticker_prior = "
            f"{sign * (delong['delta'] or 0):+.4f}, z = {_fmt(delong['z'])}, p = "
            f"{_fmt(delong['p_value'])} ⇒ "
            + ("khác biệt **có ý nghĩa** ở mức 5%."
               if (delong["p_value"] or 1) < 0.05 else
               "**không** có ý nghĩa ở mức 5% (chưa thể khẳng định mô hình hơn baseline)."))
        low, high = sorted([sign * boot_ap["ci95_lower"], sign * boot_ap["ci95_upper"]])
        notes.append(
            f"**ΔAP (paired bootstrap {boot_ap['n_boot']} vòng):** {sign * boot_ap['delta']:+.4f}, "
            f"CI 95% = [{low:+.4f}; {high:+.4f}], p = {boot_ap['p_value']:.4f} ⇒ "
            + ("kết luận tương tự DeLong."
               if (boot_ap["p_value"] < 0.05) == ((delong["p_value"] or 1) < 0.05) else
               "hai kiểm định KHÔNG đồng thuận ⇒ phải báo cáo cả hai, không chốt một phía."))
        notes.append(
            "**Cách đọc:** n = 64 mẫu nên khoảng tin cậy rộng; \"không khác biệt\" nghĩa là *dữ liệu "
            "chưa đủ để khẳng định*, không phải bằng chứng mô hình kém. Đây là lý do báo cáo dùng thêm "
            "so sánh cross-company (`validation_checks.json`) và lấy `ticker_prior` làm mốc trung thực.")
    result["conclusions"] = notes

    if write:
        (out / "significance.json").write_text(
            json.dumps(result, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        (out / "significance.md").write_text(markdown_significance(result), encoding="utf-8")
    p_value = (central or {}).get("delong_auroc", {}).get("p_value")
    print(f"Kiểm định: {len(result['systems'])} hệ thống, {len(result['pairs'])} cặp so sánh; "
          f"ΔAUROC (DeLong) vs ticker_prior: p = {_fmt(p_value)}")
    return result


def markdown_significance(result: Dict[str, Any]) -> str:
    """Sinh `reports/results/significance.md` (mọi số đọc từ JSON)."""
    model_key = result.get("model_key", "model")
    lines = ["# Kiểm định ý nghĩa thống kê trên test (DeLong + paired bootstrap)", "",
             f"- Tập so sánh: **{result['n_samples']} mẫu test** ({result['n_positive']} dương); mọi hệ "
             f"thống chấm trên **cùng** mẫu, test không được dùng để chọn cấu hình.",
             "- Hệ thống: " + ", ".join(f"`{name}`" for name in result["systems"]) + ".",
             "- Kiểm định: DeLong (1988) cho ΔAUROC; paired bootstrap cho ΔAP/ΔAUROC "
             "(resample theo lớp, seed cố định).",
             "",
             "## 1. Kết quả tổng hợp", "",
             markdown_table(["Hệ thống", "AUROC", "Average Precision"],
                            [[name, f"{m['auroc']:.4f}" if m["auroc"] is not None else "—",
                              f"{m['average_precision']:.4f}"]
                             for name, m in result["systems"].items()],
                            ["---", "---:", "---:"]),
             "",
             "## 2. Mô hình so với baseline `ticker_prior`", ""]
    lines += [f"{i + 1}. {text}" for i, text in enumerate(result.get("conclusions") or [])]
    lines += ["", "## 3. Mọi cặp so sánh (ΔAUROC DeLong · ΔAP bootstrap)", "",
              markdown_table(["A", "B", "ΔAUROC (A−B)", "p (DeLong)", "ΔAP (A−B)", "CI95 ΔAP",
                              "p (bootstrap AP)"],
                             [[p["a"], p["b"],
                               f"{(p['delong_auroc']['delta'] or 0):+.4f}",
                               _fmt(p["delong_auroc"]["p_value"]),
                               f"{p['bootstrap_ap']['delta']:+.4f}",
                               f"[{p['bootstrap_ap']['ci95_lower']:+.4f}; "
                               f"{p['bootstrap_ap']['ci95_upper']:+.4f}]",
                               f"{p['bootstrap_ap']['p_value']:.4f}"] for p in result["pairs"]],
                             ["---", "---", "---:", "---:", "---:", "---:", "---:"]),
              "",
              f"> Đọc bảng: p < 0,05 ⇒ khác biệt khó giải thích bằng ngẫu nhiên. Nếu `{model_key}` "
              f"không khác `ticker_prior` về ý nghĩa thống kê thì kết luận trung thực là lợi thế của mô "
              f"hình **chưa được chứng minh** trên bộ test này (n nhỏ) — phải đọc kèm kết quả "
              f"cross-company.",
              ""]
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    """CLI: `python -m scripts.significance [--n-boot N] [--no-write]`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--no-write", action="store_true")
    args = parser.parse_args(argv)
    print("=== Kiểm định ý nghĩa thống kê (DeLong + paired bootstrap) ===")
    run(write=not args.no_write, n_boot=args.n_boot)
    return 0


if __name__ == "__main__":
    sys.exit(main())
