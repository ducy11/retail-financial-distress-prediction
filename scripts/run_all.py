"""Chạy TOÀN BỘ pipeline theo thứ tự và ghi log — một lệnh tái lập mọi artifact.

Lệnh: python -m scripts.run_all [--skip tune,analysis] [--only train,evaluate]

Vì sao cần: kết quả trong `reports/` phải tái tạo được từ dữ liệu trong repo. Script này chạy
lần lượt data → provenance → eda → eda_deep → train → baselines → validation → tuning → evaluate →
report → analyze → prep_exp → imbalance_real → search → explain → significance → label_sensitivity →
relabel → predict → make_report → export_office, ghi log vào `reports/results/run_all.log`.

Bước nào lỗi thì ghi rõ trong log và tiếp tục (trừ `train`/`evaluate` là bước lõi).
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

#: (nhãn, module, hàm, kwargs, bước lõi?)
STEPS: List[Tuple[str, str, str, Dict[str, Any], bool]] = [
    ("data (tái tạo split từ retail-expanded)", "forecasting.data", "run", {"force": False}, False),
    ("provenance (kiểm chứng dữ liệu THẬT từ snapshot SEC)",
     "scripts.verify_provenance", "run", {}, False),
    ("eda (hình EDA + bảng tổng quan)", "scripts.eda", "run", {}, False),
    ("eda_deep (feature/nhãn/tương quan/drift chuyên sâu)", "scripts.eda_deep", "run", {}, False),
    ("train (3 họ mô hình, chọn theo AP cross-company)", "forecasting.train", "run", {}, True),
    ("baselines (dummy / ticker-prior / 1-feature)", "forecasting.baselines", "run", {}, False),
    ("validation (GroupKFold / LOCO / bootstrap)", "forecasting.validation", "run", {}, False),
    ("tuning (GridSearchCV chia theo công ty)", "forecasting.tuning", "run", {}, False),
    ("evaluate (chốt trên test, 1 lần)", "forecasting.evaluate", "run", {}, True),
    ("report (bảng dự báo từng mẫu + histogram)", "forecasting.report", "run", {}, False),
    ("analyze (importance / ablation / lỗi / ngưỡng)", "scripts.analyze", "run", {}, False),
    ("prep_exp (winsorize × scaler trên dữ liệu thật)",
     "scripts.experiment_preprocessing", "run", {}, False),
    ("imbalance_real (kỹ thuật lệch lớp trên dữ liệu thật)",
     "scripts.experiment_imbalance_real", "run", {}, False),
    ("search (random search + sổ thực nghiệm runs.csv)", "scripts.search", "run", {}, False),
    ("explain (SHAP/KernelSHAP + hình giải thích)", "scripts.explain_model", "run", {}, False),
    ("significance (DeLong + paired bootstrap)", "scripts.significance", "run", {}, False),
    ("label_sensitivity (độ nhạy theo 4 định nghĩa nhãn)",
     "scripts.label_sensitivity", "run", {}, False),
    ("relabel (nhãn quy tắc tái lập + so sánh)", "scripts.relabel", "run", {}, False),
    ("predict (demo: chấm 1 mẫu bằng mô hình đã chốt)", "scripts.predict", "run",
     {"sample_id": "HD-2024Q2"}, False),
    ("make_report (docs/BAO-CAO.md + slide.md)", "scripts.make_report", "run", {}, False),
    ("export_office (Word .docx + Slide .pptx)", "scripts.export_office", "run", {}, False),
]


def _call(module_name: str, func_name: str, kwargs: Dict[str, Any]) -> Any:
    module = importlib.import_module(module_name)
    return getattr(module, func_name)(**kwargs)


def run(skip: List[str] | None = None, only: List[str] | None = None) -> Dict[str, Any]:
    """Chạy pipeline; trả {bước: trạng thái}."""
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
            lines.append(f"[BỎ QUA] {label}")
            status[key] = "skipped"
            continue
        buffer = io.StringIO()
        try:
            with warnings.catch_warnings(), redirect_stdout(buffer), redirect_stderr(buffer):
                warnings.simplefilter("ignore")  # log sạch; cảnh báo kỹ thuật không phải kết quả
                result = _call(module_name, func_name, kwargs)
            status[key] = "ok"
            lines.append(f"[OK]     {label}")
        except (ModuleNotFoundError, AttributeError) as e:
            status[key] = f"missing ({type(e).__name__})"
            lines.append(f"[CHƯA CÓ] {label} — {e}")
        except Exception as e:  # noqa: BLE001 - log lại toàn bộ để debug
            status[key] = f"failed: {type(e).__name__}"
            lines.append(f"[LỖI]    {label} — {type(e).__name__}: {e}")
            lines.append(traceback.format_exc())
        body = buffer.getvalue().strip()
        if body:
            lines.append("    " + body.replace("\n", "\n    "))
        _ = result

    header = "=== run_all: trạng thái các bước ==="
    text = "\n".join([header] + [f"  {k}: {v}" for k, v in status.items()] + ["", *lines])
    log_path.write_text(text + "\n", encoding="utf-8")
    print(text if len(text) < 6000 else text[:6000] + "\n... (xem đầy đủ trong run_all.log)")
    return status


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--skip", default="", help="Bỏ qua các bước, phân tách bằng dấu phẩy.")
    parser.add_argument("--only", default="", help="Chỉ chạy các bước này, phân tách bằng dấu phẩy.")
    args = parser.parse_args(argv)
    status = run(skip=[s for s in args.skip.split(",") if s],
                 only=[s for s in args.only.split(",") if s])
    failed = [k for k, v in status.items() if isinstance(v, str) and v.startswith("failed")]
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
