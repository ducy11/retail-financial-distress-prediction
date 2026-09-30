"""Đo MẤT CÂN BẰNG LỚP (class imbalance) của nhãn mục tiêu `is_distressed`.

Lệnh: python -m scripts.class_balance [--no-write]

Nhãn nằm ở `data/prepared/{train,validation,test,purged}.json` (khoá `is_distressed`, xem
`forecasting/config.py::TARGET`; lớp dương = 1 = suy giảm tài chính). Các file .csv trong repo
**không** chứa nhãn huấn luyện: `data/samples/*.csv` là mẫu định dạng BCTC, còn
`reports/results/test_predictions.csv` chỉ có `actual` của riêng tập test.

Vì sao cần script riêng: đồ án **không dùng pandas** (`requirements.txt` ghi rõ "không bao gồm
pandas"), nên thay `value_counts()` và `value_counts(normalize=True) * 100` bằng đếm thuần Python.
Script trả lời 3 câu cho từng tập, cho toàn bộ corpus và cho TỪNG CÔNG TY (nơi nhãn gần như là
thuộc tính của thực thể nên mất cân bằng "thật" nằm ở đây):
1. số mẫu tuyệt đối mỗi lớp,
2. tỉ lệ % mỗi lớp,
3. tỉ số mất cân bằng IR = mẫu đa số / mẫu thiểu số,
kèm độ chính xác của đường cơ sở "đoán lớp đa số" (để thấy vì sao Accuracy không dùng được).

Phân loại mức mất cân bằng (ngưỡng của yêu cầu kiểm tra):
- `balanced`: lớp thiểu số ≥ 40% (tức ~60/40 trở xuống);
- `slightly_imbalanced`: 20% ≤ lớp thiểu số < 40%;
- `severely_imbalanced`: lớp thiểu số < 20% (tức từ 80/20 trở lên).

Kết quả: in ra màn hình + ghi `reports/results/class_balance.{json,md}` (số liệu đọc từ
`data/prepared/*` — không nhập tay).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any, Dict, List, Sequence

from forecasting.config import RESULTS_DIR, TARGET, ensure_utf8_stdio
from forecasting.data_loader import load_prepared
from forecasting.models import HYPERPARAMS

SPLITS = ("train", "validation", "test", "purged")
RULE_DIR = Path("data/prepared-rule")

#: Ngưỡng phân loại mức mất cân bằng (tỉ lệ % của lớp THIỂU SỐ).
BALANCED_MIN_MINORITY_PCT = 40.0
SEVERE_MAX_MINORITY_PCT = 20.0


def imbalance_summary(labels: Sequence[int]) -> Dict[str, Any]:
    """Thống kê mất cân bằng của một dãy nhãn 0/1 — tương đương value_counts + normalize."""
    n = len(labels)
    positive = sum(1 for v in labels if int(v) == 1)
    negative = n - positive
    if n == 0:
        return {"n": 0}
    counts = {"1": positive, "0": negative}
    percent = {"1": 100.0 * positive / n, "0": 100.0 * negative / n}
    minority = 0 if positive >= negative else 1
    majority_size, minority_size = max(positive, negative), min(positive, negative)
    minority_pct = percent[str(minority)]
    if minority_size == 0:
        level, ratio = "single_class", float("inf")
    elif minority_pct >= BALANCED_MIN_MINORITY_PCT:
        level, ratio = "balanced", majority_size / minority_size
    elif minority_pct >= SEVERE_MAX_MINORITY_PCT:
        level, ratio = "slightly_imbalanced", majority_size / minority_size
    else:
        level, ratio = "severely_imbalanced", majority_size / minority_size
    return {
        "n": n,
        "counts": counts,
        "percent": percent,
        "minority_class": minority,
        "minority_percent": minority_pct,
        "imbalance_ratio": ratio,
        "level": level,
        "majority_baseline_accuracy_pct": 100.0 * majority_size / n,
    }


def _markdown(out: Dict[str, Any]) -> str:
    """Bảng Markdown: theo tập, toàn corpus, theo công ty, nhãn quy tắc + phần đọc kết quả."""
    per_split, per_ticker = out["per_split"], out["per_ticker"]
    overall = out["overall"]
    lines = ["# Mất cân bằng lớp của nhãn `is_distressed` (sinh tự động)", "",
             f"- Cột mục tiêu: **`{out['target_column']}`** trong `{out['label_source']}` "
             f"(1 = suy giảm tài chính).",
             "- Đếm thuần Python (repo không dùng pandas) — tương đương `value_counts()` và "
             "`value_counts(normalize=True) * 100`.",
             f"- Ngưỡng phân loại: thiểu số ≥ {out['level_thresholds']['balanced_minority_pct']:.0f}% "
             f"→ cân bằng; < {out['level_thresholds']['severe_minority_pct_below']:.0f}% → nghiêm "
             f"trọng.", ""]

    def table_block(title: str, stats: Dict[str, Any], key_label: str) -> None:
        lines.extend([f"## {title}", "",
                      f"| {key_label} | n | Lớp 1 (distress) | Lớp 0 | % thiểu số | "
                      f"IR (đa số/thiểu số) | Acc nếu đoán lớp đa số | Mức |",
                      "|---|---:|---:|---:|---:|---:|---:|---|"])
        for key, st in stats.items():
            if st.get("n"):
                lines.append(
                    f"| {key} | {st['n']} | {st['counts']['1']} ({st['percent']['1']:.1f}%) | "
                    f"{st['counts']['0']} ({st['percent']['0']:.1f}%) | {st['minority_percent']:.1f}% "
                    f"| {st['imbalance_ratio']:.2f} | {st['majority_baseline_accuracy_pct']:.1f}% | "
                    f"{_level_vi(st['level'])} |")
        lines.append("")

    table_block("Theo tập", per_split, "Tập")
    table_block("Toàn bộ corpus (train+validation+test+purged)", {"toàn corpus": overall}, "Phạm vi")
    table_block("Theo công ty (nhãn gần như là thuộc tính thực thể)", per_ticker, "Công ty")
    if out.get("rule_labels_overall"):
        table_block("Nhãn quy tắc tái lập được (`data/prepared-rule`)",
                    {"nhãn quy tắc": out["rule_labels_overall"]}, "Phạm vi")

    lines.extend([
        "## Đọc kết quả", "",
        f"- Toàn corpus: **{overall['counts']['1']} mẫu lớp 1 / {overall['counts']['0']} mẫu lớp 0** "
        f"({overall['percent']['1']:.1f}% / {overall['percent']['0']:.1f}%), "
        f"IR = **{overall['imbalance_ratio']:.2f}** → **{_level_vi(overall['level'])}**.",
        f"- Train: IR = {per_split['train']['imbalance_ratio']:.2f} "
        f"({_level_vi(per_split['train']['level'])}); "
        f"validation: IR = {per_split['validation']['imbalance_ratio']:.2f}; "
        f"test: IR = {per_split['test']['imbalance_ratio']:.2f} "
        f"({_level_vi(per_split['test']['level'])}).",
        "- Ở cấp CÔNG TY mức mất cân bằng nặng hơn hẳn: "
        + "; ".join(f"{t} {st['minority_percent']:.0f}% [{_level_vi(st['level'])}]"
                    for t, st in per_ticker.items()) + ".",
        f"- Đường cơ sở “đoán lớp đa số” đã đạt "
        f"{per_split['test']['majority_baseline_accuracy_pct']:.1f}% accuracy trên test ⇒ Accuracy "
        f"là chỉ số gây hiểu nhầm; phải dùng PR-AUC/AP, F1, recall tại ngưỡng chi phí.",
        "- Repo đã xử lý một phần: `class_weight` bật cho "
        + (", ".join(out["class_weight_enabled_in_hyperparams"]) or "—")
        + "; `forecasting/tuning` refit theo **average_precision**; `forecasting/evaluation` báo "
          "cáo F1/macro-F1, MCC và ngưỡng tối ưu theo chi phí `COST_FN`/`COST_FP`.",
        "- **Không** dùng SMOTE ở bộ dữ liệu này: nhãn gần như là thuộc tính công ty và mọi đánh giá "
        "đều chia theo nhóm (`GroupKFold`/LOCO); sinh mẫu tổng hợp trong không gian 47 chiều từ 212 "
        "mẫu train của 8 thực thể sẽ tạo quan sát “của chính công ty đã có”, tức hợp thức hoá đúng "
        "loại rò rỉ cấp thực thể mà đồ án đang đo.",
    ])
    return "\n".join(lines) + "\n"


def _labels(samples: Sequence[Dict[str, Any]]) -> List[int]:
    return [int(s[TARGET]) for s in samples]


def _level_vi(level: str) -> str:
    return {"balanced": "CÂN BẰNG", "slightly_imbalanced": "mất cân bằng NHẸ",
            "severely_imbalanced": "mất cân bằng NGHIÊM TRỌNG",
            "single_class": "chỉ một lớp"}.get(level, level)


def run(write: bool = True) -> Dict[str, Any]:
    """Tính thống kê mất cân bằng cho từng tập, toàn corpus, từng công ty + nhãn quy tắc."""
    ensure_utf8_stdio()
    splits = {name: load_prepared(name) for name in SPLITS}
    per_split = {name: imbalance_summary(_labels(arr)) for name, arr in splits.items()}
    pool = [s for name in SPLITS for s in splits[name]]
    overall = imbalance_summary(_labels(pool))

    by_ticker: Dict[str, List[int]] = {}
    for sample in pool:
        by_ticker.setdefault(sample["ticker"], []).append(int(sample[TARGET]))
    per_ticker = {ticker: imbalance_summary(v) for ticker, v in sorted(by_ticker.items())}

    try:
        rule_splits = {name: load_prepared(name, RULE_DIR) for name in SPLITS}
    except FileNotFoundError:
        rule_splits = {}
    rule_overall = (imbalance_summary(_labels([s for arr in rule_splits.values() for s in arr]))
                    if rule_splits else None)

    weighted = [name for name, cfg in HYPERPARAMS.items() if cfg.get("class_weight")]
    out: Dict[str, Any] = {
        "target_column": TARGET,
        "label_source": "data/prepared/*.json",
        "level_thresholds": {"balanced_minority_pct": BALANCED_MIN_MINORITY_PCT,
                             "severe_minority_pct_below": SEVERE_MAX_MINORITY_PCT},
        "per_split": per_split,
        "overall": overall,
        "per_ticker": per_ticker,
        "rule_labels_overall": rule_overall,
        "class_weight_enabled_in_hyperparams": weighted,
    }
    text = _markdown(out)
    if write:
        RESULTS_DIR.mkdir(parents=True, exist_ok=True)
        (RESULTS_DIR / "class_balance.json").write_text(
            json.dumps(out, ensure_ascii=False, indent=2, default=float) + "\n", encoding="utf-8")
        (RESULTS_DIR / "class_balance.md").write_text(text, encoding="utf-8")
    print(text)
    return out


def main(argv=None) -> int:
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true",
                        help="Chỉ in, không ghi reports/results/class_balance.*")
    args = parser.parse_args(argv)
    run(write=not args.no_write)
    return 0


if __name__ == "__main__":
    sys.exit(main())

