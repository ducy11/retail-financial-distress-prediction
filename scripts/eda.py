"""EDA cơ bản có nhận xét: tổng quan corpus + thống kê mô tả + phân phối + tương quan.

Lệnh: python -m scripts.eda [--no-write] [--no-figures]

Ghi ra:
- `reports/results/eda_summary.json`, `reports/results/eda.md` (mọi số đọc từ đây, không nhập tay).
- 9 hình ở `reports/figures/eda/`: 1 độ phủ, 2 cân bằng lớp, 3 nhãn theo công ty × quý, 4 boxplot,
  5 chuỗi thời gian, 6 độ dài lịch sử, **7 phân phối 14 tỷ số, 8 tương quan giữa 14 tỷ số,
  9 mức tách hai lớp của từng tỷ số**.

Trả lời 3 câu hỏi của phần "Khám phá & hiểu dữ liệu" ở mức đủ dùng cho báo cáo:
1. Mỗi biến phân bố thế nào, có giá trị dị biệt (ngoại lai) không? → `ratio_stats`,
   `indicator_stats`, hình 7, và nhận xét tự động.
2. Các lớp có lệch không (tính bằng %)? → bảng cân bằng lớp theo tập + nhận xét về thước đo.
3. Biến nào tương quan với nhau và với nhãn? → hình 8–9, bảng cặp gần trùng thông tin.
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from forecasting.config import (FIGURES_DIR, MIN_HISTORY_QUARTERS, RESULTS_DIR, ensure_dirs,
                                ensure_utf8_stdio)
from forecasting.data_loader import load_prepared
from forecasting.eda import (correlation_analysis, feature_statistics, indicator_value_matrix,
                             latest_ratio_matrix, markdown_table, outlier_driven_pairs,
                             target_association)
from forecasting.features import build_feature_matrix, extract_labels, feature_names

#: Thư mục hình EDA.
EDA_DIR = FIGURES_DIR / "eda"

#: Chỉ tiêu vẽ boxplot (chọn các tỷ số có ý nghĩa phân biệt nhất trong kiểm chứng).
BOXPLOT_RATIOS = ["current_ratio_latest", "working_capital_to_assets", "net_margin_latest",
                  "debt_to_assets_latest", "inventory_to_sales_latest", "ocf_to_sales_latest"]

#: Công ty minh hoạ cho biểu đồ chuỗi thời gian.
TIMESERIES_TICKERS = ["WMT", "HD", "ROST"]

#: Ngưỡng dùng cho phần nhận xét của EDA cơ bản.
STRONG_CORR = 0.80        #: |r| ≥ 0,80 ⇒ hai tỷ số gần như trùng thông tin ⇒ cân nhắc bỏ bớt.
CORR_GAP_WARN = 0.30      #: chênh Pearson − Spearman lớn ⇒ tương quan bị ngoại lai chi phối.
SKEW_WARN = 1.0           #: |skew| > 1 ⇒ lệch rõ rệt, nên xem lại trước khi chuẩn hoá.
OUTLIER_WARN_PCT = 5.0    #: > 5% mẫu nằm ngoài [Q1 − 1,5·IQR, Q3 + 1,5·IQR].


def _load(path: str):
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _load(path: Path) -> Any:
    """Đọc JSON UTF-8."""
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _retail_rows() -> Dict[str, List[Dict[str, Any]]]:
    """{ticker: rows} từ data/retail-expanded (chuỗi thời gian chỉ tiêu)."""
    root = Path(__file__).resolve().parents[1]
    out: Dict[str, List[Dict[str, Any]]] = {}
    for f in sorted((root / "data" / "retail-expanded").glob("*-16-indicators-vnd.json")):
        doc = _load(f)
        out[doc["ticker"]] = doc["rows"]
    return out


def _vnd_fields(rows: List[Dict[str, Any]]) -> List[str]:
    keys: set[str] = set()
    for r in rows:
        keys.update(k for k in r if k.endswith("_vnd"))
    return sorted(keys)


def _label_shares(splits: Dict[str, List[Dict[str, Any]]]) -> Dict[str, Any]:
    """Tỷ lệ nhãn = 1 theo công ty (bản không vẽ hình — dùng khi `figures=False`)."""
    by_ticker: Dict[str, List[int]] = defaultdict(list)
    for name, arr in splits.items():
        for sample in arr:
            by_ticker[str(sample["ticker"])].append(int(sample["is_distressed"]))
    share = {t: sum(v) / len(v) for t, v in sorted(by_ticker.items())}
    return {"label_share_by_ticker": share,
            "companies_all_one": sorted([t for t, v in share.items() if v == 1.0]),
            "companies_all_zero": sorted([t for t, v in share.items() if v == 0.0])}


def coverage_matrix(rows_by_ticker: Dict[str, List[Dict[str, Any]]]
                    ) -> Tuple[np.ndarray, List[str], List[str]]:
    """Ma trận độ phủ (% quý có giá trị) theo [công ty × chỉ tiêu]."""
    tickers = sorted(rows_by_ticker)
    fields = _vnd_fields([r for rows in rows_by_ticker.values() for r in rows])
    matrix = np.full((len(tickers), len(fields)), np.nan)
    for i, t in enumerate(tickers):
        rows = rows_by_ticker[t]
        for j, f in enumerate(fields):
            matrix[i, j] = 100.0 * sum(1 for r in rows if r.get(f) is not None) / len(rows)
    return matrix, tickers, fields


def fig_coverage(matrix: np.ndarray, tickers: List[str], fields: List[str], path: Path) -> None:
    """Hình 1 — heatmap độ phủ chỉ tiêu theo công ty (phát hiện chỉ tiêu gần như rỗng)."""
    fig, ax = plt.subplots(figsize=(11, 4.4))
    im = ax.imshow(matrix, aspect="auto", cmap="RdYlGn", vmin=0, vmax=100)
    ax.set_xticks(range(len(fields)), [f.replace("_vnd", "") for f in fields],
                  rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(tickers)), tickers)
    ax.set_title("Độ phủ chỉ tiêu theo công ty (% số quý có dữ liệu)")
    for i in range(len(tickers)):
        for j in range(len(fields)):
            if matrix[i, j] < 60:  # chỉ chú thích các ô thiếu nhiều để hình không rối
                ax.text(j, i, f"{matrix[i, j]:.0f}", ha="center", va="center", fontsize=7, color="black")
    fig.colorbar(im, ax=ax, label="% quý có giá trị")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_class_balance(splits: Dict[str, List[Dict[str, Any]]], path: Path) -> Dict[str, Any]:
    """Hình 2 — cân bằng lớp theo từng split (kèm tỷ lệ)."""
    names = ["train", "validation", "test", "purged"]
    pos = [sum(s["is_distressed"] for s in splits[n]) for n in names]
    neg = [len(splits[n]) - p for n, p in zip(names, pos)]
    x = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.bar(x, neg, label="Không suy giảm (0)", color="steelblue")
    ax.bar(x, pos, bottom=neg, label="Suy giảm (1)", color="crimson")
    for i, n in enumerate(names):
        total = neg[i] + pos[i]
        ax.text(i, total + 3, f"{pos[i]}/{total}\n({pos[i] / total:.0%})", ha="center", fontsize=9)
    ax.set_xticks(x, names)
    ax.set_ylabel("Số mẫu")
    ax.set_title("Cân bằng lớp theo tập dữ liệu")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {n: {"distress": p, "total": p + q} for n, p, q in zip(names, pos, neg)}


def fig_label_heatmap(splits: Dict[str, List[Dict[str, Any]]], path: Path) -> Dict[str, Any]:
    """Hình 3 — nhãn theo [công ty × kỳ target]: hình này phơi bày ngay bản chất nhãn.

    Nếu nhãn là sự kiện của QUÝ thì mỗi hàng sẽ đốm xanh/đỏ rải rác; nếu nhãn gần như là thuộc
    tính của CÔNG TY thì mỗi hàng gần như đồng màu — đây là kiểm chứng bắt buộc trước khi tin
    vào AUROC in-domain.
    """
    samples = [s for n in ("train", "validation", "test", "purged") for s in splits[n]]
    periods = sorted({s["request"]["target_period_end"] for s in samples})
    pidx = {p: i for i, p in enumerate(periods)}
    tickers = sorted({s["ticker"] for s in samples})
    tidx = {t: i for i, t in enumerate(tickers)}
    M = np.full((len(tickers), len(periods)), np.nan)
    for s in samples:
        M[tidx[s["ticker"]], pidx[s["request"]["target_period_end"]]] = s["is_distressed"]

    fig, ax = plt.subplots(figsize=(12, 3.6))
    cmap = plt.get_cmap("coolwarm").copy()
    cmap.set_bad("white")
    ax.imshow(np.ma.masked_invalid(M), aspect="auto", cmap=cmap, vmin=0, vmax=1)
    ax.set_yticks(range(len(tickers)), tickers)
    step = max(1, len(periods) // 12)
    ax.set_xticks(range(0, len(periods), step), [periods[i][:7] for i in range(0, len(periods), step)],
                  rotation=45, ha="right", fontsize=8)
    ax.set_title("Nhãn theo công ty × kỳ target (đỏ = suy giảm, xanh = không, trắng = không có mẫu)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

    share = {t: float(np.nanmean(M[tidx[t]])) for t in tickers}
    return {"label_share_by_ticker": share,
            "companies_all_one": sorted([t for t, v in share.items() if v == 1.0]),
            "companies_all_zero": sorted([t for t, v in share.items() if v == 0.0])}


def fig_ratio_boxplots(splits: Dict[str, List[Dict[str, Any]]], path: Path) -> Dict[str, Any]:
    """Hình 4 — boxplot 6 tỷ số theo nhãn: tỷ số nào TÁCH được hai lớp, tỷ số nào không?"""
    samples = [s for n in ("train", "validation", "test", "purged") for s in splits[n]]
    names = feature_names()
    X = build_feature_matrix(samples)
    y = extract_labels(samples)
    cols = [(c, names.index(c)) for c in BOXPLOT_RATIOS if c in names]

    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    overlap: Dict[str, float] = {}
    for ax, (c, j) in zip(axes.ravel(), cols):
        a = X[y == 0, j]
        b = X[y == 1, j]
        a, b = a[np.isfinite(a)], b[np.isfinite(b)]
        ax.boxplot([a, b], tick_labels=["0", "1"], showfliers=False)
        ax.set_title(c, fontsize=9)
        ax.grid(alpha=0.3, axis="y")
        lo, hi = (min(a.min(), b.min()), max(a.max(), b.max())) if len(a) and len(b) else (0, 1)
        overlap[c] = float(min(a.max(), b.max()) - max(a.min(), b.min())) / (hi - lo) if hi > lo else float("nan")
    axes[0, 0].set_ylabel("Giá trị")
    axes[1, 0].set_ylabel("Giá trị")
    for ax in axes[1]:
        ax.set_xlabel("Nhãn (0/1)")
    fig.suptitle("Phân bố tỷ số tài chính theo nhãn (toàn bộ 324 mẫu)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return {"boxplot_ratios": [c for c, _ in cols], "normalized_overlap": overlap}


def fig_timeseries(rows_by_ticker: Dict[str, List[Dict[str, Any]]], path: Path) -> None:
    """Hình 5 — chuỗi thời gian 4 chỉ tiêu của 3 công ty (kiểm tra tính liên tục/mùa vụ)."""
    picks = [t for t in TIMESERIES_TICKERS if t in rows_by_ticker]
    fields = ["revenue", "net_income", "operating_cash_flow", "inventory"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 6.5))
    for ax, f in zip(axes.ravel(), fields):
        for t in picks:
            rows = rows_by_ticker[t]
            xs = [r["period_end"] for r in rows]
            ys = [(float(r[f + "_vnd"]) / 1e12 if r.get(f + "_vnd") is not None else np.nan)
                  for r in rows]
            ax.plot(range(len(xs)), ys, marker="o", ms=3, lw=1, label=t)
        ax.set_title(f, fontsize=10)
        ax.set_xticks(range(0, len(xs), max(1, len(xs) // 8)),
                      [xs[i][:7] for i in range(0, len(xs), max(1, len(xs) // 8))],
                      rotation=45, ha="right", fontsize=7)
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
    axes[0, 0].set_ylabel("nghìn tỷ VND")
    axes[1, 0].set_ylabel("nghìn tỷ VND")
    fig.suptitle("Chuỗi thời gian chỉ tiêu (theo quý, đơn vị nghìn tỷ VND)")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def history_length_stats(lengths: Sequence[int], min_quarters: int = MIN_HISTORY_QUARTERS
                         ) -> Dict[str, Any]:
    """Thống kê độ dài lịch sử (tách khỏi phần vẽ hình để dùng được khi `figures=False`)."""
    lengths = list(lengths)
    n_short = sum(1 for h in lengths if h < min_quarters)
    return {"n_samples": len(lengths), "min": int(min(lengths)), "max": int(max(lengths)),
            "median": float(np.median(lengths)), "n_below_min": int(n_short),
            "share_below_min": float(n_short / len(lengths))}


def fig_history_length(splits: Dict[str, List[Dict[str, Any]]], path: Path,
                       min_quarters: int = MIN_HISTORY_QUARTERS) -> Dict[str, Any]:
    """Hình 6 — độ dài lịch sử: bao nhiêu mẫu quá non (YoY không tính được → phải impute)?"""
    lengths = [len(s["request"]["history"]) for n in ("train", "validation", "test", "purged")
               for s in splits[n]]
    stats = history_length_stats(lengths, min_quarters)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(lengths, bins=range(1, max(lengths) + 2), color="slateblue", alpha=0.8)
    ax.axvline(min_quarters - 0.5, color="crimson", ls="--",
               label=f"ngưỡng tối thiểu = {min_quarters} quý")
    ax.set_xlabel("Số quý lịch sử của mẫu")
    ax.set_ylabel("Số mẫu")
    ax.set_title("Phân bố độ dài lịch sử (trước as_of)")
    ax.legend()
    ax.grid(alpha=0.3, axis="y")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)
    return stats


def fig_ratio_distributions(X: np.ndarray, names: Sequence[str], path: Path,
                            outlier_pct: Optional[Dict[str, float]] = None) -> None:
    """Hình 7 — histogram của 14 tỷ số: thấy ngay độ lệch, đuôi dài, chỗ dồn cục.

    Mỗi ô ghi thêm % ngoại lai theo IQR để người đọc biết tỷ số nào cần xử lý trước khi chuẩn hoá.
    """
    columns = 4
    rows = int(math.ceil(len(names) / columns))
    fig, axes = plt.subplots(rows, columns, figsize=(3.3 * columns, 2.5 * rows))
    raveled = np.atleast_1d(axes).ravel()
    for ax, name, j in zip(raveled, names, range(len(names))):
        values = X[:, j]
        values = values[np.isfinite(values)]
        ax.hist(values, bins=30, color="steelblue", alpha=0.85)
        ax.set_title(str(name).replace("_latest", ""), fontsize=9)
        share = (outlier_pct or {}).get(str(name))
        if share is not None:
            ax.set_xlabel(f"ngoại lai IQR: {share:.0f}% mẫu", fontsize=7)
        ax.grid(alpha=0.25)
        ax.tick_params(labelsize=7)
    for ax in raveled[len(names):]:
        ax.axis("off")
    fig.suptitle("Phân phối 14 tỷ số tài chính (trục tung = số mẫu, 324 mẫu)", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_ratio_correlation(corr: np.ndarray, names: Sequence[str], path: Path) -> None:
    """Hình 8 — heatmap tương quan Pearson giữa 14 tỷ số (chỉ ghi số khi |r| ≥ 0,5)."""
    short = [str(n).replace("_latest", "") for n in names]
    fig, ax = plt.subplots(figsize=(9.5, 8))
    image = ax.imshow(corr, cmap="coolwarm", vmin=-1, vmax=1)
    ax.set_xticks(range(len(names)), short, rotation=60, ha="right", fontsize=8)
    ax.set_yticks(range(len(names)), short, fontsize=8)
    for i in range(len(names)):
        for j in range(len(names)):
            value = corr[i, j]
            if i != j and math.isfinite(value) and abs(value) >= 0.5:
                ax.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if abs(value) > 0.75 else "black")
    ax.set_title(f"Tương quan Pearson giữa 14 tỷ số — ô ghi số là |r| ≥ 0,5\n"
                 f"(tương quan cặp, dùng giá trị hữu hạn)")
    fig.colorbar(image, ax=ax, shrink=0.7)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fig_ratio_vs_label(ranked: Sequence[Dict[str, Any]], path: Path) -> None:
    """Hình 9 — mức tách hai lớp của từng tỷ số: 2·AUC − 1 (đỏ = giá trị cao ⇒ dễ suy giảm)."""
    rows = [r for r in ranked if r.get("auc") is not None]
    labels = [str(r["feature"]).replace("_latest", "") for r in rows][::-1]
    effects = [(r.get("effect_rank_biserial") or 0.0) for r in rows][::-1]
    colors = ["crimson" if e > 0 else "steelblue" for e in effects]
    fig, ax = plt.subplots(figsize=(8.8, 0.42 * max(1, len(labels)) + 1.9))
    ax.barh(labels, effects, color=colors, alpha=0.85)
    ax.axvline(0, color="k", lw=1)
    ax.set_xlabel("2·AUC − 1  (đỏ: giá trị cao ⇒ nhãn 1; xanh: giá trị thấp ⇒ nhãn 1)")
    ax.set_title("Mức tách hai lớp của từng tỷ số (tính trên train+validation)", fontsize=10)
    ax.grid(alpha=0.3, axis="x")
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def basic_conclusions(ratio_stats: Sequence[Dict[str, Any]], indicator_stats: Sequence[Dict[str, Any]],
                      balance: Dict[str, Any], label_info: Dict[str, Any],
                      correlation: Dict[str, Any],
                      association: Sequence[Dict[str, Any]]) -> List[str]:
    """Nhận xét tự động cho EDA cơ bản — mỗi câu nêu số liệu + việc cần làm tiếp.

    Vì sao cần: bảng số và biểu đồ chỉ *cho thấy* dữ liệu; nhận xét mới *kết luận* được điều gì
    ảnh hưởng tới bước sau (xử lý ngoại lai, chọn thước đo, bỏ bớt biến trùng, cảnh báo rò rỉ).
    """
    lines: List[str] = []

    # (1) Phân phối & ngoại lai
    total_ratios = max(1, len(ratio_stats))
    if ratio_stats:
        skewed = [r for r in ratio_stats if r.get("skew") is not None and abs(r["skew"]) > SKEW_WARN]
        heavy = sorted((r for r in ratio_stats if r.get("skew") is not None),
                       key=lambda r: -abs(r["skew"]))[:3]
        outliers = [r for r in ratio_stats if (r.get("iqr_outlier_pct") or 0) > OUTLIER_WARN_PCT]
        lines.append(
            f"**Phân phối & ngoại lai:** {len(skewed)}/{total_ratios} tỷ số lệch rõ (|skew| > "
            f"{SKEW_WARN:.0f}), lệch nhất: "
            + ", ".join(f"{r['feature'].replace('_latest', '')} (skew {r['skew']:.2f}, "
                        f"max {r['max']:.2f})" for r in heavy)
            + f"; {len(outliers)}/{total_ratios} tỷ số có hơn {OUTLIER_WARN_PCT:.0f}% giá trị ngoại "
            f"lai theo IQR. ⇒ Trước khi huấn luyện nên **clip/winsorize** các tỷ số này (mô hình tuyến "
            f"tính bị outlier kéo mạnh; mô hình cây ít bị hơn nhưng vẫn nên xem lại giá trị cực trị).")
    missing_heavy = [r for r in ratio_stats if (r.get("missing_pct") or 0) > 20]
    if missing_heavy:
        lines.append(
            f"**Giá trị thiếu (theo từng tỷ số):** {len(missing_heavy)} tỷ số thiếu hơn 20% — "
            + ", ".join(f"{r['feature'].replace('_latest', '')} ({r['missing_pct']:.0f}%)"
                        for r in sorted(missing_heavy, key=lambda r: -r["missing_pct"])[:5])
            + ". ⇒ Phần thiếu được `SimpleImputer(median)` xử lý **trong pipeline**; riêng các tỷ số "
              "dựa trên `receivables`/`short_term_investments` chỉ có nghĩa với một phần mẫu.")

    # (2) Tỉ lệ lớp (theo %)
    totals = sum(int(v.get("total", 0)) for v in balance.values())
    positives = sum(int(v.get("distress", 0)) for v in balance.values())
    if totals:
        per_split = ", ".join(f"{name} {100.0 * v['distress'] / max(1, v['total']):.1f}%"
                              for name, v in balance.items())
        majority = max(positives, totals - positives) / totals
        lines.append(
            f"**Tỉ lệ lớp (bằng %):** lớp suy giảm chiếm {100.0 * positives / totals:.1f}% toàn corpus; "
            f"theo tập: {per_split}. ⇒ Chỉ đoán lớp đa số đã đạt {100.0 * majority:.1f}% ⇒ **KHÔNG dùng "
            f"Accuracy** làm thước đo chính; dùng Precision/Recall/**F1 & macro-F1**/PR-AUC (AP)/MCC "
            f"và nêu ngưỡng quyết định.")
    share_by_ticker = label_info.get("label_share_by_ticker", {})
    if share_by_ticker:
        worst = sorted(share_by_ticker.items(), key=lambda kv: -(kv[1] if kv[1] < 0.5 else 1 - kv[1]))[:2]
        lines.append(
            f"**Lệch lớp ở cấp công ty (quan trọng hơn cấp mẫu):** "
            f"{', '.join(label_info.get('companies_all_one', [])) or '—'} có **100% nhãn = 1** trong mọi "
            f"quý; lệch ngược lại: " + ", ".join(f"{t} chỉ {s:.1%} nhãn 1" for t, s in worst)
            + ". ⇒ Nhãn gần như là thuộc tính của công ty, nên phải chia tập/CV **theo nhóm công ty** "
              "(GroupKFold/LOCO) và luôn so với baseline `ticker_prior`, nếu không AUROC in-domain sẽ "
              "bị thổi phồng.")

    # (3) Tương quan giữa các biến
    clusters = correlation.get("clusters_abs_ge_threshold", [])
    if correlation.get("feature_order"):
        lines.append(
            f"**Tương quan giữa các tỷ số:** {correlation.get('n_pairs_abs_ge_threshold')} cặp có |r| ≥ "
            f"{correlation.get('threshold')} (gần như trùng thông tin), gom thành {len(clusters)} cụm: "
            + "; ".join(", ".join(str(n).replace("_latest", "") for n in cluster)
                        for cluster in clusters[:3])
            + ". ⇒ Nên **giữ 1 đại diện mỗi cụm** hoặc dùng mô hình có regularization để tránh hệ số "
              "bất ổn.")
        driven = outlier_driven_pairs(correlation, CORR_GAP_WARN)
        if driven:
            lines.append(
                f"**Cảnh báo tương quan giả do ngoại lai:** {len(driven)} cặp có tương quan Pearson "
                f"lệch xa Spearman hơn {CORR_GAP_WARN:.2f} — "
                + ", ".join(f"{p['a'].replace('_latest', '')}↔{p['b'].replace('_latest', '')} "
                            f"(r {p['pearson']} vs hạng {p['spearman']})" for p in driven[:3])
                + ". ⇒ Các cặp này chỉ 'liên quan' ở vài quan sát dị biệt; phải xử lý ngoại lai "
                  "trước khi kết luận về quan hệ giữa chúng.")
        else:
            lines.append(
                f"**Kiểm tra tương quan giả:** không cặp nào có Pearson lệch Spearman hơn "
                f"{CORR_GAP_WARN:.2f} ⇒ các tương quan mạnh ở trên không bị ngoại lai chi phối.")

    # (4) Tương quan với nhãn
    top = [a for a in association if a.get("auc") is not None][:5]
    if top:
        lines.append(
            "**Tỷ số tách hai lớp tốt nhất (train+validation, xếp theo |2·AUC−1|):** "
            + ", ".join(f"{a['feature'].replace('_latest', '')} (2·AUC−1 = "
                        f"{a['effect_rank_biserial']:+.2f} ⇒ {a['direction']})" for a in top)
            + ". ⇒ Đây là các biến nên giữ lại khi lọc đặc trưng; lưu ý nhóm thanh khoản/cấu trúc vốn "
              "gần như không đổi theo quý nên phần lớn sức mạnh dự báo in-domain đến từ việc phân biệt "
              "**công ty**, không phải từ động lực suy giảm của từng quý.")
    strong_indicators = [r for r in indicator_stats
                         if (r.get("iqr_outlier_pct") or 0) > OUTLIER_WARN_PCT]
    if strong_indicators:
        lines.append(
            f"**Ngoại lai ở dữ liệu gốc:** {len(strong_indicators)}/16 chỉ tiêu tiền có hơn "
            f"{OUTLIER_WARN_PCT:.0f}% quý nằm ngoài khoảng IQR — đây thường là khác biệt quy mô giữa "
            f"các chuỗi bán lẻ (WMT lớn hơn FIVE nhiều lần), nên khi so sánh giữa công ty phải dùng "
            f"**tỷ số** (đã có) thay vì giá trị tuyệt đối.")
    return lines


def run(write: bool = True, out_dir: Optional[Path] = None, fig_dir: Optional[Path] = None,
        figures: bool = True) -> Dict[str, Any]:
    """Sinh hình + bảng EDA cơ bản → `reports/figures/eda/` và `reports/results/eda.{json,md}`.

    Đây là phần "Mô tả & phân tích dữ liệu" của báo cáo: mọi con số lấy từ `eda_summary.json`
    (không nhập tay). `out_dir`/`fig_dir` cho phép test ghi vào thư mục tạm.

    Phạm vi tập dữ liệu (ghi rõ để tránh hiểu sai):
    - Thống kê mô tả / phân phối / tương quan giữa các tỷ số: **toàn bộ 324 mẫu** (mô tả dữ liệu).
    - Tương quan tỷ số ↔ nhãn: **train+validation** — không dùng test cho bất kỳ mô tả nào có thể
      dẫn tới quyết định.
    """
    ensure_dirs()
    ensure_utf8_stdio()  # hàm này cũng được gọi từ test/notebook ⇒ không phụ thuộc CLI
    results_dir = Path(out_dir) if out_dir else RESULTS_DIR
    target_fig_dir = Path(fig_dir) if fig_dir else EDA_DIR
    results_dir.mkdir(parents=True, exist_ok=True)
    target_fig_dir.mkdir(parents=True, exist_ok=True)

    rows_by_ticker = _retail_rows()
    splits = {n: load_prepared(n) for n in ("train", "validation", "test", "purged")}
    pooled = [sample for name in ("train", "validation", "test", "purged") for sample in splits[name]]
    analysis_samples = splits["train"] + splits["validation"]

    # --- 1) Thống kê mô tả: 14 tỷ số (đơn vị so sánh được) và 16 chỉ tiêu tiền (nghìn tỷ VND) ---
    X_ratio, ratio_names, _ = latest_ratio_matrix(pooled)
    ratio_stats = feature_statistics(X_ratio, ratio_names)
    X_indicator, indicator_names = indicator_value_matrix(rows_by_ticker)
    indicator_stats = feature_statistics(X_indicator, indicator_names)
    outlier_by_ratio = {str(r["feature"]): (r.get("iqr_outlier_pct") or 0.0) for r in ratio_stats}

    # --- 2) Tương quan giữa các tỷ số + tương quan với nhãn (train+validation) ---
    X_analysis, _, y_analysis = latest_ratio_matrix(analysis_samples)
    correlation = correlation_analysis(X_analysis, ratio_names, threshold=STRONG_CORR)
    association = target_association(X_analysis, y_analysis, ratio_names)

    matrix, tickers, fields = coverage_matrix(rows_by_ticker)
    if figures:
        fig_coverage(matrix, tickers, fields, target_fig_dir / "01_coverage_by_company.png")
    balance = fig_class_balance(splits, target_fig_dir / "02_class_balance.png") if figures \
        else {n: {"distress": sum(int(s["is_distressed"]) for s in arr), "total": len(arr)}
              for n, arr in splits.items()}
    label_info = fig_label_heatmap(splits, target_fig_dir / "03_label_by_company_quarter.png") \
        if figures else _label_shares(splits)
    box = fig_ratio_boxplots(splits, target_fig_dir / "04_ratio_boxplots_by_label.png") \
        if figures else {"boxplot_ratios": [], "normalized_overlap": {}}
    if figures:
        fig_timeseries(rows_by_ticker, target_fig_dir / "05_timeseries_indicators.png")
    hist = fig_history_length(splits, target_fig_dir / "06_history_length.png") if figures \
        else history_length_stats([len(s["request"]["history"]) for s in pooled])
    if figures:
        fig_ratio_distributions(X_ratio, ratio_names,
                                target_fig_dir / "07_ratio_distributions.png", outlier_by_ratio)
        pearson = np.array([[np.nan if v is None else float(v) for v in row]
                            for row in correlation["pearson_matrix"]])
        fig_ratio_correlation(pearson, ratio_names, target_fig_dir / "08_ratio_correlation.png")
        fig_ratio_vs_label(association["ranked_by_effect"], target_fig_dir / "09_ratio_vs_label.png")

    cov_min = {f.replace("_vnd", ""): float(np.nanmin(matrix[:, j])) for j, f in enumerate(fields)}
    cov_mean = {f.replace("_vnd", ""): float(np.nanmean(matrix[:, j])) for j, f in enumerate(fields)}
    conclusions = basic_conclusions(ratio_stats, indicator_stats, balance, label_info,
                                    correlation, association["ranked_by_effect"])
    figure_paths: List[str] = []
    for path in sorted(target_fig_dir.glob("*.png")):
        try:
            figure_paths.append(str(path.relative_to(FIGURES_DIR.parent)))
        except ValueError:  # fig_dir ngoài cây repo (test)
            figure_paths.append(str(path))
    summary: Dict[str, Any] = {
        "corpus": {
            "n_companies": len(rows_by_ticker),
            "n_quarters": int(sum(len(v) for v in rows_by_ticker.values())),
            "n_samples": sum(len(splits[n]) for n in splits),
            "quarters_per_company": {t: len(v) for t, v in sorted(rows_by_ticker.items())},
            "fields": [f.replace("_vnd", "") for f in fields],
        },
        "splits": balance,
        "coverage_pct": {"min_by_field": cov_min, "mean_by_field": cov_mean},
        "low_coverage_fields": sorted([f for f, v in cov_min.items() if v < 50],
                                      key=lambda f: cov_min[f]),
        "label": label_info,
        "history_length": hist,
        "ratio_boxplots": box,
        "unit_note": ("tỷ số: không đơn vị; chỉ tiêu tiền: nghìn tỷ VND (chia 1e12); "
                      "tương quan với nhãn tính trên train+validation"),
        "ratio_stats": ratio_stats,
        "indicator_stats": indicator_stats,
        "ratio_correlation": {**{key: correlation[key] for key in
                                 ("threshold", "n_pairs_abs_ge_threshold", "strongest_pairs",
                                  "clusters_abs_ge_threshold", "n_features_in_clusters",
                                  "effective_dimensionality")},
                              "outlier_driven_pairs": outlier_driven_pairs(correlation,
                                                                           CORR_GAP_WARN)},
        "ratio_vs_label": association["ranked_by_effect"],
        "conclusions": conclusions,
        "figures": figure_paths,
    }
    if write:
        (results_dir / "eda_summary.json").write_text(
            json.dumps(summary, ensure_ascii=False, indent=2, default=float), encoding="utf-8")

    lines = ["# EDA — dữ liệu bán lẻ Mỹ (SEC XBRL)", ""]
    lines += [f"- Corpus: **{summary['corpus']['n_companies']} công ty, "
              f"{summary['corpus']['n_quarters']} quý, {summary['corpus']['n_samples']} mẫu**",
              f"- Công ty có nhãn 1 toàn bộ: **{', '.join(label_info['companies_all_one']) or '—'}**",
              f"- Công ty có nhãn 0 toàn bộ: **{', '.join(label_info['companies_all_zero']) or '—'}**",
              f"- Mẫu có < {MIN_HISTORY_QUARTERS} quý lịch sử: **{hist['n_below_min']}/"
              f"{hist['n_samples']} ({hist['share_below_min']:.0%})**",
              f"- Chỉ tiêu độ phủ < 50% (theo công ty): **{', '.join(summary['low_coverage_fields']) or '—'}**",
              ""]
    lines += ["## Số quý mỗi công ty", "", "| Công ty | Số quý |", "|---|---:|"]
    lines += [f"| {t} | {n} |" for t, n in summary["corpus"]["quarters_per_company"].items()]
    lines += ["", "## Cân bằng lớp theo tập (đếm và %)", "",
              "| Tập | Suy giảm (1) | Không (0) | Tổng | % lớp 1 | % lớp 0 |",
              "|---|---:|---:|---:|---:|---:|"]
    lines += [f"| {n} | {v['distress']} | {v['total'] - v['distress']} | {v['total']} | "
              f"{100.0 * v['distress'] / max(1, v['total']):.1f}% | "
              f"{100.0 * (v['total'] - v['distress']) / max(1, v['total']):.1f}% |"
              for n, v in balance.items()]
    lines += ["", "> Đọc bảng: lớp 1 chiếm đa số ở mọi tập ⇒ **không dùng Accuracy**; theo dõi "
                  "F1/macro-F1, PR-AUC (AP) và MCC (chi tiết ở mục đánh giá của báo cáo)."]
    lines += ["", "## Tỷ lệ nhãn theo công ty (phát hiện quan trọng)", "",
              "| Công ty | Tỷ lệ nhãn = 1 |", "|---|---:|"]
    lines += [f"| {t} | {v:.2f} |" for t, v in sorted(label_info["label_share_by_ticker"].items(),
                                                       key=lambda kv: -kv[1])]
    lines += ["", "## Độ phủ chỉ tiêu (min–mean theo công ty)", "",
              "| Chỉ tiêu | Min % | Mean % |", "|---|---:|---:|"]
    lines += [f"| {f} | {cov_min[f]:.1f} | {cov_mean[f]:.1f} |"
              for f in sorted(cov_min, key=lambda k: cov_min[k])]
    lines += ["", "## Thống kê mô tả 14 tỷ số (giá trị quý gần nhất; 324 mẫu; không đơn vị)", "",
              markdown_table(
                  ["Tỷ số", "Thiếu %", "Nhỏ nhất", "Q1", "Trung vị", "Trung bình", "Q3",
                   "Lớn nhất", "Skew", "Ngoại lai IQR %"],
                  [[str(r["feature"]).replace("_latest", ""), f"{r.get('missing_pct') or 0:.1f}",
                    f"{r.get('min')}", f"{r.get('q1')}", f"{r.get('median')}", f"{r.get('mean')}",
                    f"{r.get('q3')}", f"{r.get('max')}", f"{r.get('skew')}",
                    f"{r.get('iqr_outlier_pct') or 0:.1f}"]
                   for r in sorted(ratio_stats, key=lambda r: -(r.get("iqr_outlier_pct") or 0))],
                  ["---", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:", "---:"]),
              "",
              "> Đọc bảng: cột **Ngoại lai IQR %** = % mẫu nằm ngoài [Q1 − 1,5·IQR, Q3 + 1,5·IQR]; "
              "cột **Skew** = độ lệch (|skew| > 1 là lệch rõ). Tỷ số vừa lệch vừa nhiều ngoại lai "
              "(thường là các tỷ số có mẫu số nhỏ như `debt_to_equity`) nên được **clip/winsorize** "
              "trước khi huấn luyện mô hình tuyến tính."]

    lines += ["", "## Thống kê mô tả 16 chỉ tiêu gốc (nghìn tỷ VND; mọi công ty × mọi quý)", "",
              markdown_table(
                  ["Chỉ tiêu", "Thiếu %", "Nhỏ nhất", "Trung vị", "Trung bình", "Lớn nhất",
                   "Ngoại lai IQR %"],
                  [[r["feature"], f"{r.get('missing_pct') or 0:.1f}", f"{r.get('min')}",
                    f"{r.get('median')}", f"{r.get('mean')}", f"{r.get('max')}",
                    f"{r.get('iqr_outlier_pct') or 0:.1f}"]
                   for r in sorted(indicator_stats, key=lambda r: -(r.get("missing_pct") or 0))],
                  ["---", "---:", "---:", "---:", "---:", "---:", "---:"]),
              "",
              "> Đọc bảng: chênh lệch **quy mô** giữa các công ty rất lớn (WMT lớn hơn FIVE nhiều lần) "
              "nên không so sánh giá trị tuyệt đối giữa công ty; mô hình dùng **tỷ số** (bảng trên). "
              "Chỉ tiêu thiếu nhiều là hạn chế của dữ liệu gốc SEC, đã nêu ở phần độ phủ."]

    correlation_rows = [[f"{p['a'].replace('_latest', '')} ↔ {p['b'].replace('_latest', '')}",
                         f"{p['pearson']}", f"{p['spearman']}"]
                        for p in correlation["strongest_pairs"][:8]]
    cluster_text = "; ".join(", ".join(str(n).replace("_latest", "") for n in c)
                             for c in correlation["clusters_abs_ge_threshold"][:4]) or "—"
    lines += ["", "## Tương quan giữa các tỷ số (Pearson / Spearman)", "",
              f"- **{correlation['n_pairs_abs_ge_threshold']} cặp** có |r| ≥ {correlation['threshold']} "
              f"⇒ gần như cùng một thông tin, gom thành "
              f"**{len(correlation['clusters_abs_ge_threshold'])} cụm**: {cluster_text}.",
              f"- Số chiều hiệu dụng của 14 tỷ số: "
              f"**{correlation['effective_dimensionality'].get('participation_ratio')}** "
              f"(cần {correlation['effective_dimensionality'].get('n_components_for_95pct_variance')} "
              f"thành phần cho 95% phương sai).",
              "", markdown_table(["Cặp tỷ số tương quan mạnh nhất", "Pearson", "Spearman"],
                                correlation_rows, ["---", "---:", "---:"]),
              "",
              *(["**Cặp bị ngoại lai chi phối** (Pearson lệch Spearman > "
                 f"{CORR_GAP_WARN:.2f}): " +
                 ", ".join(f"{p['a'].replace('_latest', '')} ↔ {p['b'].replace('_latest', '')} "
                           f"(r {p['pearson']} vs hạng {p['spearman']})"
                           for p in outlier_driven_pairs(correlation, CORR_GAP_WARN)[:5]), ""]
                if outlier_driven_pairs(correlation, CORR_GAP_WARN) else []),
              "> Đọc bảng: cột **Spearman** là tương quan theo hạng — nếu hai cột chênh nhau nhiều "
              "(> 0,3) thì tương quan chủ yếu do **ngoại lai** tạo ra, cần xử lý ngoại lai trước khi "
              "kết luận. Với các cặp |r| cao, nên **giữ 1 đại diện mỗi cụm** hoặc dùng "
              "regularization (`C`, `l2_regularization`) để tránh hệ số bất ổn."]

    label_rows = [[str(a["feature"]).replace("_latest", ""), f"{a.get('auc')}",
                   f"{a.get('direction')}", f"{a.get('point_biserial_r')}", f"{a.get('q_value_bh')}",
                   f"{a.get('mutual_information')}"]
                  for a in association["ranked_by_effect"]]
    lines += ["", "## Tương quan giữa tỷ số và NHÃN (train+validation)", "",
              markdown_table(["Tỷ số", "AUC 1-tỷ số", "Hướng", "r (point-biserial)", "q-value BH",
                              "Mutual information"],
                             label_rows, ["---", "---:", "---", "---:", "---:", "---:"]),
              "",
              f"> Đọc bảng: **AUC 1-tỷ số** là khả năng tự tách hai lớp của tỷ số đó; "
              f"`|2·AUC−1|` càng lớn càng tách tốt, hướng cho biết giá trị cao hay thấp ứng với suy "
              f"giảm. Cột **q-value BH** đã hiệu chỉnh cho 14 phép kiểm định (q < 0,05 mới coi là "
              f"chắc chắn). {association['n_effect_ge_0_30']} tỷ số có |2·AUC−1| ≥ 0,30; "
              f"{association['n_significant_after_bh_5pct']} tỷ số còn ý nghĩa sau hiệu chỉnh."]

    lines += ["", "## Nhận xét (sinh tự động từ số liệu trên)", ""]
    lines += [f"{index + 1}. {text}" for index, text in enumerate(conclusions)]
    lines += ["", "## Hình", ""] + [f"- `{p}`" for p in summary["figures"]]
    if write:
        (results_dir / "eda.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"EDA: {summary['corpus']['n_companies']} công ty / "
          f"{summary['corpus']['n_quarters']} quý / {summary['corpus']['n_samples']} mẫu; "
          f"{len(ratio_stats)} tỷ số mô tả, {correlation['n_pairs_abs_ge_threshold']} cặp tương quan mạnh; "
          f"{len(conclusions)} nhận xét; hình ở {target_fig_dir.name}/; "
          f"nhãn 1 toàn bộ ở {label_info['companies_all_one']}")
    return summary


def main(argv=None) -> int:
    """CLI: `python -m scripts.eda [--no-write] [--no-figures]`."""
    ensure_utf8_stdio()
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-write", action="store_true",
                        help="Chỉ in, không ghi reports/results/eda.{json,md}")
    parser.add_argument("--no-figures", action="store_true",
                        help="Bỏ vẽ hình (nhanh hơn, chỉ cần bảng số)")
    args = parser.parse_args(argv)
    print("=== EDA cơ bản: hình + bảng tổng quan + thống kê mô tả + tương quan ===")
    run(write=not args.no_write, figures=not args.no_figures)
    return 0


if __name__ == "__main__":
    sys.exit(main())
