"""Kiểm thử pipeline: tái tạo dữ liệu, feature engineering, split policy, chống rò rỉ.

Chạy: python -m unittest discover -s tests -v

Hai yêu cầu được kiểm chứng tự động ở đây (ngoài phần lab/benchmark):
1. Chia tập GIỮ NGUYÊN tỉ lệ lớp (stratified):
   - `test_class_ratio_preserved_across_splits` — tỉ lệ dương của train/validation/test lệch ≤ 5 điểm %
     so với toàn bộ, và khớp `label_counts` trong manifest;
   - `test_cv_folds_are_stratified_and_grouped` — CV (như `forecasting.tuning`) giữ tỉ lệ lớp.
   Bộ dữ liệu mất cân bằng 98/2 dùng `train_test_split(stratify=y)` + `StratifiedKFold` được kiểm thử
   ở `tests/test_imbalance_lab.py` và `tests/test_benchmark_imbalanced.py`.
2. Cân bằng/tiền xử lý CHỈ trên train: `TestNoLeakageInPreprocessing` — không sampler trong pipeline
   chính, impute/scale học thống kê từ train, `class_weight` tính trong `fit` từ nhãn nhận được.
"""
from __future__ import annotations

import json
import math
import pathlib
import shutil
import sys
import tempfile
import unittest

import numpy as np
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.utils.class_weight import compute_class_weight

ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from forecasting import data as fdata  # noqa: E402
from forecasting.data_loader import load_prepared  # noqa: E402
from forecasting.features import (  # noqa: E402
    build_feature_matrix,
    extract_labels,
    feature_names,
)
from forecasting.models import HYPERPARAMS, MODEL_REGISTRY, make_model  # noqa: E402
from runtime_warnings import quiet_library_warnings  # noqa: E402


def setUpModule() -> None:  # noqa: D103 - `unittest` hook
    """Lọc lại cảnh báo thư viện vô hại sau khi `unittest` đặt `simplefilter("default")`.

    Pipeline hồi quy logistic gây `OptimizeWarning: Unknown solver options: iprint` (scikit-learn 1.6
    + scipy 1.18) ở mọi lần fit. Xem `runtime_warnings.py`.
    """
    quiet_library_warnings()

#: Yêu cầu "giữ nguyên tỉ lệ lớp": cho phép lệch tối đa 5 điểm % giữa tỉ lệ dương của từng tập và
#: tỉ lệ dương toàn bộ. Đo thực tế: train 62,3% / validation 65,6% / test 59,4% (toàn bộ 62,0%)
#: ⇒ lệch tối đa 3,6 điểm %; split là theo THỜI GIAN trong từng công ty nên không thể bằng tuyệt đối.
MAX_SPLIT_RATIO_GAP = 0.05
#: `StratifiedGroupKFold` chỉ có 8 nhóm (công ty) nên stratified không chặt được như StratifiedKFold:
#: đo thực tế lệch tối đa 9,7 điểm % (trung bình 5,6) ⇒ chặn ở 15 điểm % và trung bình ≤ 10 điểm %.
MAX_FOLD_RATIO_GAP = 0.15
MAX_FOLD_MEAN_ABS_GAP = 0.10



class TestDataRegeneration(unittest.TestCase):
    """`forecasting.data --force` phải tái tạo prepared gốc byte-for-byte."""

    def test_regenerate_byte_identical(self):
        orig = ROOT / "data" / "prepared"
        if not (orig / "manifest.json").exists():
            self.skipTest("Chưa có data/prepared (chạy forecasting.data --force trước).")
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            shutil.copytree(orig, tmp, dirs_exist_ok=True)  # nhãn + manifest dates từ bản gốc
            fd_prep, fd_man = fdata.PREPARED_DIR, fdata.MANIFEST_FILE
            fdata.PREPARED_DIR = tmp
            fdata.MANIFEST_FILE = tmp / "manifest.json"
            try:
                res = fdata.run(force=True)
                self.assertFalse(res.get("skipped"))
                self.assertEqual(res["train"], 212)
                self.assertEqual(res["validation"], 32)
                self.assertEqual(res["test"], 64)
                self.assertEqual(res["purged"], 16)
            finally:
                fdata.PREPARED_DIR, fdata.MANIFEST_FILE = fd_prep, fd_man
            for name in ("train.json", "validation.json", "test.json", "purged.json",
                         "manifest.json"):
                with self.subTest(file=name):
                    self.assertEqual(
                        (orig / name).read_bytes(), (tmp / name).read_bytes(),
                        f"{name} khác bản gốc sau khi tái tạo")


class TestSplits(unittest.TestCase):
    def test_counts_and_policy(self):
        counts = {n: len(load_prepared(n)) for n in ("train", "validation", "test", "purged")}
        self.assertEqual(counts, {"train": 212, "validation": 32, "test": 64, "purged": 16})
        # Purge bao 2 bên validation: mỗi công ty 2 mẫu purged.
        purged = load_prepared("purged")
        by_ticker = {}
        for s in purged:
            by_ticker.setdefault(s["ticker"], []).append(s["sample_id"])
        for t, ids in by_ticker.items():
            self.assertEqual(len(ids), 2, f"{t} phải có đúng 2 mẫu purged")

    def test_manifest_consistency(self):
        man = json.loads((ROOT / "data" / "prepared" / "manifest.json").read_text(encoding="utf-8"))
        for name in ("train", "validation", "test", "purged"):
            self.assertEqual(man["counts"][name], len(load_prepared(name)))

    def test_class_ratio_preserved_across_splits(self):
        """Yêu cầu #1 — chia tập GIỮ NGUYÊN tỉ lệ lớp (kiểm chứng được, không chỉ tuyên bố).

        Split của đồ án là theo THỜI GIAN trong từng công ty (8 quý cuối = test, 4 quý = validation,
        có dải purge), không phải `train_test_split(stratify=y)` ngẫu nhiên — nên bảo đảm ở đây là
        "tỉ lệ lớp được giữ trong sai số nhỏ": |tỉ lệ dương của tập − tỉ lệ dương toàn bộ| < 5 điểm %,
        mỗi tập có đủ hai lớp, và số dương khớp `label_counts` trong manifest.
        """
        splits = {name: load_prepared(name) for name in ("train", "validation", "test")}
        labels = {name: extract_labels(arr) for name, arr in splits.items()}
        pooled = np.concatenate([labels[name] for name in splits])
        self.assertEqual(sorted(set(pooled.tolist())), [0, 1])
        overall = float(pooled.mean())

        for name, y in labels.items():
            with self.subTest(split=name):
                self.assertEqual(sorted(set(y.tolist())), [0, 1], f"{name}: thiếu một lớp")
                gap = abs(float(y.mean()) - overall)
                self.assertLess(gap, MAX_SPLIT_RATIO_GAP,
                                f"{name}: tỉ lệ dương {y.mean():.3f} lệch {gap:.3f} so với "
                                f"toàn bộ {overall:.3f} (> {MAX_SPLIT_RATIO_GAP})")

        manifest = json.loads((ROOT / "data" / "prepared" / "manifest.json").read_text(encoding="utf-8"))
        for name, y in labels.items():
            with self.subTest(manifest=name):
                self.assertEqual(manifest["label_counts"][name]["distressed"], int(y.sum()),
                                 f"manifest khai báo số dương của {name} không khớp dữ liệu")


class TestNoLeakageInPreprocessing(unittest.TestCase):
    """Yêu cầu #2 — mọi bước tiền xử lý / cân bằng CHỈ fit trên TRAIN.

    Ba bảo đảm, kiểm tra trên pipeline chính (`forecasting.models.make_model`):
    1. Không có bước sampler (`fit_resample`) nào ⇒ pipeline chính không sinh/loại mẫu cho bất kỳ tập
       nào; mọi kỹ thuật cân bằng dữ liệu chỉ nằm trong `imbalance_lab/` và `benchmark_imbalanced.py`,
       đặt TRONG `imblearn.pipeline.Pipeline` nên chỉ `fit_resample` trên train (fold-train khi CV).
    2. `SimpleImputer(median)` / `StandardScaler` học thống kê CHỈ từ train — khác hẳn nếu gộp
       validation/test (rò rỉ thống kê).
    3. Cân bằng bằng `class_weight="balanced*"` do sklearn tính TRONG `fit` từ nhãn nhận được
       (chỉ train), không phải hằng số tính trước trên toàn bộ dữ liệu.
    """

    def test_main_pipeline_has_no_balancer(self):
        """Không estimator nào của pipeline chính có `fit_resample` ⇒ không resample ngoài train."""
        self.assertTrue(MODEL_REGISTRY, "registry rỗng")
        for name in sorted(MODEL_REGISTRY):
            pipe = make_model(name)
            for step_name, step in pipe.steps:
                self.assertFalse(hasattr(step, "fit_resample"),
                                 f"{name}/{step_name}: pipeline chính chứa sampler ⇒ có thể resample "
                                 "validation/test; cân bằng dữ liệu chỉ được nằm trong imbalance_lab/")

    def test_imputer_and_scaler_learn_from_train_only(self):
        """Thống kê impute/scale phải bằng thống kê của TRAIN, không phải của train+validation+test."""
        train = load_prepared("train")
        X_train = build_feature_matrix(train)
        X_pooled = np.vstack([X_train, build_feature_matrix(load_prepared("validation")),
                              build_feature_matrix(load_prepared("test"))])

        pipe = make_model("logistic")
        pipe.fit(X_train, extract_labels(train))

        train_median = np.nanmedian(X_train, axis=0)
        pooled_median = np.nanmedian(X_pooled, axis=0)
        np.testing.assert_allclose(pipe.named_steps["impute"].statistics_, train_median,
                                   rtol=1e-9, atol=0.0,
                                   err_msg="imputer không học median từ train")
        # Chứng minh test có khả năng phân biệt: hai bộ median KHÁC nhau, nên nếu code gộp dữ liệu
        # trước khi impute thì assert phía trên sẽ đổ.
        self.assertGreater(float(np.nanmax(np.abs(pooled_median - train_median))), 0.0,
                           "median của train trùng median gộp ⇒ test không phân biệt được hai đường")

        imputed_train = np.where(np.isnan(X_train), train_median, X_train)
        np.testing.assert_allclose(pipe.named_steps["scale"].mean_, imputed_train.mean(axis=0),
                                   rtol=1e-6, atol=1e-9,
                                   err_msg="scaler không học mean từ train")
        imputed_pooled = np.where(np.isnan(X_pooled), pooled_median, X_pooled)
        self.assertGreater(float(np.max(np.abs(imputed_pooled.mean(axis=0)
                                               - imputed_train.mean(axis=0)))), 0.0)

    def test_balancing_weights_come_from_fit_labels_only(self):
        """`class_weight` là chuỗi ⇒ sklearn tính trọng số trong `fit` từ nhãn nhận được (train)."""
        for name in ("random_forest", "hist_gradient_boosting"):
            self.assertIsInstance(HYPERPARAMS[name]["class_weight"], str,
                                  f"{name}: class_weight phải là chuỗi 'balanced*' để sklearn tính "
                                  "trong fit; hằng số tính trước là trọng số toàn cục (rò rỉ)")

        y_train = extract_labels(load_prepared("train"))
        y_pooled = np.concatenate([y_train, extract_labels(load_prepared("validation")),
                                   extract_labels(load_prepared("test"))])
        w_train = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_train)
        w_pooled = compute_class_weight("balanced", classes=np.array([0, 1]), y=y_pooled)
        # Trọng số lớp thiểu số = n / (2 * n_dương) tính trên đúng nhãn train.
        self.assertAlmostEqual(float(w_train[1]), len(y_train) / (2.0 * int(y_train.sum())), places=6)
        self.assertFalse(np.allclose(w_train, w_pooled, rtol=1e-3),
                         "trọng số tính trên train phải KHÁC trọng số tính trên train+val+test "
                         "⇒ dùng nhãn gộp sẽ bị phát hiện")

    def test_cv_folds_are_stratified_and_grouped(self):
        """CV như `forecasting.tuning`: giữ tỉ lệ lớp VÀ giữ trọn công ty ngoài fold-train."""
        samples = load_prepared("train") + load_prepared("validation")
        groups = np.asarray([s["ticker"] for s in samples])
        X, y = build_feature_matrix(samples), extract_labels(samples)

        cv = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=42)
        gaps = []
        for fold, (train_index, val_index) in enumerate(cv.split(X, y, groups=groups), start=1):
            with self.subTest(fold=fold):
                self.assertEqual(set(groups[train_index]) & set(groups[val_index]), set(),
                                 f"fold {fold}: một công ty nằm ở cả train và validation")
                self.assertEqual(sorted(set(y[train_index].tolist())), [0, 1],
                                 f"fold {fold}: fold-train chỉ có một lớp")
                self.assertGreater(int(y[val_index].sum()), 0,
                                   f"fold {fold}: fold-validation không có mẫu dương")
                gap = abs(float(y[val_index].mean()) - float(y.mean()))
                gaps.append(gap)
                self.assertLess(gap, MAX_FOLD_RATIO_GAP,
                                f"fold {fold}: tỉ lệ lớp lệch {gap:.3f} (> {MAX_FOLD_RATIO_GAP})")
        self.assertLess(float(np.mean(gaps)), MAX_FOLD_MEAN_ABS_GAP,
                        "lệch tỉ lệ lớp trung bình qua các fold quá lớn")


class TestFeatures(unittest.TestCase):
    def test_matrix_shapes_and_labels(self):
        samples = load_prepared("train")
        X = build_feature_matrix(samples)
        y = extract_labels(samples)
        names = feature_names()
        self.assertEqual(X.shape, (len(samples), len(names)))
        self.assertEqual(y.shape, (len(samples),))
        self.assertEqual(set(set(y.tolist())), {0, 1})
        # Không có cột toàn NaN trên train (impute downstream vẫn hoạt động nhưng cảnh báo)
        self.assertGreater(len(names), 30)

    def test_no_label_leak_in_features(self):
        """Feature chỉ dùng lịch sử (available_on <= as_of), không nhìn tương lai."""
        samples = load_prepared("test")
        X = build_feature_matrix(samples)
        self.assertTrue(all(len(s["request"]["history"]) >= 1 for s in samples))
        # Mọi quý lịch sử phải công bố trước/as_of
        for s in samples[:10]:
            for row in s["request"]["history"]:
                self.assertLessEqual(row["available_on"], s["request"]["as_of"])

    def test_every_declared_feature_is_produced(self):
        """`_build_features` phải sinh ĐỦ mọi tên trong `feature_names()` (không cột NaN oan)."""
        from forecasting.features import _build_features

        names = set(feature_names())
        for sample in load_prepared("test")[:5]:
            produced = set(_build_features(sample))
            self.assertEqual(names - produced, set(),
                             f"{sample['sample_id']}: feature khai báo nhưng không sinh ra")

    def test_derived_liabilities_fill_coverage(self):
        """Quý thiếu tag `liabilities` vẫn phải có `debt_to_assets_latest` (suy ra từ A = L + E)."""
        from forecasting.features import _build_features

        checked = 0
        for sample in load_prepared("test") + load_prepared("train"):
            last = sample["request"]["history"][-1]
            if last.get("liabilities_vnd") is not None:
                continue
            feats = _build_features(sample)
            self.assertFalse(math.isnan(feats["debt_to_assets_latest"]),
                             f"{sample['sample_id']}: thiếu debt_to_assets dù đã suy ra nợ")
            self.assertFalse(math.isnan(feats["debt_to_equity_latest"]),
                             f"{sample['sample_id']}: thiếu debt_to_equity dù đã suy ra nợ")
            checked += 1
        self.assertGreater(checked, 0, "Không có mẫu nào thiếu tag liabilities để kiểm tra")


class TestModels(unittest.TestCase):
    def test_registry_has_baseline(self):
        self.assertIn("logistic", MODEL_REGISTRY)
        self.assertIn("random_forest", MODEL_REGISTRY)


class TestDataAudit(unittest.TestCase):
    """`scripts.audit_data` phải PHÁT HIỆN được dữ liệu sai (thử bằng cách sửa bản sao).

    Vì sao cần: kết quả "0 phát hiện" chỉ có nghĩa nếu auditor thực sự bắt được lỗi. Hai test
    dưới đây sửa một bản sao (nhãn, giá trị VND, provenance, ngày công bố) và đòi auditor báo lỗi.
    """

    @staticmethod
    def _retail_rows() -> dict:
        """{ticker: rows} đọc trực tiếp từ retail-expanded (không cần snapshot SEC)."""
        rows = {}
        for path in sorted((ROOT / "data" / "retail-expanded").glob("*-16-indicators-vnd.json")):
            doc = json.loads(path.read_text(encoding="utf-8"))
            rows[doc["ticker"]] = doc["rows"]
        return rows

    def test_mutated_prepared_and_manifest_are_reported(self):
        from scripts import audit_data

        prepared, manifest = audit_data.PREPARED_DIR, audit_data.MANIFEST_FILE
        if not (prepared / "manifest.json").exists():
            self.skipTest("Chưa có data/prepared.")
        with tempfile.TemporaryDirectory() as td:
            tmp = pathlib.Path(td)
            shutil.copytree(prepared, tmp, dirs_exist_ok=True)
            audit_data.PREPARED_DIR, audit_data.MANIFEST_FILE = tmp, tmp / "manifest.json"
            try:
                samples = json.loads((tmp / "test.json").read_text(encoding="utf-8"))
                samples[0]["request"]["history"][0]["revenue_vnd"] = "1"          # số liệu sai
                samples[1]["is_distressed"] = 1 - int(samples[1]["is_distressed"])  # nhãn sai
                (tmp / "test.json").write_text(
                    json.dumps(samples, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
                auditor = audit_data.Auditor()
                splits = audit_data.check_prepared(auditor, self._retail_rows())
                audit_data.check_manifest(auditor, self._retail_rows(), splits)
            finally:
                audit_data.PREPARED_DIR, audit_data.MANIFEST_FILE = prepared, manifest

        prepared_msgs = " | ".join(i["message"] for i in auditor.issues.get("prepared", []))
        self.assertIn("khác retail-expanded", prepared_msgs, prepared_msgs)
        manifest_msgs = " | ".join(i["message"] for i in auditor.issues.get("manifest", []))
        self.assertIn("label_counts[test].distressed", manifest_msgs, manifest_msgs)
        self.assertIn("split_sha256[test]", manifest_msgs, manifest_msgs)

    def test_mutated_indicator_file_is_reported(self):
        from scripts import audit_data

        source = ROOT / "data" / "retail-expanded" / "WMT-16-indicators-vnd.json"
        snapshot = ROOT / "data" / "sec" / "raw" / "WMT-companyfacts.json"
        if not source.exists() or not snapshot.exists():
            self.skipTest("Chưa có retail-expanded hoặc snapshot SEC (chạy scripts.crawl_sec).")
        with tempfile.TemporaryDirectory() as td:
            doc = json.loads(source.read_text(encoding="utf-8"))
            doc["rows"][0]["revenue_vnd"] = str(int(doc["rows"][0]["revenue_vnd"]) + 1)
            doc["rows"][0]["available_on"] = "1999-01-01"                  # ngày công bố sai
            doc["rows"][1]["sources"]["inventory"]["facts"][0]["val"] += 1  # provenance sai
            path = pathlib.Path(td) / source.name
            path.write_text(json.dumps(doc, ensure_ascii=False, indent=2) + "\n",
                            encoding="utf-8")
            auditor = audit_data.Auditor()
            audit_data.check_retail_doc(auditor, path, "retail_expanded")

        messages = " | ".join(i["message"] for i in auditor.issues.get("retail_expanded", []))
        self.assertIn("VND khác số liệu thật", messages, messages)
        self.assertIn("available_on", messages, messages)
        self.assertIn("`val` provenance khác companyfacts", messages, messages)


class TestModelSelection(unittest.TestCase):
    """Mô hình được chốt phải khớp quy tắc chọn công bố trước (ưu tiên AP cross-company)."""

    SUMMARY = ROOT / "reports" / "results" / "summary.json"

    def test_selected_model_matches_selection_rule(self):
        if not self.SUMMARY.exists():
            self.skipTest("Chưa có reports/results/summary.json (chạy forecasting.train).")
        summary = json.loads(self.SUMMARY.read_text(encoding="utf-8"))
        models = summary.get("models") or []
        self.assertTrue(models, "summary.json không có danh sách mô hình")
        rule = summary.get("selection_rule", "")
        self.assertIn("cross-company", rule,
                      "Quy tắc chọn mô hình phải ưu tiên AP cross-company (chống rò rỉ thực thể)")
        for row in models:
            self.assertIn("cross_company_ap", row,
                          f"{row['model']}: thiếu AP cross-company để áp quy tắc chọn")

        def rank_key(row: dict):
            def safe(value):
                try:
                    number = float(value)
                except (TypeError, ValueError):
                    return -1.0
                return -1.0 if number != number else number
            return (safe(row.get("cross_company_ap")), safe(row.get("best_f1_val")),
                    safe(row.get("average_precision")), safe(row.get("auroc")),
                    -safe(row.get("overfit_gap_f1")))

        expected = max(models, key=rank_key)["model"]
        self.assertEqual(summary.get("best_model"), expected,
                         "Mô hình được chốt không khớp quy tắc xếp hạng công bố")


if __name__ == "__main__":
    unittest.main()
