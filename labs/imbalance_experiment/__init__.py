"""Standalone experiment comparing single and hybrid resampling methods on imbalanced data.

It reuses the tested building blocks of `labs.imbalance_lab` and keeps every resampling step inside
`imblearn.pipeline.Pipeline`, so fitting only ever sees fold-train data. Entry point:
`python -m labs.imbalance_experiment.main`.
"""
from __future__ import annotations

__all__ = ["config", "data_loader", "pipeline_builder", "evaluation", "insights", "main"]
