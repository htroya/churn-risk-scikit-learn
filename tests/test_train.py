import json
from pathlib import Path

import numpy as np
import pytest

from src.train import build_dataset, select_threshold, train


def test_dataset_is_domain_shaped_and_reproducible() -> None:
    first = build_dataset(rows=500, seed=7)
    second = build_dataset(rows=500, seed=7)
    assert first.equals(second)
    assert first["churn"].n_unique() == 2
    assert first["tenure_months"].min() >= 1
    assert first["service_usage"].is_between(0, 100).all()
    with pytest.raises(ValueError, match="al menos 100"):
        build_dataset(rows=99)


def test_threshold_selection_honors_recall_target() -> None:
    actual = np.array([0, 0, 1, 1, 1])
    probability = np.array([0.1, 0.3, 0.4, 0.7, 0.9])
    threshold, analysis = select_threshold(actual, probability, target_recall=2 / 3)
    selected = analysis.filter("selected").row(0, named=True)
    assert threshold == selected["threshold"]
    assert selected["recall"] >= 2 / 3
    with pytest.raises(ValueError, match="entre 0 y 1"):
        select_threshold(actual, probability, target_recall=0)


def test_training_exports_metrics_plots_and_explanations(tmp_path: Path) -> None:
    metrics = train(tmp_path, rows=2_000, seed=42, target_recall=0.80)
    assert metrics["recall"] >= 0.79
    assert metrics["precision"] > metrics["test_prevalence"]
    assert metrics["f1"] > metrics["baseline_f1"]
    assert metrics["roc_auc"] > metrics["baseline_roc_auc"]
    assert metrics["baseline_roc_auc"] == 0.5

    expected = {
        "churn_model.joblib",
        "metrics.json",
        "feature_importance.csv",
        "threshold_analysis.csv",
        "confusion_matrix.csv",
        "model_comparison.csv",
        "roc_curve.csv",
        "precision_recall_curve.csv",
        "test_predictions.csv",
        "confusion_matrix.png",
        "roc_curve.png",
        "precision_recall_curve.png",
        "feature_importance.png",
        "threshold_rationale.md",
        "business_interpretation.md",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    persisted = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert persisted == metrics
    assert "datos sintéticos" in (tmp_path / "business_interpretation.md").read_text(
        encoding="utf-8"
    )
