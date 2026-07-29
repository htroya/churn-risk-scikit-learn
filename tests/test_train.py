import json
from pathlib import Path

import joblib

from src.score import score_batch
from src.train import FEATURES, build_dataset, train


def test_dataset_has_semantic_features_and_reproducible_target() -> None:
    first = build_dataset(rows=600, seed=7)
    second = build_dataset(rows=600, seed=7)
    assert first.equals(second)
    assert set(FEATURES) <= set(first.columns)
    assert 0.05 < first["churn"].mean() < 0.60


def test_training_exports_governed_model_package(tmp_path: Path) -> None:
    metrics = train(
        tmp_path,
        rows=1_500,
        seed=9,
        minimum_recall=0.70,
        quick_validation=True,
    )
    assert metrics["recall"] >= 0.70
    assert metrics["roc_auc"] >= 0.72
    expected = {
        "churn_model.joblib",
        "metrics.json",
        "model_leaderboard.csv",
        "feature_importance.csv",
        "training_baseline.json",
        "model_card.md",
        "manifest.json",
    }
    assert expected <= {path.name for path in tmp_path.iterdir()}
    persisted = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert persisted == metrics
    bundle = joblib.load(tmp_path / "churn_model.joblib")
    assert bundle["model_version"] == "1.0.0"


def test_batch_scoring_is_versioned_and_segmented(tmp_path: Path) -> None:
    train(
        tmp_path,
        rows=1_200,
        seed=12,
        minimum_recall=0.68,
        quick_validation=True,
    )
    frame = build_dataset(rows=600, seed=13).drop("churn")
    scores = score_batch(tmp_path / "churn_model.joblib", frame)
    assert scores.height == 600
    assert scores["churn_probability"].is_between(0, 1).all()
    assert scores["model_version"].n_unique() == 1
    assert scores["risk_segment"].n_unique() >= 2
