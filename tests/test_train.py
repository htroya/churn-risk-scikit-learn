import json
from pathlib import Path

from src.train import train


def test_training_exports_model_metrics_and_explanations(tmp_path: Path) -> None:
    metrics = train(tmp_path, rows=1_200)
    assert metrics["recall"] >= 0.70
    assert metrics["roc_auc"] >= 0.75
    assert (tmp_path / "churn_model.joblib").exists()
    assert (tmp_path / "feature_importance.csv").exists()
    persisted = json.loads((tmp_path / "metrics.json").read_text(encoding="utf-8"))
    assert persisted == metrics
