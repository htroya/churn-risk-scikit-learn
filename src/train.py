from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

FEATURES = [
    "tenure_months",
    "monthly_charge",
    "support_tickets",
    "late_payments",
    "service_usage",
    "contract_score",
]


def build_dataset(rows: int = 4_000, seed: int = 42) -> pl.DataFrame:
    features, target = make_classification(
        n_samples=rows,
        n_features=len(FEATURES),
        n_informative=5,
        n_redundant=1,
        weights=[0.72, 0.28],
        class_sep=1.0,
        random_state=seed,
    )
    frame = pl.DataFrame(features, schema=FEATURES, orient="row")
    return frame.with_columns(pl.Series("churn", target))


def train(output: Path, rows: int = 4_000) -> dict[str, float]:
    output.mkdir(parents=True, exist_ok=True)
    frame = build_dataset(rows=rows)
    x_train, x_test, y_train, y_test = train_test_split(
        frame.select(FEATURES).to_numpy(),
        frame["churn"].to_numpy(),
        test_size=0.25,
        stratify=frame["churn"].to_numpy(),
        random_state=42,
    )
    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(
                    class_weight="balanced",
                    max_iter=1_000,
                    random_state=42,
                ),
            ),
        ]
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    prediction = (probability >= 0.40).astype(int)
    metrics = {
        "recall": round(float(recall_score(y_test, prediction)), 4),
        "precision": round(float(precision_score(y_test, prediction)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
    }
    joblib.dump(model, output / "churn_model.joblib")
    (output / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    coefficients = model.named_steps["model"].coef_[0]
    pl.DataFrame({"feature": FEATURES, "coefficient": coefficients}).with_columns(
        pl.col("coefficient").abs().alias("importance")
    ).sort("importance", descending=True).write_csv(output / "feature_importance.csv")
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modelo reproducible de riesgo de churn")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--rows", type=int, default=4_000)
    args = parser.parse_args()
    print(train(args.output, rows=args.rows))
