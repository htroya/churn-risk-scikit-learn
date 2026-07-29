from __future__ import annotations

import argparse
import hashlib
import json
import logging
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path

import joblib
import numpy as np
import polars as pl
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.inspection import permutation_importance
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    brier_score_loss,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import StratifiedKFold, cross_validate, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

LOGGER = logging.getLogger("churn_training")

NUMERIC_FEATURES = [
    "tenure_months",
    "monthly_charge",
    "support_tickets",
    "late_payments",
    "service_usage",
    "contract_score",
    "satisfaction_score",
]
CATEGORICAL_FEATURES = ["segment", "contract_type", "autopay"]
FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES
TARGET = "churn"


@dataclass(frozen=True)
class TrainingConfig:
    rows: int = 8_000
    seed: int = 42
    test_size: float = 0.25
    minimum_recall: float = 0.78
    cv_folds: int = 3
    quick_validation: bool = False


def build_dataset(rows: int = 8_000, seed: int = 42) -> pl.DataFrame:
    """Build a deterministic dataset whose churn drivers have business meaning."""
    if rows < 500:
        raise ValueError("rows must be at least 500")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 121, rows)
    monthly_charge = np.round(rng.normal(68, 24, rows).clip(15, 180), 2)
    support_tickets = rng.poisson(1.8, rows).clip(0, 12)
    late_payments = rng.poisson(1.1, rows).clip(0, 8)
    service_usage = np.round(rng.normal(64, 22, rows).clip(0, 100), 2)
    contract_score = rng.integers(1, 6, rows)
    satisfaction = np.round(
        (92 - support_tickets * 8 - late_payments * 4 + rng.normal(0, 10, rows)).clip(1, 100),
        2,
    )
    segment = rng.choice(["hogar", "pyme", "corporativo"], rows, p=[0.68, 0.23, 0.09])
    contract_type = rng.choice(
        ["mensual", "anual", "bianual"],
        rows,
        p=[0.48, 0.36, 0.16],
    )
    autopay = rng.choice(["si", "no"], rows, p=[0.62, 0.38])

    linear_risk = (
        -1.25
        - tenure * 0.014
        + (monthly_charge - 65) * 0.012
        + support_tickets * 0.30
        + late_payments * 0.36
        - service_usage * 0.012
        - contract_score * 0.12
        - satisfaction * 0.017
        + (contract_type == "mensual") * 0.68
        + (autopay == "no") * 0.42
        + (segment == "hogar") * 0.12
    )
    probability = 1 / (1 + np.exp(-linear_risk))
    churn = rng.binomial(1, probability)

    return pl.DataFrame(
        {
            "customer_id": [f"C{index:07d}" for index in range(1, rows + 1)],
            "tenure_months": tenure,
            "monthly_charge": monthly_charge,
            "support_tickets": support_tickets,
            "late_payments": late_payments,
            "service_usage": service_usage,
            "contract_score": contract_score,
            "satisfaction_score": satisfaction,
            "segment": segment,
            "contract_type": contract_type,
            "autopay": autopay,
            TARGET: churn,
        }
    )


def _preprocessor() -> ColumnTransformer:
    numeric_indices = list(range(len(NUMERIC_FEATURES)))
    categorical_indices = list(
        range(len(NUMERIC_FEATURES), len(NUMERIC_FEATURES) + len(CATEGORICAL_FEATURES))
    )
    return ColumnTransformer(
        [
            (
                "numeric",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="median")),
                        ("scale", StandardScaler()),
                    ]
                ),
                numeric_indices,
            ),
            (
                "categorical",
                Pipeline(
                    [
                        ("impute", SimpleImputer(strategy="most_frequent")),
                        (
                            "encode",
                            OneHotEncoder(handle_unknown="ignore", sparse_output=False),
                        ),
                    ]
                ),
                categorical_indices,
            ),
        ]
    )


def candidate_models(seed: int, quick_validation: bool = False) -> dict[str, Pipeline]:
    forest_estimators = 20 if quick_validation else 140
    boosting_iterations = 25 if quick_validation else 160
    estimators = {
        "logistic_regression": LogisticRegression(
            class_weight="balanced",
            max_iter=2_000,
            random_state=seed,
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=forest_estimators,
            min_samples_leaf=8,
            class_weight="balanced_subsample",
            n_jobs=1,
            random_state=seed,
        ),
        "hist_gradient_boosting": HistGradientBoostingClassifier(
            learning_rate=0.08,
            max_iter=boosting_iterations,
            max_leaf_nodes=24,
            l2_regularization=1.0,
            random_state=seed,
        ),
    }
    return {
        name: Pipeline([("preprocess", _preprocessor()), ("model", estimator)])
        for name, estimator in estimators.items()
    }


def select_model(
    x_train: np.ndarray,
    y_train: np.ndarray,
    config: TrainingConfig,
) -> tuple[str, Pipeline, list[dict[str, float | str]]]:
    cv = StratifiedKFold(
        n_splits=config.cv_folds,
        shuffle=True,
        random_state=config.seed,
    )
    leaderboard: list[dict[str, float | str]] = []
    models = candidate_models(config.seed, config.quick_validation)
    for name, pipeline in models.items():
        scores = cross_validate(
            pipeline,
            x_train,
            y_train,
            cv=cv,
            scoring={"roc_auc": "roc_auc", "recall": "recall", "precision": "precision"},
            n_jobs=1,
        )
        leaderboard.append(
            {
                "model": name,
                "cv_roc_auc": round(float(scores["test_roc_auc"].mean()), 4),
                "cv_recall": round(float(scores["test_recall"].mean()), 4),
                "cv_precision": round(float(scores["test_precision"].mean()), 4),
            }
        )
    leaderboard.sort(key=lambda row: float(row["cv_roc_auc"]), reverse=True)
    selected_name = str(leaderboard[0]["model"])
    selected_model = models[selected_name].fit(x_train, y_train)
    return selected_name, selected_model, leaderboard


def optimize_threshold(
    y_true: np.ndarray,
    probabilities: np.ndarray,
    minimum_recall: float,
) -> float:
    precision, recall, thresholds = precision_recall_curve(y_true, probabilities)
    candidates: list[tuple[float, float, float]] = []
    for index, threshold in enumerate(thresholds):
        if recall[index] >= minimum_recall:
            candidates.append((float(precision[index]), float(recall[index]), float(threshold)))
    if not candidates:
        return 0.5
    candidates.sort(key=lambda item: (item[0], item[1]), reverse=True)
    return round(candidates[0][2], 4)


def _lift_at_fraction(y_true: np.ndarray, probability: np.ndarray, fraction: float = 0.10) -> float:
    size = max(1, int(len(y_true) * fraction))
    top = y_true[np.argsort(probability)[::-1][:size]].mean()
    baseline = y_true.mean()
    return round(float(top / baseline), 4) if baseline else 0.0


def _baseline(frame: pl.DataFrame) -> dict[str, object]:
    numeric = {
        feature: {
            "mean": round(float(frame[feature].mean()), 6),
            "std": round(float(frame[feature].std()), 6),
            "p05": round(float(frame[feature].quantile(0.05)), 6),
            "p95": round(float(frame[feature].quantile(0.95)), 6),
        }
        for feature in NUMERIC_FEATURES
    }
    categorical = {
        feature: {
            str(row[feature]): round(row["len"] / frame.height, 6)
            for row in frame.group_by(feature).len().to_dicts()
        }
        for feature in CATEGORICAL_FEATURES
    }
    return {"numeric": numeric, "categorical": categorical}


def _write_model_card(
    output: Path,
    selected_model: str,
    metrics: dict[str, float | str],
    config: TrainingConfig,
) -> None:
    content = f"""# Model Card — Churn Risk

## Intended use

Priorizar clientes para revisión humana y campañas de retención. El score no
debe utilizarse para negar servicios ni tomar decisiones automatizadas de alto
impacto.

## Training

- Dataset: sintético y reproducible, {config.rows:,} registros.
- Modelo seleccionado: `{selected_model}`.
- Validación: {config.cv_folds}-fold estratificada y conjunto holdout.
- Umbral operativo: {metrics['threshold']}.

## Holdout metrics

| Metric | Value |
|---|---:|
| ROC-AUC | {metrics['roc_auc']} |
| Recall | {metrics['recall']} |
| Precision | {metrics['precision']} |
| F1 | {metrics['f1']} |
| Brier score | {metrics['brier_score']} |
| Lift@10% | {metrics['lift_at_10']} |

## Governance

Antes de producción deben validarse representatividad, sesgo por segmento,
calibración, deriva, explicabilidad, privacidad, aprobación del dueño del
proceso y monitoreo de resultados reales.
"""
    (output / "model_card.md").write_text(content, encoding="utf-8")


def _manifest(output: Path, files: list[str], run_id: str) -> None:
    artifacts = {}
    for name in files:
        path = output / name
        artifacts[name] = {
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
    (output / "manifest.json").write_text(
        json.dumps(
            {
                "run_id": run_id,
                "created_at": datetime.now(UTC).isoformat(),
                "artifacts": artifacts,
            },
            indent=2,
        ),
        encoding="utf-8",
    )


def train(
    output: Path,
    rows: int = 8_000,
    seed: int = 42,
    minimum_recall: float = 0.78,
    quick_validation: bool = False,
) -> dict[str, float | str]:
    output.mkdir(parents=True, exist_ok=True)
    config = TrainingConfig(
        rows=rows,
        seed=seed,
        minimum_recall=minimum_recall,
        cv_folds=2 if rows < 2_500 else 3,
        quick_validation=quick_validation,
    )
    frame = build_dataset(rows=config.rows, seed=config.seed)
    x = frame.select(FEATURES).to_numpy()
    y = frame[TARGET].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=config.test_size,
        stratify=y,
        random_state=config.seed,
    )
    selected_name, model, leaderboard = select_model(x_train, y_train, config)
    probability = model.predict_proba(x_test)[:, 1]
    threshold = optimize_threshold(y_test, probability, config.minimum_recall)
    prediction = (probability >= threshold).astype(int)
    metrics: dict[str, float | str] = {
        "selected_model": selected_name,
        "threshold": threshold,
        "recall": round(float(recall_score(y_test, prediction)), 4),
        "precision": round(float(precision_score(y_test, prediction, zero_division=0)), 4),
        "f1": round(float(f1_score(y_test, prediction)), 4),
        "roc_auc": round(float(roc_auc_score(y_test, probability)), 4),
        "brier_score": round(float(brier_score_loss(y_test, probability)), 4),
        "lift_at_10": _lift_at_fraction(y_test, probability),
        "positive_rate": round(float(y.mean()), 4),
    }

    bundle = {
        "pipeline": model,
        "threshold": threshold,
        "features": FEATURES,
        "numeric_features": NUMERIC_FEATURES,
        "categorical_features": CATEGORICAL_FEATURES,
        "model_version": "1.0.0",
    }
    joblib.dump(bundle, output / "churn_model.joblib")
    (output / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    pl.DataFrame(leaderboard).write_csv(output / "model_leaderboard.csv")

    importance = permutation_importance(
        model,
        x_test,
        y_test,
        scoring="roc_auc",
        n_repeats=1 if quick_validation else 4,
        random_state=config.seed,
    )
    pl.DataFrame(
        {
            "feature": FEATURES,
            "importance_mean": importance.importances_mean,
            "importance_std": importance.importances_std,
        }
    ).sort("importance_mean", descending=True).write_csv(output / "feature_importance.csv")
    (output / "training_baseline.json").write_text(
        json.dumps(_baseline(frame.select(FEATURES)), indent=2),
        encoding="utf-8",
    )
    _write_model_card(output, selected_name, metrics, config)
    _manifest(
        output,
        [
            "churn_model.joblib",
            "metrics.json",
            "model_leaderboard.csv",
            "feature_importance.csv",
            "training_baseline.json",
            "model_card.md",
        ],
        run_id=f"train-{datetime.now(UTC).strftime('%Y%m%dT%H%M%SZ')}",
    )
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Training pipeline for explainable churn risk")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--rows", type=int, default=8_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--minimum-recall", type=float, default=0.78)
    parser.add_argument("--log-level", default="INFO")
    args = parser.parse_args()
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper()),
        format="%(asctime)s %(levelname)s %(name)s %(message)s",
    )
    LOGGER.info("Starting model selection and training")
    LOGGER.info(
        "Training completed: %s",
        train(args.output, args.rows, args.seed, args.minimum_recall),
    )


if __name__ == "__main__":
    main()
