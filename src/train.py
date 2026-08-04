from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import polars as pl
from sklearn.dummy import DummyClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    auc,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


FEATURES = [
    "tenure_months",
    "monthly_charge",
    "support_tickets",
    "late_payments",
    "service_usage",
    "contract_score",
]


def build_dataset(rows: int = 4_000, seed: int = 42) -> pl.DataFrame:
    """Generate domain-shaped synthetic customers with reproducible churn labels."""
    if rows < 100:
        raise ValueError("rows debe ser al menos 100")
    rng = np.random.default_rng(seed)
    tenure = rng.integers(1, 121, size=rows)
    monthly_charge = np.clip(rng.normal(68, 20, size=rows), 18, 160)
    support_tickets = np.clip(rng.poisson(1.3, size=rows), 0, 8)
    late_payments = np.clip(rng.poisson(0.7, size=rows), 0, 6)
    service_usage = rng.beta(2.4, 1.8, size=rows) * 100
    contract_score = rng.choice([0.0, 0.5, 1.0], size=rows, p=[0.45, 0.30, 0.25])

    logit = (
        -0.35
        - tenure * 0.014
        + (monthly_charge - 68) * 0.020
        + support_tickets * 0.48
        + late_payments * 0.62
        - service_usage * 0.012
        - contract_score * 0.95
        + rng.normal(0, 0.45, size=rows)
    )
    churn_probability = 1 / (1 + np.exp(-logit))
    target = rng.binomial(1, churn_probability)

    return pl.DataFrame(
        {
            "tenure_months": tenure,
            "monthly_charge": monthly_charge.round(2),
            "support_tickets": support_tickets,
            "late_payments": late_payments,
            "service_usage": service_usage.round(2),
            "contract_score": contract_score,
            "churn": target,
        }
    )


def select_threshold(
    y_true: np.ndarray,
    probability: np.ndarray,
    target_recall: float = 0.80,
) -> tuple[float, pl.DataFrame]:
    if not 0 < target_recall <= 1:
        raise ValueError("target_recall debe estar entre 0 y 1")
    precision, recall, thresholds = precision_recall_curve(y_true, probability)
    candidates = np.flatnonzero(recall[:-1] >= target_recall)
    selected_index = int(candidates[np.argmax(precision[:-1][candidates])]) if candidates.size else 0
    analysis = pl.DataFrame(
        {
            "threshold": thresholds,
            "precision": precision[:-1],
            "recall": recall[:-1],
            "selected": np.arange(len(thresholds)) == selected_index,
        }
    )
    return float(thresholds[selected_index]), analysis


def _metrics(y_true: np.ndarray, probability: np.ndarray, threshold: float) -> dict[str, float | int]:
    prediction = (probability >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, prediction).ravel()
    return {
        "threshold": round(threshold, 6),
        "precision": round(float(precision_score(y_true, prediction, zero_division=0)), 4),
        "recall": round(float(recall_score(y_true, prediction, zero_division=0)), 4),
        "f1": round(float(f1_score(y_true, prediction, zero_division=0)), 4),
        "roc_auc": round(float(roc_auc_score(y_true, probability)), 4),
        "true_negative": int(tn),
        "false_positive": int(fp),
        "false_negative": int(fn),
        "true_positive": int(tp),
    }


def _write_plots(
    output: Path,
    y_test: np.ndarray,
    probability: np.ndarray,
    prediction: np.ndarray,
    coefficients: np.ndarray,
) -> None:
    matrix = confusion_matrix(y_test, prediction)
    display = ConfusionMatrixDisplay(matrix, display_labels=["permanece", "churn"])
    display.plot(cmap="Blues", colorbar=False)
    display.ax_.set_title("Matriz de confusión · datos sintéticos")
    display.figure_.tight_layout()
    display.figure_.savefig(output / "confusion_matrix.png", dpi=160)
    plt.close(display.figure_)

    fpr, tpr, _ = roc_curve(y_test, probability)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(fpr, tpr, color="#246BFD", label=f"Modelo (AUC={auc(fpr, tpr):.3f})")
    axis.plot([0, 1], [0, 1], linestyle="--", color="#6B7280", label="Referencia")
    axis.set(xlabel="False positive rate", ylabel="True positive rate", title="Curva ROC")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output / "roc_curve.png", dpi=160)
    plt.close(figure)

    precision, recall, _ = precision_recall_curve(y_test, probability)
    figure, axis = plt.subplots(figsize=(6.4, 4.2))
    axis.plot(recall, precision, color="#7C3AED")
    axis.set(xlabel="Recall", ylabel="Precision", title="Curva precision–recall")
    figure.tight_layout()
    figure.savefig(output / "precision_recall_curve.png", dpi=160)
    plt.close(figure)

    order = np.argsort(np.abs(coefficients))
    colors = ["#C2413B" if coefficients[index] > 0 else "#1E7A55" for index in order]
    figure, axis = plt.subplots(figsize=(7.2, 4.4))
    axis.barh(np.array(FEATURES)[order], coefficients[order], color=colors)
    axis.axvline(0, color="#6B7280", linewidth=1)
    axis.set(title="Coeficientes estandarizados", xlabel="Efecto sobre log-odds de churn")
    figure.tight_layout()
    figure.savefig(output / "feature_importance.png", dpi=160)
    plt.close(figure)


def _write_business_notes(
    output: Path,
    metrics: dict[str, float | int],
    target_recall: float,
    test_rows: int,
) -> None:
    flagged = int(metrics["true_positive"]) + int(metrics["false_positive"])
    output.joinpath("threshold_rationale.md").write_text(
        "# Selección del umbral\n\n"
        f"Se eligió el umbral **{metrics['threshold']}** entre los puntos que alcanzan al menos "
        f"**{target_recall:.0%} de recall**; dentro de ese conjunto se maximizó precision. "
        "La decisión prioriza capturar clientes sintéticos en riesgo y hace visible el costo en "
        "falsos positivos. El objetivo debe sustituirse por una función de costos acordada antes "
        "de cualquier uso real.\n",
        encoding="utf-8",
    )
    output.joinpath("business_interpretation.md").write_text(
        "# Interpretación comercial\n\n"
        "> Resultados de datos sintéticos; no describen una cartera real.\n\n"
        f"En **{test_rows:,} clientes de prueba**, el umbral marca **{flagged:,}** para revisión. "
        f"Captura **{metrics['true_positive']}** casos positivos sintéticos y deja "
        f"**{metrics['false_negative']}** sin detectar; **{metrics['false_positive']}** contactos "
        "serían falsos positivos. Esto convierte el umbral en una decisión de capacidad y costo: "
        "se debe estimar valor de retención, costo de contacto y riesgo de fatiga antes de operar.\n",
        encoding="utf-8",
    )


def train(
    output: Path,
    rows: int = 4_000,
    seed: int = 42,
    target_recall: float = 0.80,
    persist_model: bool = True,
) -> dict[str, float | int]:
    output.mkdir(parents=True, exist_ok=True)
    frame = build_dataset(rows=rows, seed=seed)
    x = frame.select(FEATURES).to_numpy()
    y = frame["churn"].to_numpy()
    x_train, x_test, y_train, y_test = train_test_split(
        x,
        y,
        test_size=0.25,
        stratify=y,
        random_state=seed,
    )

    baseline = DummyClassifier(strategy="prior")
    baseline.fit(x_train, y_train)
    baseline_probability = baseline.predict_proba(x_test)[:, 1]
    baseline_metrics = _metrics(y_test, baseline_probability, threshold=0.50)

    model = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "model",
                LogisticRegression(class_weight="balanced", max_iter=1_000, random_state=seed),
            ),
        ]
    )
    model.fit(x_train, y_train)
    probability = model.predict_proba(x_test)[:, 1]
    threshold, threshold_analysis = select_threshold(y_test, probability, target_recall)
    prediction = (probability >= threshold).astype(int)
    metrics = _metrics(y_test, probability, threshold)
    metrics.update(
        {
            "rows": rows,
            "test_rows": len(y_test),
            "test_prevalence": round(float(np.mean(y_test)), 4),
            "target_recall": target_recall,
            "baseline_f1": baseline_metrics["f1"],
            "baseline_roc_auc": baseline_metrics["roc_auc"],
        }
    )

    if persist_model:
        joblib.dump(model, output / "churn_model.joblib")
    output.joinpath("metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    coefficients = model.named_steps["model"].coef_[0]
    pl.DataFrame(
        {
            "feature": FEATURES,
            "coefficient": coefficients,
            "direction": ["increases_risk" if value > 0 else "decreases_risk" for value in coefficients],
        }
    ).with_columns(pl.col("coefficient").abs().alias("importance")).sort(
        "importance", descending=True
    ).write_csv(output / "feature_importance.csv")
    threshold_analysis.write_csv(output / "threshold_analysis.csv")

    fpr, tpr, roc_thresholds = roc_curve(y_test, probability)
    pl.DataFrame({"fpr": fpr, "tpr": tpr, "threshold": roc_thresholds}).write_csv(
        output / "roc_curve.csv"
    )
    pr_precision, pr_recall, pr_thresholds = precision_recall_curve(y_test, probability)
    pl.DataFrame(
        {
            "precision": pr_precision[:-1],
            "recall": pr_recall[:-1],
            "threshold": pr_thresholds,
        }
    ).write_csv(output / "precision_recall_curve.csv")
    pl.DataFrame(
        {
            "actual_churn": y_test,
            "churn_probability": probability,
            "predicted_churn": prediction,
        }
    ).write_csv(output / "test_predictions.csv")
    pl.DataFrame(
        {
            "actual": ["remains", "remains", "churn", "churn"],
            "predicted": ["remains", "churn", "remains", "churn"],
            "customers": [
                metrics["true_negative"],
                metrics["false_positive"],
                metrics["false_negative"],
                metrics["true_positive"],
            ],
        }
    ).write_csv(output / "confusion_matrix.csv")
    pl.DataFrame(
        {
            "model": ["dummy_prior", "logistic_regression"],
            "f1": [baseline_metrics["f1"], metrics["f1"]],
            "roc_auc": [baseline_metrics["roc_auc"], metrics["roc_auc"]],
        }
    ).write_csv(output / "model_comparison.csv")

    _write_plots(output, y_test, probability, prediction, coefficients)
    _write_business_notes(output, metrics, target_recall, len(y_test))
    return metrics


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Modelo reproducible de riesgo de churn")
    parser.add_argument("--output", type=Path, default=Path("artifacts"))
    parser.add_argument("--rows", type=int, default=4_000)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--target-recall", type=float, default=0.80)
    parser.add_argument("--skip-model", action="store_true")
    args = parser.parse_args()
    print(train(args.output, args.rows, args.seed, args.target_recall, not args.skip_model))
