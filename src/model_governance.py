from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
from sklearn.metrics import brier_score_loss, precision_score, recall_score, roc_auc_score


@dataclass(frozen=True)
class ThresholdDecision:
    threshold: float
    selected: int
    capacity: int
    precision: float
    recall: float
    expected_net_value: float


def expected_calibration_error(
    actual: np.ndarray, probability: np.ndarray, bins: int = 10
) -> float:
    if bins < 2:
        raise ValueError("bins debe ser al menos 2")
    edges = np.linspace(0.0, 1.0, bins + 1)
    bucket = np.clip(np.digitize(probability, edges[1:-1]), 0, bins - 1)
    error = 0.0
    for index in range(bins):
        mask = bucket == index
        if mask.any():
            error += mask.mean() * abs(float(actual[mask].mean()) - float(probability[mask].mean()))
    return float(error)


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, bins: int = 10
) -> float:
    if len(reference) == 0 or len(current) == 0:
        raise ValueError("las poblaciones no pueden estar vacías")
    edges = np.unique(np.quantile(reference, np.linspace(0, 1, bins + 1)))
    if len(edges) < 3:
        return 0.0
    edges[0], edges[-1] = -np.inf, np.inf
    ref = np.histogram(reference, bins=edges)[0] / len(reference)
    cur = np.histogram(current, bins=edges)[0] / len(current)
    ref = np.clip(ref, 1e-6, None)
    cur = np.clip(cur, 1e-6, None)
    return float(np.sum((cur - ref) * np.log(cur / ref)))


def select_capacity_threshold(
    actual: np.ndarray,
    probability: np.ndarray,
    capacity_ratio: float = 0.20,
    contact_cost: float = 4.0,
    retained_value: float = 85.0,
) -> ThresholdDecision:
    if not 0 < capacity_ratio <= 1:
        raise ValueError("capacity_ratio debe estar entre 0 y 1")
    capacity = max(1, int(np.floor(len(probability) * capacity_ratio)))
    candidates = np.unique(np.r_[probability, 1.0])
    decisions: list[ThresholdDecision] = []
    for threshold in candidates:
        predicted = probability >= threshold
        selected = int(predicted.sum())
        if selected == 0 or selected > capacity:
            continue
        true_positives = int(np.sum(predicted & (actual == 1)))
        net_value = true_positives * retained_value - selected * contact_cost
        decisions.append(
            ThresholdDecision(
                threshold=float(threshold),
                selected=selected,
                capacity=capacity,
                precision=float(precision_score(actual, predicted, zero_division=0)),
                recall=float(recall_score(actual, predicted, zero_division=0)),
                expected_net_value=float(net_value),
            )
        )
    if not decisions:
        raise ValueError("no existe un umbral compatible con la capacidad")
    return max(decisions, key=lambda item: (item.expected_net_value, item.recall, item.precision))


def segment_performance(
    actual: np.ndarray,
    probability: np.ndarray,
    segments: np.ndarray,
    threshold: float,
) -> pl.DataFrame:
    rows: list[dict] = []
    for segment in sorted(np.unique(segments).tolist()):
        mask = segments == segment
        predicted = probability[mask] >= threshold
        rows.append(
            {
                "segment": str(segment),
                "rows": int(mask.sum()),
                "prevalence": float(actual[mask].mean()),
                "selection_rate": float(predicted.mean()),
                "precision": float(precision_score(actual[mask], predicted, zero_division=0)),
                "recall": float(recall_score(actual[mask], predicted, zero_division=0)),
                "brier_score": float(brier_score_loss(actual[mask], probability[mask])),
            }
        )
    frame = pl.DataFrame(rows)
    max_recall = frame["recall"].max()
    return frame.with_columns((pl.lit(max_recall) - pl.col("recall")).alias("recall_gap"))


def rolling_time_splits(
    observation_dates: list[date], min_train_months: int = 6, test_months: int = 1
) -> list[tuple[np.ndarray, np.ndarray]]:
    if min_train_months < 2 or test_months < 1:
        raise ValueError("ventana temporal inválida")
    months = np.array([value.year * 12 + value.month for value in observation_dates])
    unique_months = np.unique(months)
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for start in range(min_train_months, len(unique_months) - test_months + 1):
        train_months = unique_months[:start]
        test_window = unique_months[start : start + test_months]
        splits.append((np.flatnonzero(np.isin(months, train_months)), np.flatnonzero(np.isin(months, test_window))))
    return splits


def evaluate_governance(
    actual: np.ndarray,
    champion_probability: np.ndarray,
    challenger_probability: np.ndarray,
    segments: np.ndarray,
    reference_probability: np.ndarray,
    output_dir: Path,
    capacity_ratio: float = 0.20,
) -> dict[str, object]:
    if not (
        len(actual) == len(champion_probability) == len(challenger_probability) == len(segments)
    ):
        raise ValueError("actual, probabilidades y segmentos deben tener la misma longitud")
    output_dir.mkdir(parents=True, exist_ok=True)
    decision = select_capacity_threshold(actual, champion_probability, capacity_ratio=capacity_ratio)
    champion = {
        "roc_auc": float(roc_auc_score(actual, champion_probability)),
        "brier_score": float(brier_score_loss(actual, champion_probability)),
        "ece": expected_calibration_error(actual, champion_probability),
    }
    challenger = {
        "roc_auc": float(roc_auc_score(actual, challenger_probability)),
        "brier_score": float(brier_score_loss(actual, challenger_probability)),
        "ece": expected_calibration_error(actual, challenger_probability),
    }
    selected_model = (
        "challenger"
        if challenger["roc_auc"] > champion["roc_auc"] + 0.005
        and challenger["brier_score"] <= champion["brier_score"]
        else "champion"
    )
    drift_psi = population_stability_index(reference_probability, champion_probability)
    segments_frame = segment_performance(actual, champion_probability, segments, decision.threshold)
    segments_frame.write_csv(output_dir / "segment_performance.csv")

    registry = {
        "selected_model": selected_model,
        "champion": champion,
        "challenger": challenger,
        "threshold": decision.__dict__,
        "monitoring": {
            "probability_psi": drift_psi,
            "drift_level": "high" if drift_psi >= 0.25 else "watch" if drift_psi >= 0.10 else "stable",
            "maximum_segment_recall_gap": float(segments_frame["recall_gap"].max()),
        },
    }
    serialized = json.dumps(registry, indent=2, sort_keys=True)
    (output_dir / "governance_report.json").write_text(serialized, encoding="utf-8")
    manifest = {
        "model_version": "2026.08",
        "evaluation_rows": len(actual),
        "report_sha256": hashlib.sha256(serialized.encode("utf-8")).hexdigest(),
        "decision_owner": "retention-analytics",
        "automatic_action_allowed": False,
    }
    (output_dir / "model_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True), encoding="utf-8"
    )
    return registry

