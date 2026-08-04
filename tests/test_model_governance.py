from datetime import date
from pathlib import Path

import numpy as np
import polars as pl
import pytest

from src.model_governance import (
    evaluate_governance,
    expected_calibration_error,
    population_stability_index,
    rolling_time_splits,
    select_capacity_threshold,
)


def test_calibration_drift_and_capacity_are_quantified() -> None:
    actual = np.array([0, 0, 0, 1, 1, 1, 1, 0, 1, 0])
    probability = np.array([0.05, 0.15, 0.25, 0.85, 0.75, 0.65, 0.95, 0.35, 0.55, 0.45])
    assert expected_calibration_error(actual, probability, bins=5) >= 0
    assert population_stability_index(probability, probability.copy()) == pytest.approx(0.0)
    decision = select_capacity_threshold(actual, probability, capacity_ratio=0.5)
    assert decision.selected <= decision.capacity == 5
    assert decision.expected_net_value > 0


def test_rolling_splits_never_mix_future_into_training() -> None:
    dates = [date(2025 + month // 12, month % 12 + 1, 1) for month in range(14)]
    splits = rolling_time_splits(dates, min_train_months=6, test_months=2)
    assert len(splits) == 7
    for train, test in splits:
        assert train.max() < test.min()


def test_governance_exports_registry_segments_and_drift(tmp_path: Path) -> None:
    rng = np.random.default_rng(42)
    actual = rng.binomial(1, 0.28, 800)
    champion = np.clip(actual * 0.55 + rng.normal(0.22, 0.16, 800), 0.001, 0.999)
    challenger = np.clip(actual * 0.58 + rng.normal(0.20, 0.14, 800), 0.001, 0.999)
    segments = np.array(["hogar", "pyme", "corporativo", "hogar"] * 200)
    reference = np.clip(rng.normal(0.30, 0.17, 1_000), 0.001, 0.999)

    report = evaluate_governance(
        actual, champion, challenger, segments, reference, tmp_path, capacity_ratio=0.25
    )
    assert report["selected_model"] in {"champion", "challenger"}
    assert report["threshold"]["selected"] <= report["threshold"]["capacity"]
    assert (tmp_path / "governance_report.json").exists()
    assert (tmp_path / "model_manifest.json").exists()
    segments_frame = pl.read_csv(tmp_path / "segment_performance.csv")
    assert set(segments_frame["segment"]) == {"hogar", "pyme", "corporativo"}
    assert segments_frame["recall_gap"].max() >= 0


def test_capacity_rejects_invalid_ratio() -> None:
    with pytest.raises(ValueError, match="entre 0 y 1"):
        select_capacity_threshold(np.array([0, 1]), np.array([0.1, 0.9]), capacity_ratio=0)

