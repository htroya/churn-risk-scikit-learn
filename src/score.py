from __future__ import annotations

import argparse
import json
from datetime import UTC, datetime
from pathlib import Path

import joblib
import polars as pl

from src.train import FEATURES, build_dataset


def score_batch(model_path: Path, frame: pl.DataFrame) -> pl.DataFrame:
    missing = sorted(set(FEATURES) - set(frame.columns))
    if missing:
        raise ValueError(f"Scoring input is missing features: {missing}")
    bundle = joblib.load(model_path)
    probability = bundle["pipeline"].predict_proba(frame.select(FEATURES).to_numpy())[:, 1]
    threshold = float(bundle["threshold"])
    return frame.select("customer_id").with_columns(
        pl.Series("churn_probability", probability).round(6),
        pl.Series("predicted_churn", probability >= threshold),
        pl.Series(
            "risk_segment",
            [
                (
                    "critical"
                    if value >= 0.75
                    else "high"
                    if value >= 0.55
                    else "medium"
                    if value >= 0.30
                    else "low"
                )
                for value in probability
            ],
        ),
        pl.lit(bundle["model_version"]).alias("model_version"),
        pl.lit(datetime.now(UTC)).alias("scored_at"),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch scoring for churn risk")
    parser.add_argument("--model", type=Path, default=Path("artifacts/churn_model.joblib"))
    parser.add_argument("--input", type=Path)
    parser.add_argument("--output", type=Path, default=Path("artifacts/churn_scores.parquet"))
    parser.add_argument("--rows", type=int, default=1_000)
    args = parser.parse_args()

    frame = pl.read_parquet(args.input) if args.input else build_dataset(args.rows).drop("churn")
    scores = score_batch(args.model, frame)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    scores.write_parquet(args.output, compression="zstd")
    print(
        json.dumps(
            {
                "rows": scores.height,
                "high_or_critical": scores.filter(
                    pl.col("risk_segment").is_in(["high", "critical"])
                ).height,
                "output": str(args.output),
            }
        )
    )


if __name__ == "__main__":
    main()
