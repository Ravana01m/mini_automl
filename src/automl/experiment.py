"""Lightweight local experiment history (JSON + SQLite)."""

from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

EXPERIMENT_DIR = Path("experiments")
JSON_PATH = EXPERIMENT_DIR / "history.jsonl"
DB_PATH = EXPERIMENT_DIR / "history.sqlite"


def _ensure() -> None:
    EXPERIMENT_DIR.mkdir(parents=True, exist_ok=True)


def log_experiment(pipeline: Any, elapsed_s: float) -> dict[str, Any]:
    _ensure()
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "dataset_shape": None,
        "task": pipeline.task_type_,
        "target": pipeline.target_col_,
        "best_model": pipeline.trainer_.best_model_name_ if pipeline.trainer_ else None,
        "metrics": pipeline.advanced_metrics_,
        "baseline_metrics": pipeline.baseline_metrics_,
        "training_time_s": round(elapsed_s, 2),
        "feature_count": None,
        "selected_features": (pipeline.feature_report_ or {}).get("selected_features"),
        "configuration": pipeline.config.to_dict() if pipeline.config else None,
    }
    if pipeline.profile_ is not None:
        record["dataset_shape"] = [pipeline.profile_.n_rows, pipeline.profile_.n_cols]
        record["feature_count"] = pipeline.profile_.n_cols - 1
    with JSON_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, default=str) + "\n")
    _write_sqlite(record)
    return record


def _write_sqlite(record: dict[str, Any]) -> None:
    conn = sqlite3.connect(DB_PATH)
    try:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS experiments (
                timestamp TEXT,
                task TEXT,
                target TEXT,
                best_model TEXT,
                training_time_s REAL,
                metrics TEXT,
                configuration TEXT
            )
            """
        )
        conn.execute(
            """
            INSERT INTO experiments VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record["timestamp"],
                record["task"],
                record["target"],
                record["best_model"],
                record["training_time_s"],
                json.dumps(record.get("metrics"), default=str),
                json.dumps(record.get("configuration"), default=str),
            ),
        )
        conn.commit()
    finally:
        conn.close()


def load_history(limit: int = 50) -> list[dict[str, Any]]:
    if not JSON_PATH.exists():
        return []
    rows = []
    for line in JSON_PATH.read_text(encoding="utf-8").splitlines()[-limit:]:
        if line.strip():
            rows.append(json.loads(line))
    return list(reversed(rows))
