"""EDA correlation analysis. This is exploratory, not used for CV feature selection."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def correlation_analysis(
    df: pd.DataFrame,
    target_col: str | None = None,
    thresholds: tuple[float, ...] = (0.80, 0.90, 0.95),
) -> dict[str, Any]:
    numeric = df.select_dtypes(include=[np.number])
    if numeric.shape[1] < 2:
        return {
            "pearson": pd.DataFrame(),
            "spearman": pd.DataFrame(),
            "pairs": pd.DataFrame(),
            "target": pd.DataFrame(),
            "note": "EDA-only correlation. Feature selection uses training-fold statistics inside CV.",
        }

    pearson = numeric.corr(method="pearson")
    spearman = numeric.corr(method="spearman")
    pairs = []
    cols = list(pearson.columns)
    for i, a in enumerate(cols):
        for b in cols[i + 1 :]:
            p = pearson.loc[a, b]
            s = spearman.loc[a, b]
            if not np.isfinite(p):
                continue
            flags = [t for t in thresholds if abs(p) >= t]
            pairs.append(
                {
                    "feature_a": a,
                    "feature_b": b,
                    "pearson": float(p),
                    "spearman": float(s) if np.isfinite(s) else np.nan,
                    "strong_at": max(flags) if flags else None,
                }
            )
    pair_df = pd.DataFrame(pairs)
    if not pair_df.empty:
        pair_df = pair_df.reindex(pair_df["pearson"].abs().sort_values(ascending=False).index)

    target_df = pd.DataFrame()
    if target_col and target_col in numeric.columns:
        rows = []
        for col in numeric.columns:
            if col == target_col:
                continue
            rows.append(
                {
                    "feature": col,
                    "pearson": float(pearson.loc[col, target_col])
                    if np.isfinite(pearson.loc[col, target_col])
                    else np.nan,
                    "spearman": float(spearman.loc[col, target_col])
                    if np.isfinite(spearman.loc[col, target_col])
                    else np.nan,
                }
            )
        target_df = pd.DataFrame(rows)
        if not target_df.empty:
            target_df = target_df.reindex(target_df["pearson"].abs().sort_values(ascending=False).index)

    pos = pair_df.head(8) if not pair_df.empty else pd.DataFrame()
    neg = (
        pair_df.sort_values("pearson").head(8) if not pair_df.empty else pd.DataFrame()
    )
    return {
        "pearson": pearson,
        "spearman": spearman,
        "pairs": pair_df.reset_index(drop=True),
        "strongest_positive": pos,
        "strongest_negative": neg,
        "target": target_df.reset_index(drop=True),
        "note": "EDA-only correlation. Feature selection uses training-fold statistics inside CV.",
    }
