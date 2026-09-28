from __future__ import annotations

import pandas as pd

REQUIRED = ("note_id", "start", "end", "concept_id")


def as_span_table(df: pd.DataFrame) -> pd.DataFrame:
    """Renvoie une copie avec des colonnes de span typées. Les colonnes en trop sont gardées."""
    out = df.copy()
    missing = [c for c in REQUIRED if c not in out.columns]
    if missing:
        raise ValueError(f"missing columns {missing}; got {list(out.columns)}")
    out["note_id"] = out["note_id"].astype(str)
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    out["concept_id"] = out["concept_id"].astype(int)
    out = out.loc[out["start"] < out["end"]].copy()
    return out.reset_index(drop=True)


def submission_view(df: pd.DataFrame) -> pd.DataFrame:
    return as_span_table(df)[list(REQUIRED)].copy()
