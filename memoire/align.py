"""Appariement des mentions prédites aux mentions de référence."""
from __future__ import annotations

from typing import Optional

import pandas as pd

from .schema import as_span_table


def span_iou(start_a: int, end_a: int, start_b: int, end_b: int) -> float:
    inter = max(0, min(end_a, end_b) - max(start_a, start_b))
    union = (end_a - start_a) + (end_b - start_b) - inter
    if union <= 0:
        return 0.0
    return inter / union


def best_alignment(
    gold: pd.DataFrame,
    pred: pd.DataFrame,
    min_span_iou: float = 0.5,
) -> pd.DataFrame:
    """Une ligne par mention de référence, avec la meilleure prédiction qui se chevauche s'il y en a une."""
    gold_s = as_span_table(gold)
    pred_s = as_span_table(pred) if len(pred) else gold_s.iloc[0:0].copy()
    pred_by_note = {nid: g for nid, g in pred_s.groupby("note_id")}

    rows: list[dict] = []
    for gold_row in gold_s.itertuples(index=False):
        cands = pred_by_note.get(str(gold_row.note_id), pred_s.iloc[0:0])
        best: Optional[pd.Series] = None
        best_iou = 0.0
        for pred_row in cands.itertuples(index=False):
            iou = span_iou(
                start_a=int(gold_row.start),
                end_a=int(gold_row.end),
                start_b=int(pred_row.start),
                end_b=int(pred_row.end),
            )
            if iou < min_span_iou:
                continue
            if iou > best_iou:
                best_iou = iou
                best = pred_row
        rows.append(
            {
                "note_id": gold_row.note_id,
                "gold_start": int(gold_row.start),
                "gold_end": int(gold_row.end),
                "gold_concept_id": int(gold_row.concept_id),
                "pred_start": None if best is None else int(best.start),
                "pred_end": None if best is None else int(best.end),
                "pred_concept_id": None if best is None else int(best.concept_id),
                "span_iou": best_iou,
            }
        )
    return pd.DataFrame(rows)


def is_mention_hit(row: pd.Series) -> bool:
    if pd.isna(row.get("pred_concept_id")):
        return False
    return int(row["pred_concept_id"]) == int(row["gold_concept_id"])
