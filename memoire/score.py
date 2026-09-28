from __future__ import annotations

"""Scores mIoU caractère et rappel au niveau des mentions."""
import pandas as pd

from .align import best_alignment, is_mention_hit
from .paths import SNOBERT
from .schema import as_span_table, submission_view

import sys

if str(SNOBERT / "submission") not in sys.path:
    sys.path.insert(0, str(SNOBERT / "submission"))

from iou import iou_per_class  # noqa: E402


def mean_char_iou(pred: pd.DataFrame, gold: pd.DataFrame) -> float:
    pred_s = submission_view(pred)
    gold_s = submission_view(gold)
    if pred_s.empty:
        return 0.0
    score = iou_per_class(pred_s, gold_s, mean=True)
    return float(score)


def mention_hits(
    gold: pd.DataFrame, pred: pd.DataFrame, min_span_iou: float = 0.5
) -> pd.Series:
    gold_s = as_span_table(gold)
    pred_s = as_span_table(pred) if len(pred) else gold_s.iloc[0:0].copy()
    aligned = best_alignment(gold=gold_s, pred=pred_s, min_span_iou=min_span_iou)
    return aligned.apply(is_mention_hit, axis=1)


def mention_recall(
    gold: pd.DataFrame, pred: pd.DataFrame, min_span_iou: float = 0.5
) -> float:
    hits = mention_hits(gold=gold, pred=pred, min_span_iou=min_span_iou)
    if len(hits) == 0:
        return 0.0
    return float(hits.mean())
