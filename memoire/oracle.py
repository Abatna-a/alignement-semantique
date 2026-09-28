from __future__ import annotations

import pandas as pd

from .align import best_alignment, is_mention_hit
from .schema import as_span_table, submission_view


def attach_system_hits(
    gold: pd.DataFrame,
    dict_pred: pd.DataFrame,
    sno_pred: pd.DataFrame,
    min_span_iou: float = 0.5,
) -> pd.DataFrame:
    gold_s = as_span_table(gold)
    d_al = best_alignment(gold=gold_s, pred=dict_pred, min_span_iou=min_span_iou)
    s_al = best_alignment(gold=gold_s, pred=sno_pred, min_span_iou=min_span_iou)
    out = gold_s[["note_id", "start", "end", "concept_id"]].copy()
    out = out.rename(
        columns={"start": "gold_start", "end": "gold_end", "concept_id": "gold_concept_id"}
    )
    out["d_start"] = d_al["pred_start"]
    out["d_end"] = d_al["pred_end"]
    out["d_concept_id"] = d_al["pred_concept_id"]
    out["d_span_iou"] = d_al["span_iou"]
    out["s_start"] = s_al["pred_start"]
    out["s_end"] = s_al["pred_end"]
    out["s_concept_id"] = s_al["pred_concept_id"]
    out["s_span_iou"] = s_al["span_iou"]
    out["d_hit"] = d_al.apply(is_mention_hit, axis=1)
    out["s_hit"] = s_al.apply(is_mention_hit, axis=1)
    out["oracle_hit"] = out["d_hit"] | out["s_hit"]
    return out


def oracle_submission(aligned: pd.DataFrame) -> pd.DataFrame:
    """Émet la bonne réponse pour chaque mention de référence qu'au moins un système a trouvée."""
    rows: list[dict] = []
    for row in aligned.itertuples(index=False):
        if not (row.d_hit or row.s_hit):
            continue
        take_dict = bool(row.d_hit) and (
            (not row.s_hit) or float(row.d_span_iou) >= float(row.s_span_iou)
        )
        if take_dict:
            start, end, concept_id, source = row.d_start, row.d_end, row.d_concept_id, "dict"
        else:
            start, end, concept_id, source = row.s_start, row.s_end, row.s_concept_id, "snobert"
        rows.append(
            {
                "note_id": row.note_id,
                "start": int(start),
                "end": int(end),
                "concept_id": int(concept_id),
                "source": source,
            }
        )
    if not rows:
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id", "source"])
    return submission_view(pd.DataFrame(rows))


def complementarity_delta(dict_score: float, sno_score: float, oracle_score: float) -> float:
    return oracle_score - max(dict_score, sno_score)


def oracle_dominates(dict_score: float, sno_score: float, oracle_score: float) -> bool:
    """Invariant : l'oracle est au moins aussi bon que chaque système (au bruit flottant près)."""
    return oracle_score + 1e-12 >= dict_score and oracle_score + 1e-12 >= sno_score
