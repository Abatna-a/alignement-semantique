from __future__ import annotations

import pandas as pd

QUEUE_REGIMES = ("CONFLIT", "D_SEUL", "S_SEUL", "AUCUN")
AUTO_REGIME = "ACCORD"


def in_review_queue(regime: str) -> bool:
    return regime in QUEUE_REGIMES


def add_queue_flag(aligned: pd.DataFrame) -> pd.DataFrame:
    out = aligned.copy()
    out["in_queue"] = out["regime"].map(in_review_queue)
    out["is_error"] = ~out["oracle_hit"].astype(bool)
    return out


def queue_metrics(aligned: pd.DataFrame) -> dict[str, float]:
    """File orientée rappel : on émet tout, on relit d'abord ce qui n'est pas en ACCORD."""
    flagged = add_queue_flag(aligned)
    n = max(len(flagged), 1)
    n_err = int(flagged["is_error"].sum())
    n_queue = int(flagged["in_queue"].sum())
    n_err_queue = int((flagged["in_queue"] & flagged["is_error"]).sum())
    n_err_accord = int((~flagged["in_queue"] & flagged["is_error"]).sum())
    return {
        "n_mentions": float(n),
        "n_errors": float(n_err),
        "n_queue": float(n_queue),
        "queue_coverage": n_queue / n,
        "error_rate_all": n_err / n,
        "error_recall": (n_err_queue / n_err) if n_err else 1.0,
        "error_precision": (n_err_queue / n_queue) if n_queue else 0.0,
        "n_errors_in_accord": float(n_err_accord),
        "accord_error_share": (n_err_accord / n_err) if n_err else 0.0,
    }


def queue_by_regime(aligned: pd.DataFrame) -> pd.DataFrame:
    flagged = add_queue_flag(aligned)
    n_err = max(int(flagged["is_error"].sum()), 1)
    rows = []
    for regime, group in flagged.groupby("regime"):
        n = len(group)
        n_e = int(group["is_error"].sum())
        rows.append(
            {
                "regime": regime,
                "in_queue": in_review_queue(str(regime)),
                "n": n,
                "n_errors": n_e,
                "p_error": n_e / n if n else 0.0,
                "share_of_all_errors": n_e / n_err,
            }
        )
    table = pd.DataFrame(rows)
    order = ["ACCORD", "CONFLIT", "D_SEUL", "S_SEUL", "AUCUN"]
    table["regime"] = pd.Categorical(table["regime"], categories=order, ordered=True)
    return table.sort_values("regime").reset_index(drop=True)


def sample_examples(aligned: pd.DataFrame, per_regime: int = 2, seed: int = 0) -> pd.DataFrame:
    flagged = add_queue_flag(aligned)
    cols = [
        "note_id",
        "gold_start",
        "gold_end",
        "gold_concept_id",
        "d_concept_id",
        "s_concept_id",
        "d_hit",
        "s_hit",
        "oracle_hit",
        "regime",
        "in_queue",
        "is_error",
    ]
    present = [c for c in cols if c in flagged.columns]
    parts = []
    for regime, group in flagged.groupby("regime"):
        take = min(per_regime, len(group))
        if take == 0:
            continue
        parts.append(group.sample(n=take, random_state=seed)[present])
    if not parts:
        return flagged.iloc[0:0]
    return pd.concat(parts, ignore_index=True)
