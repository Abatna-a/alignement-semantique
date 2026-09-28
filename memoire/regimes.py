from __future__ import annotations

"""Régimes d'accord / conflit / silence entre les deux moteurs."""
import pandas as pd


def mention_regime(row: pd.Series) -> str:
    d_span = pd.notna(row.get("d_concept_id"))
    s_span = pd.notna(row.get("s_concept_id"))
    if d_span and s_span:
        if int(row["d_concept_id"]) == int(row["s_concept_id"]):
            return "ACCORD"
        return "CONFLIT"
    if d_span:
        return "D_SEUL"
    if s_span:
        return "S_SEUL"
    return "AUCUN"


def add_regimes(aligned: pd.DataFrame) -> pd.DataFrame:
    out = aligned.copy()
    out["regime"] = out.apply(mention_regime, axis=1)
    return out


def regime_table(aligned: pd.DataFrame) -> pd.DataFrame:
    rows = []
    n_all = max(len(aligned), 1)
    for regime, group in aligned.groupby("regime"):
        n = len(group)
        rows.append(
            {
                "regime": regime,
                "n": n,
                "pct": n / n_all,
                "p_error": 1.0 - float(group["oracle_hit"].mean()) if n else 1.0,
                "d_only_correct": int((group["d_hit"] & ~group["s_hit"]).sum()),
                "s_only_correct": int((group["s_hit"] & ~group["d_hit"]).sum()),
                "both_correct": int((group["d_hit"] & group["s_hit"]).sum()),
            }
        )
    table = pd.DataFrame(rows)
    order = ["ACCORD", "CONFLIT", "D_SEUL", "S_SEUL", "AUCUN"]
    table["regime"] = pd.Categorical(table["regime"], categories=order, ordered=True)
    return table.sort_values("regime").reset_index(drop=True)
