from __future__ import annotations

import pandas as pd

from .dictionary import normalize_term
from .schema import as_span_table, submission_view


def snobert_neural_residual(
    snobert_full: pd.DataFrame,
    notes: pd.DataFrame,
    term_to_cid: dict[str, int],
) -> pd.DataFrame:
    """Spans SnoBERT dont la surface n'est pas dans le dictionnaire statique d'apprentissage.

    Un vrai rerun SapBERT seul exige les poids NER/SapBERT (absents du disque).
    On garde la longue traîne dense : mentions que le dictionnaire SnoBERT ne peut
    pas écraser. Les spans dont le texte est dans le dictionnaire gardent l'ID
    (éventuellement écrasé) et sont retirés, même si le NER les a aussi trouvés.
    """
    full = as_span_table(snobert_full)
    texts = notes.set_index("note_id")["text"].astype(str)
    keep_idx: list[int] = []
    for i, row in full.iterrows():
        mention = normalize_term(texts.loc[row.note_id][int(row.start) : int(row.end)])
        dict_cid = term_to_cid.get(mention)
        if dict_cid is None:
            keep_idx.append(i)
            continue
        if int(row.concept_id) != int(dict_cid):
            keep_idx.append(i)
    if not keep_idx:
        return full.iloc[0:0][["note_id", "start", "end", "concept_id"]].copy()
    return submission_view(full.loc[keep_idx])
