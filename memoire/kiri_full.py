"""Dictionnaire KIRIs complet sur le train de l'oracle (cutmed), sans fuite hold-out.

mock_train et la partie majuscules tournent sur cutmed seulement. Les clés OMOP /
unigrammes / permutations sont reprises de demo_dict/assets/kiri_dicts.pkl après
retrait des mentions présentes dans l'or hold-out mais jamais dans l'or cutmed.
L'inférence utilise la table officielle, les abréviations et le post-traitement
d'extension de termes.
"""

from __future__ import annotations

import pickle
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from .paths import (
    ABBR_DICT,
    DICTMETHOD_SRC,
    FLAT_SNOMED,
    KIRI_FULL_CACHE,
    LEAKED_KIRI_DICTS,
    OUTPUT_DIR,
    TERM_EXTENSION,
)
from .schema import submission_view

if str(DICTMETHOD_SRC) not in sys.path:
    sys.path.insert(0, str(DICTMETHOD_SRC))

from mimic_common import common_headers  # noqa: E402
from mimic_postprocess_attributes import postprocess_annotations  # noqa: E402
from mimic_predict import get_case_sensitive_dict, join_predictions, predict  # noqa: E402
from mimic_train import (  # noqa: E402
    extract_uppercase_mentions,
    get_allowed_sections,
    limit_any_to_allowed_sections,
    mock_train,
    words_counter,
)


def _norm_mention(text: str) -> str:
    return " ".join(str(text).lower().split())


def key_mention(key: object) -> str:
    if isinstance(key, tuple) and len(key) >= 2:
        return _norm_mention(key[1])
    return _norm_mention(key)


def add_source_column(notes: pd.DataFrame, annotations: pd.DataFrame) -> pd.DataFrame:
    texts = notes.set_index("note_id")["text"].astype(str)
    sources: list[str] = []
    for row in annotations.itertuples(index=False):
        note_id = str(row.note_id)
        start = int(row.start)
        end = int(row.end)
        raw = texts.loc[note_id][start:end]
        sources.append(" ".join(raw.split()))
    out = annotations.copy()
    out["source orig"] = sources
    out["source"] = [_norm_mention(s) for s in sources]
    return out


def test_only_mentions(train_ann: pd.DataFrame, test_ann: pd.DataFrame) -> set[str]:
    train_m = set(train_ann["source"].map(_norm_mention))
    test_m = set(test_ann["span"].astype(str).map(_norm_mention))
    return test_m - train_m


def _cid_to_type(flat_path: Path) -> pd.Series:
    sno_fsn = (
        pd.read_csv(flat_path)
        .drop_duplicates("concept_name")
        .set_index("concept_id")["concept_name"]
    )
    replacements = {
        "procedure": "procedure",
        "body structure": "body structure",
        "disorder": "finding",
        "finding": "finding",
        "morphologic abnormality": "body structure",
        "cell structure": "body structure",
        "regime/therapy": "finding",
    }
    return sno_fsn.apply(lambda x: str(x).split("(")[-1][:-1]).replace(replacements)


def _merge_external(
    dest: dict,
    source: dict,
    blocked_mentions: set[str],
) -> int:
    n_added = 0
    for key, value in source.items():
        if key in dest:
            continue
        if key_mention(key) in blocked_mentions:
            continue
        dest[key] = int(value)
        n_added += 1
    return n_added


def build_kiri_full(
    train_notes: pd.DataFrame,
    train_ann: pd.DataFrame,
    test_ann: pd.DataFrame,
    leaked_path: Path = LEAKED_KIRI_DICTS,
) -> tuple[dict, dict]:
    annotations = add_source_column(notes=train_notes, annotations=train_ann)
    texts = train_notes.set_index("note_id")["text"].astype(str)
    texts_lc = texts.str.lower()
    headers = [h.lower() for h in common_headers]

    words_counter.clear()
    words_counter.update(Counter("\n".join(texts_lc).split()))

    d, _scores_by_note, _scores_by_mention = mock_train(
        texts_lc, annotations, headers, run_name="cutmed_full"
    )
    uc_d = extract_uppercase_mentions(d, annotations)

    with leaked_path.open("rb") as handle:
        leaked_d, leaked_uc = pickle.load(handle)

    blocked = test_only_mentions(train_ann=annotations, test_ann=test_ann)
    n_d = _merge_external(dest=d, source=leaked_d, blocked_mentions=blocked)
    n_uc = _merge_external(dest=uc_d, source=leaked_uc, blocked_mentions=blocked)
    print(f"merged leaked keys: d+{n_d} uc+{n_uc}; blocked test-only mentions {len(blocked)}")

    cid_types = _cid_to_type(FLAT_SNOMED)
    try:
        allowed_sec = get_allowed_sections(texts_lc, annotations, headers, cid_types)
        limit_any_to_allowed_sections(d, allowed_sec, cid_types)
    except KeyError as exc:
        print(f"skip limit_any_to_allowed_sections ({exc})")
    return d, uc_d


def save_kiri_full(path: Path, d: dict, uc_d: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump((d, uc_d), handle)


def load_kiri_full(path: Path) -> tuple[dict, dict]:
    with path.open("rb") as handle:
        d, uc_d = pickle.load(handle)
    return d, uc_d


def predict_kiri_full(notes: pd.DataFrame, d: dict, uc_d: dict) -> pd.DataFrame:
    texts = notes.set_index("note_id")["text"].astype(str)
    headers = [h.lower() for h in common_headers]
    pred_lc = predict(texts.str.lower(), headers, d, submission=True, run_name="holdout_lc")
    uc_merged = dict(uc_d)
    uc_merged.update(get_case_sensitive_dict())
    with ABBR_DICT.open("rb") as handle:
        abbr_dict = pickle.load(handle)
    abbr_dict.update(uc_merged)
    pred_uc = predict(texts, common_headers, abbr_dict, submission=True, run_name="holdout_uc")
    pred = join_predictions((pred_lc, pred_uc))
    att_df = pd.read_csv(TERM_EXTENSION) if TERM_EXTENSION.exists() else pd.DataFrame()
    if not att_df.empty:
        pred = postprocess_annotations(texts, pred, att_df, submission=True)
    return submission_view(pred)


def get_or_build_kiri_full(
    train_notes: pd.DataFrame,
    train_ann: pd.DataFrame,
    test_ann: pd.DataFrame,
    cache_path: Path = KIRI_FULL_CACHE,
    force: bool = False,
) -> tuple[dict, dict]:
    if cache_path.exists() and not force:
        return load_kiri_full(cache_path)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    d, uc_d = build_kiri_full(
        train_notes=train_notes, train_ann=train_ann, test_ann=test_ann
    )
    save_kiri_full(cache_path, d, uc_d)
    print(f"saved {cache_path} d={len(d)} uc={len(uc_d)}")
    return d, uc_d
