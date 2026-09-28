"""Construit notes/annotations au format SnoBERT à partir de PARHAF infectiologie (NER seul).

Types SNOMED stage-1 vs PARHAF (pas de SCTID, pas de liaison) :

  Clinical finding  ~ Infection (et Bacteriemie est aussi un finding)
  Body structure    ~ Site (anatomie et certains labos sont Site ici)
  Procedure         ~ pas d'équivalent net ; Bacterie est un organisme, pas une procédure

Les balises BIO deviennent B-Infection / I-Infection / B-Site / ... plus O (9 classes).

Usage (depuis snobert/) :
  python src/prepare_parhaf_ner.py
  python src/main_parhaf.py
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
PARHAF_CSV = ROOT.parent / "parhaf" / "csv"
OUT_DIR = ROOT / "data" / "parhaf_infectiology"
ANN_COLS = ["note_id", "start", "end", "span", "concept_id", "annotation_id"]


# Découpage local des 134 CR (pas le split officiel PARHAF train/dev).
N_TRAIN = 100
N_VAL = 20
N_TEST = 14
SPLIT_SEED = 42


def restore_newlines(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return str(value).replace("\\n", "\n")


def assign_local_folds(notes: pd.DataFrame) -> pd.DataFrame:
    n = len(notes)
    expected = N_TRAIN + N_VAL + N_TEST
    if n != expected:
        raise ValueError(f"expected {expected} notes, got {n}")
    perm = np.random.RandomState(SPLIT_SEED).permutation(n)
    fold = np.empty(n, dtype=int)
    fold[perm[:N_TRAIN]] = 0
    fold[perm[N_TRAIN : N_TRAIN + N_VAL]] = 1
    fold[perm[N_TRAIN + N_VAL :]] = 2
    out = notes.copy()
    out["fold"] = fold
    return out


def main() -> None:
    docs = pd.read_csv(PARHAF_CSV / "infectiology_document_metadata.csv")
    spans = pd.read_csv(PARHAF_CSV / "infectiology_spans.csv")

    notes_rows: list[dict[str, object]] = []
    text_by_report: dict[str, str] = {}
    for row in docs.itertuples(index=False):
        text = restore_newlines(row.full_text)
        text_by_report[str(row.report)] = text
        notes_rows.append({"note_id": str(row.report), "text": text, "fold": -1})
    notes = pd.DataFrame(notes_rows)
    notes = assign_local_folds(notes)

    ann_rows: list[dict[str, object]] = []
    n_mismatch = 0
    for i, row in enumerate(spans.itertuples(index=False), start=1):
        note_id = str(row.report)
        text = text_by_report[note_id]
        start = int(row.begin)
        end = int(row.end)
        gold = restore_newlines(row.span_text)
        if text[start:end] != gold:
            n_mismatch += 1
        ann_rows.append(
            {
                "note_id": note_id,
                "start": start,
                "end": end,
                "span": gold,
                "concept_id": str(row.attribute_LABEL),
                "annotation_id": i,
            }
        )
    annotations = pd.DataFrame(ann_rows)
    if n_mismatch:
        raise ValueError(f"{n_mismatch} span offsets do not match restored text")
    annotations = sort_annotations(annotations, notes["note_id"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    notes.to_csv(OUT_DIR / "notes.csv", index=False)
    annotations.to_csv(OUT_DIR / "annotations.csv", index=False)
    write_split_csvs(notes, annotations, OUT_DIR)
    print(
        f"notes={len(notes)} "
        f"(train={(notes.fold==0).sum()} val={(notes.fold==1).sum()} "
        f"test={(notes.fold==2).sum()})"
    )
    print(f"annotations={len(annotations)}")
    print(annotations["concept_id"].value_counts().to_string())
    print(f"wrote {OUT_DIR}")
    print("BIO example: Infection -> B-Infection / I-Infection (not B-find)")


def sort_annotations(annotations: pd.DataFrame, note_ids: pd.Series) -> pd.DataFrame:
    """Groupe par CR dans l'ordre des notes, puis par offset croissant."""
    order = {str(nid): i for i, nid in enumerate(note_ids.astype(str))}
    out = annotations.copy()
    out["start"] = out["start"].astype(int)
    out["end"] = out["end"].astype(int)
    out["_ord"] = out["note_id"].astype(str).map(order)
    out = out.sort_values(["_ord", "start", "end"], kind="mergesort")
    return out[ANN_COLS].reset_index(drop=True)


def write_split_csvs(
    notes: pd.DataFrame, annotations: pd.DataFrame, out_dir: Path
) -> None:
    names = {0: "train", 1: "val", 2: "test"}
    for fold, name in names.items():
        notes_part = notes[notes.fold == fold].copy()
        ids = set(notes_part["note_id"].astype(str))
        ann_part = annotations[annotations["note_id"].astype(str).isin(ids)].copy()
        ann_part = sort_annotations(ann_part, notes_part["note_id"])
        notes_part.to_csv(out_dir / f"notes_{name}.csv", index=False)
        ann_part.to_csv(out_dir / f"annotations_{name}.csv", index=False)
        print(
            f"  {name}: notes={len(notes_part)} annotations={len(ann_part)} "
            f"-> notes_{name}.csv annotations_{name}.csv"
        )


if __name__ == "__main__":
    main()
