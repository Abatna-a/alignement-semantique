"""Reconstruit les tables PARHAF en une ligne par ligne, y compris un clone de train_annotations.csv."""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))
from io_flat import collapse_ws, write_csv_oneline, write_jsonl
CSV_DIR = ROOT / "csv"
FLAT_DIR = ROOT / "flat"


def load_existing_csv(name: str) -> pd.DataFrame:
    return pd.read_csv(CSV_DIR / name)


def build_demo_annotations(spans: pd.DataFrame) -> pd.DataFrame:
    expert = spans.loc[spans["is_expert"]].copy()
    expert = expert.sort_values(
        by=["dataset", "note_id", "start", "end"],
        kind="mergesort",
    ).reset_index(drop=True)
    demo = pd.DataFrame(
        {
            "annotation_id": range(1, len(expert) + 1),
            "note_id": expert["note_id"].astype(str),
            "start": expert["start"].astype(int),
            "end": expert["end"].astype(int),
            "span": expert["span"].map(collapse_ws),
            "concept_id": expert["label"].map(collapse_ws),
            "annotation_type": expert["dataset"].astype(str),
        }
    )
    return demo


def build_rich_annotations(spans: pd.DataFrame) -> pd.DataFrame:
    keep = [
        "dataset",
        "split",
        "is_expert",
        "note_id",
        "start",
        "end",
        "span",
        "label",
        "span_type",
        "span_id",
        "attribute_NEGATION",
        "attribute_Categorie",
        "attribute_LABEL",
        "attribute_Normalized",
        "attribute_Presence",
        "attribute_Biomarker",
        "attribute_Justification",
        "attribute_Conclusion",
    ]
    present = [col for col in keep if col in spans.columns]
    rich = spans.loc[:, present].copy()
    rich["start"] = rich["start"].astype(int)
    rich["end"] = rich["end"].astype(int)
    rich["span"] = rich["span"].map(collapse_ws)
    return rich.sort_values(
        by=["dataset", "note_id", "start"],
        kind="mergesort",
    ).reset_index(drop=True)


def build_notes_jsonl(documents: pd.DataFrame) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for row in documents.itertuples(index=False):
        rows.append(
            {
                "note_id": str(row.report),
                "dataset": str(row.dataset),
                "split": str(row.split),
                "is_expert": bool(row.is_expert),
                "doc_label": collapse_ws(getattr(row, "doc_label", "")),
                "text": "" if pd.isna(row.full_text) else str(row.full_text),
            }
        )
    return rows


def rewrite_csv_dir_oneline() -> None:
    for path in sorted(CSV_DIR.glob("*.csv")):
        frame = pd.read_csv(path)
        write_csv_oneline(frame=frame, path=path)
        print(f"rewrote {path.name}: {len(frame)} rows")


def main() -> None:
    FLAT_DIR.mkdir(parents=True, exist_ok=True)
    spans = load_existing_csv("all_spans.csv")
    documents = load_existing_csv("all_documents.csv")
    relations = load_existing_csv("all_relations.csv")

    demo = build_demo_annotations(spans)
    write_csv_oneline(frame=demo, path=FLAT_DIR / "train_annotations.csv")
    for dataset_name, part in demo.groupby("annotation_type", sort=True):
        part = part.reset_index(drop=True)
        part["annotation_id"] = range(1, len(part) + 1)
        write_csv_oneline(
            frame=part,
            path=FLAT_DIR / f"train_annotations_{dataset_name}.csv",
        )

    rich = build_rich_annotations(spans)
    write_csv_oneline(frame=rich, path=FLAT_DIR / "parhaf_annotations.csv")
    write_csv_oneline(
        frame=rich.loc[rich["is_expert"]].copy(),
        path=FLAT_DIR / "parhaf_annotations_expert.csv",
    )
    write_csv_oneline(frame=relations, path=FLAT_DIR / "parhaf_relations.csv")

    note_rows = build_notes_jsonl(documents)
    write_jsonl(rows=note_rows, path=FLAT_DIR / "train_notes.jsonl")
    notes = pd.DataFrame(note_rows)
    write_csv_oneline(frame=notes, path=FLAT_DIR / "train_notes.csv")

    rewrite_csv_dir_oneline()

    print("\nflat files:")
    print(f"  {FLAT_DIR / 'train_annotations.csv'}  n={len(demo)}")
    print(f"  {FLAT_DIR / 'train_notes.jsonl'}     n={len(note_rows)}")
    print("concept_id here is the PARHAF label, not a SNOMED SCTID.")


if __name__ == "__main__":
    main()
