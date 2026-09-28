"""Télécharge les jeux PARHAF annotés sur Hugging Face et écrit des CSV locaux."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import pandas as pd
from huggingface_hub import hf_hub_download

ROOT = Path(__file__).resolve().parent
import sys

sys.path.insert(0, str(ROOT))
from io_flat import write_csv_oneline
PARQUET_DIR = ROOT / "parquet"
CSV_DIR = ROOT / "csv"

SPAN_LABEL_COLUMNS: tuple[str, ...] = (
    "attribute_LABEL",
    "attribute_Categorie",
    "attribute_Justification",
    "attribute_Conclusion",
    "attribute_Nomenclature",
    "span_type",
)

DOC_LABEL_COLUMNS: tuple[str, ...] = (
    "attribute_Nomenclature",
    "annotation_type",
)


@dataclass(frozen=True)
class DatasetSpec:
    name: str
    repo: str
    tables: tuple[str, ...]
    expert_splits: tuple[str, ...]


DATASETS: tuple[DatasetSpec, ...] = (
    DatasetSpec(
        name="pseudo",
        repo="HealthDataHub/PARHAF-pseudo-annotated",
        tables=("document_metadata", "spans"),
        expert_splits=("train", "dev"),
    ),
    DatasetSpec(
        name="infectiology",
        repo="HealthDataHub/PARHAF-infectiology-annotated",
        tables=("document_metadata", "spans", "relations"),
        expert_splits=("train", "dev"),
    ),
    DatasetSpec(
        name="biomarkers",
        repo="HealthDataHub/PARHAF-biomarkers-annotated",
        tables=("document_metadata", "spans", "relations"),
        expert_splits=("train", "dev"),
    ),
    DatasetSpec(
        name="rtt",
        repo="HealthDataHub/PARHAF-response_to_treatment-annotated",
        tables=("document_metadata", "spans"),
        expert_splits=("train", "dev"),
    ),
    DatasetSpec(
        name="conclusion",
        repo="HealthDataHub/PARHAF-conclusion-annotated",
        tables=("document_metadata", "spans"),
        expert_splits=("dev",),
    ),
)


def parquet_filename(table: str, split: str) -> str:
    return f"{table}_{split}.parquet"


def download_parquet(spec: DatasetSpec, table: str, split: str) -> Path:
    relative_path = f"data/{table}/{parquet_filename(table, split)}"
    local_dir = PARQUET_DIR / spec.name
    local_dir.mkdir(parents=True, exist_ok=True)
    downloaded = hf_hub_download(
        repo_id=spec.repo,
        filename=relative_path,
        repo_type="dataset",
        local_dir=str(local_dir),
    )
    return Path(downloaded)


def first_non_empty(row: pd.Series, columns: tuple[str, ...]) -> str:
    for column in columns:
        if column not in row.index:
            continue
        value = row[column]
        if pd.isna(value):
            continue
        text = str(value).strip()
        if text and text.lower() != "nan":
            return text
    return ""


def with_split_and_source(
    frame: pd.DataFrame, spec: DatasetSpec, split: str
) -> pd.DataFrame:
    out = frame.copy()
    out["dataset"] = spec.name
    out["split"] = split
    out["is_expert"] = split in spec.expert_splits
    out["hf_repo"] = spec.repo
    return out


def load_table(spec: DatasetSpec, table: str) -> pd.DataFrame:
    parts: list[pd.DataFrame] = []
    for split in ("train", "dev"):
        path = download_parquet(spec=spec, table=table, split=split)
        part = pd.read_parquet(path)
        parts.append(with_split_and_source(frame=part, spec=spec, split=split))
    return pd.concat(parts, ignore_index=True)


def write_csv(frame: pd.DataFrame, path: Path) -> None:
    write_csv_oneline(frame=frame, path=path)


def unify_documents(frame: pd.DataFrame) -> pd.DataFrame:
    label = frame.apply(
        lambda row: first_non_empty(row, DOC_LABEL_COLUMNS), axis=1
    )
    columns = ["dataset", "split", "is_expert", "hf_repo", "report", "full_text"]
    extra = [col for col in frame.columns if col not in columns]
    unified = frame.loc[:, [col for col in columns if col in frame.columns]].copy()
    unified["doc_label"] = label
    for col in extra:
        if col not in unified.columns:
            unified[col] = frame[col]
    return unified


def unify_spans(frame: pd.DataFrame) -> pd.DataFrame:
    label = frame.apply(
        lambda row: first_non_empty(row, SPAN_LABEL_COLUMNS), axis=1
    )
    begin = frame["begin"] if "begin" in frame.columns else pd.NA
    end = frame["end"] if "end" in frame.columns else pd.NA
    unified = pd.DataFrame(
        {
            "dataset": frame["dataset"],
            "split": frame["split"],
            "is_expert": frame["is_expert"],
            "hf_repo": frame["hf_repo"],
            "note_id": frame["report"],
            "start": begin,
            "end": end,
            "span": frame["span_text"] if "span_text" in frame.columns else "",
            "label": label,
            "span_type": frame["span_type"] if "span_type" in frame.columns else "",
            "span_id": frame["span_id"] if "span_id" in frame.columns else pd.NA,
        }
    )
    extra_cols = [
        col
        for col in frame.columns
        if col.startswith("attribute_") and col not in unified.columns
    ]
    for col in extra_cols:
        unified[col] = frame[col]
    return unified


def count_rows(frame: pd.DataFrame, name: str, table: str) -> pd.DataFrame:
    grouped = (
        frame.groupby(["dataset", "split", "is_expert"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    grouped["table"] = table
    grouped["file"] = name
    return grouped


def main() -> None:
    CSV_DIR.mkdir(parents=True, exist_ok=True)
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)

    all_docs: list[pd.DataFrame] = []
    all_spans: list[pd.DataFrame] = []
    all_rels: list[pd.DataFrame] = []
    manifest_parts: list[pd.DataFrame] = []

    for spec in DATASETS:
        print(f"Fetching {spec.name} from {spec.repo}")
        for table in spec.tables:
            frame = load_table(spec=spec, table=table)
            out_name = f"{spec.name}_{table}.csv"
            write_csv(frame=frame, path=CSV_DIR / out_name)
            manifest_parts.append(
                count_rows(frame=frame, name=out_name, table=table)
            )
            print(f"  {table}: {len(frame)} rows -> {out_name}")
            if table == "document_metadata":
                all_docs.append(unify_documents(frame))
            elif table == "spans":
                all_spans.append(unify_spans(frame))
            elif table == "relations":
                all_rels.append(frame)

    documents = pd.concat(all_docs, ignore_index=True)
    spans = pd.concat(all_spans, ignore_index=True)
    write_csv(frame=documents, path=CSV_DIR / "all_documents.csv")
    write_csv(frame=spans, path=CSV_DIR / "all_spans.csv")
    write_csv(
        frame=spans.loc[spans["is_expert"]].copy(),
        path=CSV_DIR / "expert_spans.csv",
    )
    if all_rels:
        relations = pd.concat(all_rels, ignore_index=True)
        write_csv(frame=relations, path=CSV_DIR / "all_relations.csv")
        manifest_parts.append(
            count_rows(frame=relations, name="all_relations.csv", table="relations")
        )

    manifest_parts.append(
        count_rows(frame=documents, name="all_documents.csv", table="documents")
    )
    manifest_parts.append(
        count_rows(frame=spans, name="all_spans.csv", table="spans")
    )
    manifest = pd.concat(manifest_parts, ignore_index=True)
    write_csv(frame=manifest, path=CSV_DIR / "manifest.csv")

    print("\nDone.")
    print(f"Documents: {len(documents)}")
    print(f"Spans: {len(spans)} (expert={int(spans['is_expert'].sum())})")
    print(f"CSV dir: {CSV_DIR}")


if __name__ == "__main__":
    main()
