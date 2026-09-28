"""Export tabulaire une ligne par ligne : pas de retours à la ligne bruts dans les cellules."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def flatten_cell(value: object) -> object:
    if not isinstance(value, str):
        return value
    return (
        value.replace("\r\n", "\\n")
        .replace("\n", "\\n")
        .replace("\r", "\\n")
        .replace("\t", " ")
    )


def collapse_ws(value: object) -> str:
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return ""
    return " ".join(str(value).split())


def flatten_frame(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    for column in out.columns:
        out[column] = out[column].map(flatten_cell)
    return out


def write_csv_oneline(frame: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    flatten_frame(frame).to_csv(path, index=False, encoding="utf-8")


def write_jsonl(rows: list[dict[str, object]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
