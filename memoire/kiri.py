from __future__ import annotations

import pickle
import re
import sys
from collections import Counter
from pathlib import Path

import pandas as pd

from .paths import DICTMETHOD_SRC
from .schema import as_span_table, submission_view

if str(DICTMETHOD_SRC) not in sys.path:
    sys.path.insert(0, str(DICTMETHOD_SRC))

from mimic_common import (  # noqa: E402
    common_headers,
    get_header_by_pos,
    get_sections,
    internal_blacklist,
    remove_overlaps,
)

SECTION_ANY = "any"


def _mention_text(note: str, start: int, end: int) -> str:
    return " ".join(note[start:end].lower().split())


def _token_spans(text: str) -> list[tuple[str, int, int]]:
    return [(m.group(), m.start(), m.end()) for m in re.finditer(r"[a-z0-9]+", text.lower())]


def build_kiri_core(
    notes: pd.DataFrame,
    annotations: pd.DataFrame,
    min_count: int = 1,
) -> dict[tuple[str, str], int]:
    """Cœur KIRIs : (en-tête de section, mention) → SCTID majoritaire, sur l'apprentissage seulement.

    Recherche naïve gardée pour les tests. Les tableaux hold-out utilisent kiri_full.py.
    """
    texts = notes.set_index("note_id")["text"].astype(str)
    headers = [h.lower() for h in common_headers]
    gold = as_span_table(annotations)
    combined: dict[tuple[str, str], Counter] = {}

    for note_id, group in gold.groupby("note_id"):
        if note_id not in texts.index:
            continue
        text = texts.loc[note_id].lower()
        h_positions, pos_header = get_sections(text, headers)
        for row in group.itertuples(index=False):
            mention = _mention_text(texts.loc[note_id], int(row.start), int(row.end))
            if len(mention) < 2 or mention in internal_blacklist:
                continue
            header = get_header_by_pos(
                int(row.start), h_positions, pos_header, headers
            )
            if header is None:
                header = SECTION_ANY
            cid = int(row.concept_id)
            combined.setdefault((header, mention), Counter())[cid] += 1
            combined.setdefault((SECTION_ANY, mention), Counter())[cid] += 1

    mapping: dict[tuple[str, str], int] = {}
    for key, counts in combined.items():
        total = sum(counts.values())
        concept_id, freq = counts.most_common(1)[0]
        if total < min_count:
            continue
        if key[0] == SECTION_ANY and freq / total < 0.5:
            continue
        mapping[key] = int(concept_id)
    return mapping


def save_kiri_core(path: Path, mapping: dict[tuple[str, str], int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(mapping, handle)


def load_kiri_core(path: Path) -> dict[tuple[str, str], int]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_kiri(
    notes: pd.DataFrame,
    mapping: dict[tuple[str, str], int],
    skip_prefix: int = 100,
) -> pd.DataFrame:
    """Recherche par n-grammes clé section, puis retrait des chevauchements (plus long / section)."""
    if not mapping:
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id"])

    headers = [h.lower() for h in common_headers]
    max_n = max(len(mention.split()) for _, mention in mapping)
    rows: list[dict] = []

    for note in notes.itertuples(index=False):
        raw = str(note.text)
        lowered = raw.lower()
        h_positions, pos_header = get_sections(lowered, headers)
        tokens = _token_spans(raw)
        hits: list[dict] = []
        for i, _tok in enumerate(tokens):
            parts: list[str] = []
            for n in range(1, max_n + 1):
                if i + n > len(tokens):
                    break
                parts.append(tokens[i + n - 1][0])
                mention = " ".join(parts)
                start = tokens[i][1]
                end = tokens[i + n - 1][2]
                if start < skip_prefix:
                    continue
                header = get_header_by_pos(start, h_positions, pos_header, headers)
                if header is None:
                    continue
                header_l = header.lower()
                if "medication" in header_l or "service" in header_l or "date of birth" in header_l:
                    continue
                concept_id = mapping.get((header, mention))
                section = header
                if concept_id is None:
                    concept_id = mapping.get((SECTION_ANY, mention))
                    section = SECTION_ANY
                if concept_id is None:
                    continue
                hits.append(
                    {
                        "note_id": note.note_id,
                        "start": start,
                        "end": end,
                        "concept_id": int(concept_id),
                        "section": section,
                    }
                )
        if not hits:
            continue
        framed = pd.DataFrame(hits)
        cleaned = remove_overlaps(framed)
        rows.extend(cleaned.to_dict(orient="records"))

    if not rows:
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id"])
    return submission_view(pd.DataFrame(rows))
