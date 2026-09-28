from __future__ import annotations

import pickle
import re
from collections import Counter
from pathlib import Path

import pandas as pd

from .schema import as_span_table, submission_view

IGNORE_HEADERS = (
    "medications on admission:",
    "___ on admission:",
    "discharge medications:",
)


def normalize_term(text: str) -> str:
    t = text.lower().replace("\n", " ")
    t = re.sub("[^a-z0-9]", " ", t)
    t = re.sub(r"\s+", " ", t).strip()
    return t


def normalize_keep_len(text: str) -> str:
    t = text.lower().replace("\n", " ")
    t = re.sub("[^a-z0-9]", " ", t)
    return t


def _header_spans(text: str) -> dict[str, tuple[int, int]]:
    lowered = text.lower().replace("\n", " ")
    found: dict[str, int] = {}
    for header in IGNORE_HEADERS:
        pos = lowered.find(header)
        if pos != -1:
            found[header] = pos
    ordered = sorted(found.items(), key=lambda kv: kv[1])
    spans: dict[str, tuple[int, int]] = {}
    for i, (header, start) in enumerate(ordered):
        end = len(text) if i == len(ordered) - 1 else ordered[i + 1][1]
        spans[header] = (start, end)
    return spans


def _in_ignored_header(start: int, end: int, headers: dict[str, tuple[int, int]]) -> bool:
    for header in IGNORE_HEADERS:
        span = headers.get(header)
        if span is None:
            continue
        if max(start, span[0]) <= min(end, span[1]):
            return True
    return False


def _token_spans(norm_text: str) -> list[tuple[str, int, int]]:
    tokens: list[tuple[str, int, int]] = []
    for match in re.finditer(r"[a-z0-9]+", norm_text):
        tokens.append((match.group(), match.start(), match.end()))
    return tokens


def build_most_common_concept(notes: pd.DataFrame, annotations: pd.DataFrame) -> dict[str, int]:
    notes_ix = notes.set_index("note_id")["text"]
    counts: dict[str, Counter] = {}
    for row in as_span_table(annotations).itertuples(index=False):
        text = notes_ix.loc[row.note_id]
        term = normalize_term(text[int(row.start) : int(row.end)])
        if not term:
            continue
        counts.setdefault(term, Counter())[int(row.concept_id)] += 1
    return {term: counter.most_common(1)[0][0] for term, counter in counts.items()}


def save_concept_dict(path: Path, mapping: dict[str, int]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("wb") as handle:
        pickle.dump(mapping, handle)


def load_concept_dict(path: Path) -> dict[str, int]:
    with path.open("rb") as handle:
        return pickle.load(handle)


def predict_with_dict(
    notes: pd.DataFrame,
    most_common_concept: dict[str, int],
    prefer_longer: bool = True,
) -> pd.DataFrame:
    """Recherche par n-grammes (même famille que le dictionnaire statique SnoBERT, sans grosse regex)."""
    if not most_common_concept:
        return pd.DataFrame(columns=["note_id", "start", "end", "concept_id", "term"])

    max_n = max(len(term.split()) for term in most_common_concept)
    rows: list[tuple] = []
    for note in notes.itertuples(index=False):
        raw = note.text
        norm = normalize_keep_len(raw)
        headers = _header_spans(raw)
        tokens = _token_spans(norm)
        hits: list[tuple[int, int, int, str]] = []
        for i, _token in enumerate(tokens):
            phrase_parts: list[str] = []
            for n in range(1, max_n + 1):
                if i + n > len(tokens):
                    break
                phrase_parts.append(tokens[i + n - 1][0])
                phrase = " ".join(phrase_parts)
                concept_id = most_common_concept.get(phrase)
                if concept_id is None:
                    continue
                start = tokens[i][1]
                end = tokens[i + n - 1][2]
                if _in_ignored_header(start=start, end=end, headers=headers):
                    continue
                hits.append((start, end, int(concept_id), phrase))
        if prefer_longer:
            hits = _drop_overlaps_prefer_longer(hits)
        for start, end, concept_id, phrase in hits:
            rows.append((note.note_id, start, end, concept_id, phrase))

    pred = pd.DataFrame(rows, columns=["note_id", "start", "end", "concept_id", "term"])
    if pred.empty:
        return pred
    return submission_view(pred)


def _drop_overlaps_prefer_longer(
    hits: list[tuple[int, int, int, str]],
) -> list[tuple[int, int, int, str]]:
    ordered = sorted(hits, key=lambda h: (h[1] - h[0], h[1]), reverse=True)
    kept: list[tuple[int, int, int, str]] = []
    for hit in ordered:
        if any(max(hit[0], k[0]) < min(hit[1], k[1]) for k in kept):
            continue
        kept.append(hit)
    return sorted(kept, key=lambda h: h[0])
