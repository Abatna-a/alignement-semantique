"""Démo PARHAF infectio : référence contre prédictions NER figées (CSV, sans recharger le modèle)."""

from html import escape
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

st.set_page_config(page_title="PARHAF NER: gold vs prédictions", layout="wide")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from demo_ui import (
    TABLE_CLICK_CAPTION,
    apply_light_theme,
    paint_char_located_note,
    render_clickable_table,
    render_note_panel,
    reset_clickable,
    want_detail_tables,
)

apply_light_theme()

BASE = Path(__file__).parent
SNOBERT_DATA = Path(
    os.environ.get(
        "PARHAF_DATA",
        Path(__file__).resolve().parents[2] / "snobert" / "data" / "parhaf_infectiology",
    )
)


@st.cache_data
def load_parhaf_data(split: str):
    notes = pd.read_csv(SNOBERT_DATA / f"notes_{split}.csv")
    truth = pd.read_csv(SNOBERT_DATA / f"annotations_{split}.csv")
    pred_path = BASE / "data" / f"submission_{split}.csv"
    pred = pd.read_csv(pred_path)
    for df in (truth, pred):
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
    if "label" not in pred.columns and "concept_id" in pred.columns:
        pred["label"] = pred["concept_id"].astype(str)
    truth = truth.rename(columns={"span": "extracted_span", "concept_id": "label"})
    return notes, truth, pred


def colored_html(text: str, category: str) -> str:
    styles = {
        "green": {"bg": "#28a745", "label": "Gold capturé / pred match"},
        "yellow": {"bg": "#ffc107", "label": "Prédiction absente du gold"},
        "red": {"bg": "#dc3545", "label": "Gold manqué"},
    }
    style = styles[category]
    color = "white" if category in ("green", "red") else "#1e293b"
    return (
        f'<span title="{style["label"]}" style="background-color: {style["bg"]}; '
        f'color: {color}; padding: 2px 4px; border-radius: 4px; '
        f'cursor: help; font-weight: bold;">{text}</span>'
    )


def paint_text(text: str, spans: list[tuple[int, int, str]], focus=None) -> str:
    """Colorie chaque caractère or/préd. Priorité : rouge > jaune > vert.
    
        focus est un (start, end, catégorie) optionnel choisi dans un tableau :
        il gagne sur ses caractères, reçoit un cadre rouge et devient l'ancre de défilement.
        """
    n = len(text)
    priority = {"red": 3, "yellow": 2, "green": 1}
    color: list[str | None] = [None] * n
    prio: list[int] = [0] * n
    for start, end, cat in spans:
        rank = priority[cat]
        lo = max(0, start)
        hi = min(n, end)
        for i in range(lo, hi):
            if rank > prio[i]:
                color[i] = cat
                prio[i] = rank

    f_start, f_end = (focus[0], focus[1]) if focus else (-1, -1)
    if focus:
        for i in range(max(0, f_start), min(n, f_end)):
            color[i] = focus[2]

    parts: list[str] = []
    i = 0
    while i < n:
        cat = color[i]
        j = i + 1
        # Un run ne chevauche jamais les bornes du focus : le cadre l'enveloppe exactement.
        while j < n and color[j] == cat and j != f_start and j != f_end:
            j += 1
        chunk = escape(text[i:j])
        chunk = chunk if cat is None else colored_html(chunk, cat)
        if i == f_start and j == f_end:
            chunk = focus_anchor(chunk, focus[2] if focus else "red")
        parts.append(chunk)
        i = j
    return "".join(parts)


def overlaps(s1: int, e1: int, s2: int, e2: int) -> bool:
    return max(0, min(e1, e2) - max(s1, s2)) > 0


@st.cache_data
def global_metrics(conf_thresh: float, _preds: pd.DataFrame, _truth: pd.DataFrame, note_ids):
    filtered = _preds[_preds["confidence_score"] >= conf_thresh]
    tp = fp = fn = 0
    for nid in note_ids:
        gold = _truth[_truth["note_id"] == nid]
        pred = filtered[filtered["note_id"] == nid]
        t_spans = [(int(r.start), int(r.end), str(r.label)) for r in gold.itertuples()]
        p_spans = [(int(r.start), int(r.end), str(r.label)) for r in pred.itertuples()]
        hit = set()
        for ps, pe, pl in p_spans:
            matched = False
            for ts, te, tl in t_spans:
                if overlaps(ps, pe, ts, te) and pl == tl:
                    matched = True
                    hit.add((ts, te, tl))
            if matched:
                tp += 1
            else:
                fp += 1
        for item in t_spans:
            if item not in hit:
                fn += 1
    prec = tp / (tp + fp) if (tp + fp) else 0.0
    rec = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
    return prec, rec, f1


st.sidebar.header("Données")
split = st.sidebar.radio(
    "Split",
    options=["test", "val"],
    index=0,
    help="test = 14 CR jamais vus au save ; val = 20 CR du mIoU",
)

notes_df, truth_df, all_preds_df = load_parhaf_data(split)

st.title("PARHAF infectio — gold vs NER")
st.write(
    "CamemBERT-bio, checkpoint `S1_18_score_0.7940`. "
    "L’interface lit `submission_val.csv` / `submission_test.csv` "
    "(pas de rechargement du modèle). Vert = overlap + même type, "
    "jaune = pred hors gold, rouge = gold manqué."
)

st.sidebar.header("Filtres")
confidence_threshold = st.sidebar.slider(
    "Seuil de confiance",
    min_value=0.0,
    max_value=1.0,
    value=0.0,
    step=0.05,
)

available_notes = notes_df["note_id"].unique()
selected_note_id = st.sidebar.selectbox("CR (note_id)", options=available_notes)

raw_text = notes_df[notes_df["note_id"] == selected_note_id]["text"].values[0]
note_truth_df = truth_df[truth_df["note_id"] == selected_note_id]
note_preds_df = all_preds_df[all_preds_df["note_id"] == selected_note_id]
filtered_preds_df = note_preds_df[
    note_preds_df["confidence_score"] >= confidence_threshold
].copy()

t_spans = [(int(r.start), int(r.end), str(r.label)) for r in note_truth_df.itertuples()]
p_spans = [
    (int(r.start), int(r.end), str(r.label)) for r in filtered_preds_df.itertuples()
]

green_spans = set()
yellow_spans = set()
red_spans = set()
hit_true = set()

for ps, pe, pl in p_spans:
    matched = False
    for ts, te, tl in t_spans:
        if overlaps(ps, pe, ts, te) and pl == tl:
            matched = True
            hit_true.add((ts, te, tl))
    if matched:
        green_spans.add((ps, pe, pl))
    else:
        yellow_spans.add((ps, pe, pl))

for item in t_spans:
    if item not in hit_true:
        red_spans.add(item)

st.subheader("Métriques")
g_prec, g_rec, g_f1 = global_metrics(
    confidence_threshold, all_preds_df, truth_df, available_notes
)
local_tp, local_fp, local_fn = len(green_spans), len(yellow_spans), len(red_spans)
local_prec = local_tp / (local_tp + local_fp) if (local_tp + local_fp) else 0.0
local_rec = local_tp / (local_tp + local_fn) if (local_tp + local_fn) else 0.0
local_f1 = (
    2 * local_prec * local_rec / (local_prec + local_rec)
    if (local_prec + local_rec)
    else 0.0
)

st.markdown(f"##### Local — `{selected_note_id}`")
c1, c2, c3 = st.columns(3)
c1.metric("Matches (vert)", local_tp)
c2.metric("Pred hors gold (jaune)", local_fp)
c3.metric("Gold manqués (rouge)", local_fn)
s1, s2, s3 = st.columns(3)
s1.metric("Précision", f"{local_prec * 100:.1f} %")
s2.metric("Rappel", f"{local_rec * 100:.1f} %")
s3.metric("F1", f"{local_f1 * 100:.1f} %")

st.markdown(f"##### Global — split **{split}** ({len(available_notes)} CR)")
g1, g2, g3 = st.columns(3)
g1.metric("Précision", f"{g_prec * 100:.1f} %")
g2.metric("Rappel", f"{g_rec * 100:.1f} %")
g3.metric("F1", f"{g_f1 * 100:.1f} %")

# Chaque or est vert (capturé) ou rouge (manqué). Chaque pred est verte (accord) ou jaune (ajout).
# Chevauchements : peindre au caractère, rouge > jaune > vert, aucun span perdu.
assert len(green_spans) + len(yellow_spans) == len(p_spans)
assert len(hit_true) + len(red_spans) == len(t_spans)

paint_spans = []
for s, e, _ in green_spans:
    paint_spans.append((s, e, "green"))
for s, e, _ in hit_true:
    paint_spans.append((s, e, "green"))
for s, e, _ in yellow_spans:
    paint_spans.append((s, e, "yellow"))
for s, e, _ in red_spans:
    paint_spans.append((s, e, "red"))

locate_spans = [(s, e, "yellow") for s, e, _ in yellow_spans] + [
    (s, e, "red") for s, e, _ in red_spans
]
html_text = paint_char_located_note(raw_text, paint_spans, locate_spans)

show_tables = want_detail_tables()
if show_tables:
    col_text, col_tables = st.columns([5, 5])
    with col_text:
        render_note_panel(html_text, wide=False)
    table_host = col_tables
else:
    render_note_panel(html_text, wide=True)
    table_host = None

if table_host is not None:
    with table_host:
        st.subheader("Détail des prédictions")
        st.caption(TABLE_CLICK_CAPTION)
        tab1, tab2 = st.tabs(["Jaune — pred hors gold", "Rouge — gold manqué"])
        with tab1:
            if not yellow_spans:
                st.info("Aucun ajout.")
            else:
                y_mask = filtered_preds_df.apply(
                    lambda r: (int(r.start), int(r.end), str(r.label)) in yellow_spans,
                    axis=1,
                )
                yellow_df = filtered_preds_df[y_mask].copy()
                yellow_df["confidence_score"] = yellow_df["confidence_score"].map(
                    lambda x: f"{x:.2f}"
                )
                display_yellow_df = reset_clickable(
                    yellow_df.sort_values(["start", "end"])[
                        ["start", "end", "extracted_span", "label", "confidence_score"]
                    ]
                )
                render_clickable_table(
                    display_yellow_df,
                    mention_col="extracted_span",
                    category="yellow",
                    columns=["extracted_span", "label", "confidence_score"],
                )
        with tab2:
            if not red_spans:
                st.info("Aucun manque.")
            else:
                rows = []
                for s, e, lab in red_spans:
                    rows.append(
                        {
                            "start": s,
                            "end": e,
                            "span_attendu": raw_text[s:e],
                            "label_attendu": lab,
                        }
                    )
                display_red_df = reset_clickable(
                    pd.DataFrame(rows).sort_values(["start", "end"])
                )
                render_clickable_table(
                    display_red_df,
                    mention_col="span_attendu",
                    category="red",
                    columns=["span_attendu", "label_attendu"],
                )
