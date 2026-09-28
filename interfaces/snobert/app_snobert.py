"""Interface Streamlit : SnoBERT contre annotations."""
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Mise en page large pour une visualisation côte à côte
st.set_page_config(page_title="SnoBERT: Annotations vs Predictions", layout="wide")

sys.path.append(str(Path(__file__).resolve().parents[1]))

from demo_ui import (
    TABLE_CLICK_CAPTION,
    apply_light_theme,
    paint_located_note,
    render_clickable_table,
    render_note_panel,
    reset_clickable,
    want_detail_tables,
)

apply_light_theme()

@st.cache_data
def load_snobert_data():
    """
    Charge les notes de test, les annotations de référence et les prédictions SnoBERT.
    Mis en cache pour ne pas relire les CSV à chaque mouvement du curseur.
    """
    base_dir = Path(__file__).parent
    
    notes_df = pd.read_csv(base_dir / "data" / "test_notes.csv")
    truth_df = pd.read_csv(base_dir / "data" / "test_annotations.csv")
    pred_df = pd.read_csv(base_dir / "data" / "submission.csv")
    
    # start et end en entiers pour le découpage et le matching
    for df in [truth_df, pred_df]:
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        
    return notes_df, truth_df, pred_df

def get_colored_html(text, category):
    """
    Renvoie le texte en HTML avec une couleur de fond selon la catégorie d'accord.
    """
    styles = {
        "green":  {"bg": "#28a745", "label": "Match"},
        "yellow": {"bg": "#ffc107", "label": "New SnoBERT Prediction"},
        "red":    {"bg": "#dc3545", "label": "Missed Annotation"}
    }
    
    style = styles.get(category)
    if not style:
        return text
        
    text_color = "white" if category in ["green", "red"] else "#1e293b"
    return f'<span title="{style["label"]}" style="background-color: {style["bg"]}; color: {text_color}; padding: 2px 4px; border-radius: 4px; cursor: help; font-weight: bold;">{text}</span>'

def check_overlap(s1, e1, s2, e2):
    """
    Vrai si deux spans se chevauchent d'au moins un caractère.
    """
    return max(0, min(e1, e2) - max(s1, s2)) > 0


########################################################


@st.cache_data
def calculate_global_metrics(conf_thresh, _all_preds_df, _truth_df, _available_notes):
    """
    Calcule précision, rappel et F1 micro sur toutes les notes,
    pour le seuil de confiance courant.
    """
    global_tp, global_fp, global_fn = 0, 0, 0
    filtered_all = _all_preds_df[_all_preds_df["confidence_score"] >= conf_thresh]
    
    for nid in _available_notes:
        n_truth = _truth_df[_truth_df["note_id"] == nid]
        n_preds = filtered_all[filtered_all["note_id"] == nid]
        
        t_spans = [(int(row.start), int(row.end)) for row in n_truth.itertuples()]
        p_spans = [(int(row.start), int(row.end)) for row in n_preds.itertuples()]
        
        h_true = set()
        for p_start, p_end in p_spans:
            is_match = False
            for t_start, t_end in t_spans:
                if check_overlap(p_start, p_end, t_start, t_end):
                    is_match = True
                    h_true.add((t_start, t_end))
            if is_match:
                global_tp += 1
            else:
                global_fp += 1
                
        for t_start, t_end in t_spans:
            if (t_start, t_end) not in h_true:
                global_fn += 1
                
    g_prec = (global_tp / (global_tp + global_fp)) if (global_tp + global_fp) > 0 else 0.0
    g_rec = (global_tp / (global_tp + global_fn)) if (global_tp + global_fn) > 0 else 0.0
    g_f1 = 2 * (g_prec * g_rec) / (g_prec + g_rec) if (g_prec + g_rec) > 0 else 0.0
    
    return g_prec, g_rec, g_f1

########################################################


notes_df, truth_df, all_preds_df = load_snobert_data()

st.title(" SnoBERT : Annotations vs Prédictions ")
st.write("Visualisation comparative de l'extraction et de l'alignement SNOMED CT")

# --- CONTRÔLES DE LA BARRE LATÉRALE ---
st.sidebar.header("Configuration & Filtres")

confidence_threshold = st.sidebar.slider(
    "Seuil de confiance SapBERT (Confidence Score)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.0, 
    step=0.05
)

available_notes = notes_df["note_id"].unique()
selected_note_id = st.sidebar.selectbox(
    "Sélectionner une note clinique (NOTE_ID) :", 
    options=available_notes
)

raw_text = notes_df[notes_df["note_id"] == selected_note_id]["text"].values[0]

note_truth_df = truth_df[truth_df["note_id"] == selected_note_id]
note_preds_df = all_preds_df[all_preds_df["note_id"] == selected_note_id]

filtered_preds_df = note_preds_df[note_preds_df["confidence_score"] >= confidence_threshold].copy()

# --- LOGIQUE DE CHEVAUCHEMENT ---
true_spans = [(int(row.start), int(row.end)) for row in note_truth_df.itertuples()]
pred_spans = [(int(row.start), int(row.end)) for row in filtered_preds_df.itertuples()] if not filtered_preds_df.empty else []

green_spans = set()
yellow_spans = set()
red_spans = set()
hit_true_spans = set()

# 1. Évaluer les prédictions
for p_start, p_end in pred_spans:
    is_match = False
    for t_start, t_end in true_spans:
        if check_overlap(p_start, p_end, t_start, t_end):
            is_match = True
            hit_true_spans.add((t_start, t_end))
            
    if is_match:
        green_spans.add((p_start, p_end))
    else:
        yellow_spans.add((p_start, p_end))

# 2. Identifier les références manquées
for t_start, t_end in true_spans:
    if (t_start, t_end) not in hit_true_spans:
        red_spans.add((t_start, t_end))

########################################################


# # --- UI METRICS ---
# st.subheader("Métriques de performance")

# # Calculer vrais positifs (TP), faux positifs (FP) et faux négatifs (FN)
# tp = len(green_spans)
# fp = len(yellow_spans)
# fn = len(red_spans)

# # Calculer rappel, précision et F1
# local_recall = (tp / (tp + fn) * 100) if (tp + fn) > 0 else 0.0
# local_precision = (tp / (tp + fp) * 100) if (tp + fp) > 0 else 0.0
# local_f1 = (2 * local_precision * local_recall) / (local_precision + local_recall) if (local_precision + local_recall) > 0 else 0.0

# # Cinq colonnes pour la mise en page
# col1, col2, col3, col4, col5 = st.columns(5)

# col1.metric("Matches (Vert)", tp)
# col2.metric("Ajouts Dictionnaire (Jaune)", fp)
# col3.metric("Non prédits (Rouge)", fn)
# col4.metric("precision (precision)", f"{local_precision:.1f} %")
# col4.metric("Rappel (Recall)", f"{local_recall:.1f} %")
# col5.metric("F1-Score", f"{local_f1:.1f} %")

########################################################



# --- MÉTRIQUES INTERFACE ---
st.subheader("Métriques de performance")

# Métriques globales
global_precision, global_recall, global_f1 = calculate_global_metrics(
    confidence_threshold, all_preds_df, truth_df, available_notes
)

# Métriques locales
local_tp = len(green_spans)
local_fp = len(yellow_spans)
local_fn = len(red_spans)

local_precision = (local_tp / (local_tp + local_fp)) if (local_tp + local_fp) > 0 else 0.0
local_recall_val = (local_tp / (local_tp + local_fn)) if (local_tp + local_fn) > 0 else 0.0
local_f1 = 2 * (local_precision * local_recall_val) / (local_precision + local_recall_val) if (local_precision + local_recall_val) > 0 else 0.0

# Afficher les métriques de la note
st.markdown("##### Statistiques Locales (Note sélectionnée)")
col1, col2, col3 = st.columns(3)
col1.metric("Matches (Vert)", local_tp)
col2.metric("Détections (Jaune)", local_fp)
col3.metric("Non prédits (Rouge)", local_fn)
scol1, scol2 , scol3 = st.columns(3)
scol1.metric("Précision", f"{local_precision * 100:.1f} %")
scol2.metric("Rappel", f"{local_recall_val * 100:.1f} %")
scol3.metric("F1-Score", f"{local_f1 * 100:.1f} %")

# Display Global Metrics
st.markdown("##### Statistiques Globales (Moyenne sur toutes les notes)")
g_col1, g_col2, g_col3 = st.columns(3)
g_col1.metric("Précision Globale", f"{global_precision * 100:.1f} %")
g_col2.metric("Rappel Global", f"{global_recall * 100:.1f} %")
g_col3.metric("F1-Score Global", f"{global_f1 * 100:.1f} %")




color_spans = (
    [(s, e, "green") for s, e in green_spans]
    + [(s, e, "yellow") for s, e in yellow_spans]
    + [(s, e, "red") for s, e in red_spans]
)
locate_spans = [(s, e, "yellow") for s, e in yellow_spans] + [(s, e, "red") for s, e in red_spans]
html_text = paint_located_note(raw_text, color_spans, locate_spans)

# --- TEXTE ET TABLEAUX DE DÉTAIL ---
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

        tab1, tab2 = st.tabs(["🟨 Ajouts (Jaune)", "🟥 Manques (Rouge)"])

        with tab1:
            st.write("**Entités détectées absentes des annotations.**")
            if not yellow_spans:
                st.info("Aucun ajout.")
            else:
                y_mask = filtered_preds_df.apply(lambda r: (int(r.start), int(r.end)) in yellow_spans, axis=1)
                yellow_df = filtered_preds_df[y_mask].copy()

                display_cols = ["start", "end", "extracted_span", "snomed_term", "medical_type", "confidence_score", "concept_id"]
                yellow_df["confidence_score"] = yellow_df["confidence_score"].apply(lambda x: f"{x:.2f}")
                display_yellow_df = reset_clickable(
                    yellow_df.sort_values(by=["start", "end"])[display_cols]
                )
                render_clickable_table(
                    display_yellow_df,
                    mention_col="extracted_span",
                    category="yellow",
                    columns=["extracted_span", "snomed_term", "medical_type", "confidence_score", "concept_id"],
                )

        with tab2:
            st.write("**Annotations absentes des prédictions.**")
            if not red_spans:
                st.info("Aucun manque.")
            else:
                r_data = []
                for s, e in red_spans:
                    t_row = note_truth_df[(note_truth_df.start == s) & (note_truth_df.end == e)].iloc[0]

                    r_data.append({
                        "start": s,
                        "end": e,
                        "span_attendu": raw_text[s:e],
                        "concept_id_attendu": t_row.concept_id if "concept_id" in t_row else "N/A",
                    })

                red_df = reset_clickable(pd.DataFrame(r_data).sort_values(by=["start", "end"]))
                render_clickable_table(
                    red_df,
                    mention_col="span_attendu",
                    category="red",
                    columns=["span_attendu", "concept_id_attendu"],
                )