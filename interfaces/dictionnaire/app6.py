"""Interface Streamlit : dictionnaire contre annotations."""
import streamlit as st
import pandas as pd
import pickle
import re
import sys
from pathlib import Path

# Le moteur dictionnaire et l'interface commune sont un niveau au-dessus.
REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO / "dictionnaire" / "src"))
sys.path.append(str(REPO / "interfaces"))

from mimic_common import common_headers, get_header_by_pos, get_sections
from mimic_predict import make_predictions


# Mise en page large pour une visualisation côte à côte
st.set_page_config(page_title="Evaluateur & Diagnostic NLP", layout="wide")

from demo_ui import (  # noqa: E402
    TABLE_CLICK_CAPTION,
    apply_light_theme,
    paint_located_note,
    render_clickable_table,
    render_note_panel,
    reset_clickable,
    want_detail_tables,
)

apply_light_theme()



# --- MÉTRIQUES GLOBALES ---
@st.cache_data
def calculate_global_metrics(_all_raw_df, _train_annotations, conf_thresh):
    """Calcule précision, rappel et F1 au niveau du jeu entier."""
    tp, fp, fn = 0, 0, 0
    filtered_df = _all_raw_df[_all_raw_df["score"] >= conf_thresh]
    
    for nid in _train_annotations["note_id"].unique():
        t_spans = set((int(row.start), int(row.end)) for row in _train_annotations[_train_annotations["note_id"] == nid].itertuples())
        p_spans = set((int(row.start), int(row.end)) for row in filtered_df[filtered_df["note_id"] == nid].itertuples())
        
        tp += len(t_spans.intersection(p_spans))
        fp += len(p_spans - t_spans)
        fn += len(t_spans - p_spans)
        
    prec = (tp / (tp + fp)) if (tp + fp) > 0 else 0.0
    rec = (tp / (tp + fn)) if (tp + fn) > 0 else 0.0
    f1 = 2 * (prec * rec) / (prec + rec) if (prec + rec) > 0 else 0.0
    return prec, rec, f1


# --- PRÉDICTIONS GLOBALES ET SCORES (EN CACHE) ---
@st.cache_data
def get_all_scored_predictions(_mimic_train, _phrase_dict, _uc_dict, _scores_by_mention, _common_headers):
    """
    Generates predictions for ALL notes and applies the exact same 
    adjusted scoring logic (with TAU) to maintain consistency.
    """
    # 1. Charger toutes les notes
    available_nids = _mimic_train["note_id"].unique()
    all_notes = _mimic_train.set_index("note_id")["text"]
    
    # 2. Prédictions brutes sur tout le jeu
    all_preds = make_predictions(all_notes, _phrase_dict, _uc_dict, submission=True)
    
    if all_preds.empty:
        return all_preds
        
    # 3. Précalculer les positions des en-têtes pour gagner du temps
    lowercase_headers = [h.lower() for h in _common_headers]
    note_sections = {}
    
    for nid, text in all_notes.items():
        h_pos, pos_h = get_sections(str(text).lower(), lowercase_headers)
        note_sections[nid] = (h_pos, pos_h)
        
    # 4. Logique de score avec la pénalité TAU
    TAU = 0.3
    
    def compute_row_score(r):
        # Extraire la section pour cette prédiction
        h_pos, pos_h = note_sections.get(r.note_id, ([], {}))
        section = get_header_by_pos(int(r.start), h_pos, pos_h, lowercase_headers) or "other"
        
        # Normaliser les chaînes pour le dictionnaire
        sec_clean = str(section).lower().strip()
        dict_clean = str(r.dict_entry).lower().strip()
        
        # Chercher dans l'historique des scores
        vals = _scores_by_mention.get((sec_clean, dict_clean), [])
        if not vals:
            vals = _scores_by_mention.get((section, r.dict_entry), [])
            
        # Appliquer la formule P / (P + tau * M)
        if vals:
            p_matches = sum(1 for v in vals if v == 1 or v == 1.0)
            m_additions = len(vals) - p_matches
            denominator = p_matches + (TAU * m_additions)
            
            if denominator > 0:
                return round(p_matches / denominator, 2)
            else:
                return 0.0
                
        # Score par défaut si jamais vu à l'apprentissage
        return 1.0 
        
    # 5. Appliquer le score à tout le DataFrame
    all_preds["score"] = all_preds.apply(compute_row_score, axis=1)
    
    return all_preds



@st.cache_data
def load_precomputed_global_preds():
    """
    Loads the pre-computed global predictions from the offline script.
    """
    base_dir = Path(__file__).parent
    path = base_dir / "data/dict_global_predictions.csv"
    
    if path.exists():
        df = pd.read_csv(path)
        # Entiers pour le calcul des chevauchements
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        return df
    else:
        st.warning("Fichier dict_global_predictions.csv introuvable. Veuillez exécuter le script offline d'abord.")
        return pd.DataFrame()



def get_levenshtein_similarity(s1, s2):
    """
    Computes a normalized similarity ratio between 0.0 and 1.0 based on Levenshtein Distance.
    """
    s1, s2 = str(s1).lower().strip(), str(s2).lower().strip()
    if s1 == s2:
        return 1.0
    if not s1 or not s2:
        return 0.0
        
    if len(s1) < len(s2):
        s1, s2 = s2, s1

    distances = range(len(s2) + 1)
    for i, c1 in enumerate(s1):
        distances_ = [i + 1]
        for j, c2 in enumerate(s2):
            if c1 == c2:
                distances_.append(distances[j])
            else:
                distances_.append(1 + min((distances[j], distances[j + 1], distances_[-1])))
        distances = distances_

    lev_dist = distances[-1]
    max_len = max(len(s1), len(s2))
    return round(1.0 - (lev_dist / max_len), 2)



# En-têtes standards pour le masquage des coordonnées
common_headers = [
    "Allergies", "History of Present Illness", "Family History", "Name",
    "Major Surgical or Invasive Procedure", "Admission Date", "Discharge Disposition",
    "Past Medical History", "Attending", "Service", "Date of Birth",
    "Discharge Instructions", "Discharge Condition", "Chief Complaint",
    "Physical Exam", "Pertinent Results", "Discharge Medications",
    "Social History", "Followup Instructions", "Medications on Admission",
    "Discharge Diagnosis"
]

@st.cache_resource
def load_pipeline_data():
    """Charge dictionnaires, métadonnées, scores de debug et jeux."""
    base_dir = Path(__file__).parent
    
    # Charger la terminologie et les modèles
    with open(base_dir / "assets/kiri_dicts.pkl", "rb") as f:
        kiri_dicts = pickle.load(f)
        
    scores = {}
    scores_path = base_dir / "debug/debug.pkl"
    if scores_path.exists():
        with open(scores_path, "rb") as f:
            scores = pickle.load(f).get("scores_by_mention", {})
            
    # Charger le mapping des concepts pour résoudre les types
    term_path = base_dir / "data/flattened_terminology.csv"
    if not term_path.exists():
        term_path = base_dir / "data/interim/flattened_terminology.csv"
        
    c_map = {}
    if term_path.exists():
        term = pd.read_csv(term_path)
        c_map = dict(zip(term["concept_id"], term["concept_name"]))

    # Charger les jeux de données
    mimic_train = pd.read_csv(base_dir / "data/train_notes.csv")
    train_annotations = pd.read_csv(base_dir / "data/train_annotations_cln.csv")
    
    return kiri_dicts, mimic_train, train_annotations, scores, c_map

# Charger toutes les ressources
kiri_dicts, mimic_train, train_annotations, scores_by_mention, concept_map = load_pipeline_data()

st.title(" Annotations vs prédictions ")
st.write("Visualisation comparative")

# --- CONTRÔLES DE LA BARRE LATÉRALE ---
st.sidebar.header("Configuration & Filtres")

confidence_threshold = st.sidebar.slider(
    "Seuil de confiance (Confidence Threshold)", 
    min_value=0.0, 
    max_value=1.0, 
    value=0.0, 
    step=0.05
)

# Ne garder que les notes qui ont des annotations
annotated_note_ids = train_annotations["note_id"].unique()
available_notes = mimic_train[mimic_train["note_id"].isin(annotated_note_ids)]

selected_note_id = st.sidebar.selectbox(
    "Sélectionner une note clinique (NOTE_ID) :", 
    options=available_notes["note_id"].tolist()
)

# Texte brut
raw_text = available_notes[available_notes["note_id"] == selected_note_id]["text"].values[0]

# --- CHAÎNE DE PRÉDICTION ---
phrase_dict, uc_dict = kiri_dicts
texts_series = pd.Series({selected_note_id: raw_text})

# Forcer les prédictions brutes
raw_pred_df = make_predictions(texts_series, phrase_dict, uc_dict, submission=True)

# Enrichir avec section, scores et types si non vide
# Enrichir avec section, scores et types si non vide
if not raw_pred_df.empty:
    text_lc = str(raw_text).lower()
    lowercase_headers = [h.lower() for h in common_headers]
    h_positions, pos_header = get_sections(text_lc, lowercase_headers)
    
    raw_pred_df["section"] = raw_pred_df.apply(
        lambda r: get_header_by_pos(int(r.start), h_positions, pos_header, lowercase_headers) or "other", axis=1
    )

    ####


    # Extraire les combinaisons uniques pour gagner du temps
    detected = raw_pred_df[['section', 'dict_entry']].drop_duplicates()
    conf_map = {}

    for _, r in detected.iterrows():
        # Mettre en minuscules pour coller aux clés d'historique
        sec_clean = str(r['section']).lower().strip()
        dict_clean = str(r['dict_entry']).lower().strip()
        
        # Essayer d'abord les clés nettoyées, sinon l'original
        vals = scores_by_mention.get((sec_clean, dict_clean), [])
        if not vals:
            vals = scores_by_mention.get((r['section'], r['dict_entry']), [])
            
        if vals:
            correct_count = sum(1 for v in vals if v == 1 or v == 1.0)
            precision = correct_count / len(vals)
            # Stocker dans la map avec des clés en minuscules
            conf_map[(sec_clean, dict_clean)] = round(precision, 2)

    # Appliquer les scores de précision avec les clés normalisées
    raw_pred_df["score"] = raw_pred_df.apply(
        lambda r: conf_map.get((str(r['section']).lower().strip(), str(r['dict_entry']).lower().strip()), 1.0),
        axis=1
    )

    # Compute if context was successfully found in training history
    raw_pred_df["seen_in_train"] = raw_pred_df.apply(
        lambda r: (str(r['section']).lower().strip(), str(r['dict_entry']).lower().strip()) in conf_map,
        axis=1
    )
    
    raw_pred_df["mention"] = [raw_text[int(r.start):int(r.end)] for r in raw_pred_df.itertuples()]


    # Résoudre les types sémantiques et les termes d'origine
    raw_terms = raw_pred_df["concept_id"].map(concept_map).fillna("Unknown")
    def split_term_and_type(val):
        val_str = str(val).strip()
        if val_str.endswith(")") and " (" in val_str:
            idx = val_str.rfind(" (")
            return val_str[:idx].strip(), val_str[idx + 2 : -1].strip()
        return val_str, "Unknown"

    processed_terms = raw_terms.apply(split_term_and_type)
    # --- ADDED: Extract original_term ---
    raw_pred_df["original_term"] = [t[0] for t in processed_terms]
    raw_pred_df["type"] = [t[1] for t in processed_terms]

    # --- ADDED: Metric score_2 tracking string distance variance ---
    raw_pred_df["score_2"] = raw_pred_df.apply(
        lambda r: get_levenshtein_similarity(r["mention"], r["original_term"]), 
        axis=1
    )

    # Structural masking
    header_spans = []
    for h in common_headers:
        pattern = re.compile(re.escape(h) + r"\s*:?", re.IGNORECASE)
        for match in pattern.finditer(raw_text):
            header_spans.append((match.start(), match.end()))
            
    valid_mask = []
    for r in raw_pred_df.itertuples():
        is_inside_header = any(int(r.start) >= hs and int(r.end) <= he for hs, he in header_spans)
        valid_mask.append(not is_inside_header)
        
    raw_pred_df = raw_pred_df[valid_mask].copy()

# Seuil de confiance utilisateur → DataFrame de prédiction final
if not raw_pred_df.empty:
    pred_df = raw_pred_df[raw_pred_df['score'] >= confidence_threshold].copy()
else:
    pred_df = pd.DataFrame()

# Annotations de référence
true_df = train_annotations[train_annotations["note_id"] == selected_note_id]

# --- MATCHING LOGIC ---
true_spans = set((int(row.start), int(row.end)) for row in true_df.itertuples())
pred_spans = set((int(row.start), int(row.end)) for row in pred_df.itertuples()) if not pred_df.empty else set()

# true_spans = set(true_df['span'].str.lower())
# pred_spans = set(pred_df['span'].str.lower()) if not pred_df.empty else set()

green_spans = true_spans.intersection(pred_spans)          
yellow_spans = pred_spans - true_spans                    
red_spans = true_spans - pred_spans  



# --- HELPER FOR TOOLTIP UI ---
def get_colored_html(text, category):
    styles = {
        "green":  {"bg": "#28a745", "label": "Match"},
        "yellow": {"bg": "#ffc107", "label": "New Prediction"},
        "red":    {"bg": "#dc3545", "label": "Missed Annotation"}
    }
    
    style = styles.get(category)
    if not style:
        return text
        
    return f'<span title="{style["label"]}" style="background-color: {style["bg"]}; color: white; padding: 2px 4px; border-radius: 4px; cursor: help; font-weight: bold;">{text}</span>'

 


# --- MÉTRIQUES INTERFACE ---
st.subheader("Métriques de performance")


local_tp, local_fp, local_fn = len(green_spans), len(yellow_spans), len(red_spans)
local_precision = (local_tp / (local_tp + local_fp)) if (local_tp + local_fp) > 0 else 0.0
local_recall = (local_tp / (local_tp + local_fn)) if (local_tp + local_fn) > 0 else 0.0
local_f1 = (2 * local_precision * local_recall) / (local_precision + local_recall) if (local_precision + local_recall) > 0 else 0.0


st.markdown("##### Statistiques Locales (Note sélectionnée)")

c1, c2, c3 = st.columns(3)
c1.metric("Matches", local_tp); c2.metric("Ajouts", local_fp); c3.metric("Manqués", local_fn)
sc1, sc2, sc3 = st.columns(3)
sc1.metric("Précision", f"{local_precision*100:.1f}%"); sc2.metric("Rappel", f"{local_recall*100:.1f}%"); sc3.metric("F1-Score", f"{local_f1*100:.1f}%")



##############################################
# st.markdown("##### Statistiques Globales")


# # g_prec, g_rec, g_f1 = calculate_global_metrics(raw_pred_df, train_annotations, confidence_threshold)

# # 1. Préparer toutes les notes du corpus d'entraînement
# all_notes = available_notes.set_index("note_id")["text"]

# # 2. Générer les prédictions pour TOUTES les notes (Attention : cela peut prendre quelques secondes)
# all_raw_preds = make_predictions(all_notes, phrase_dict, uc_dict, submission=True)

# # 3. Appliquer le score de confiance globalement (simplifié ici pour correspondre à votre logique)
# # Si all_raw_preds n'a pas de colonne 'score', on lui force un score de 1.0 par défaut pour que le filtre fonctionne
# if "score" not in all_raw_preds.columns:
#     all_raw_preds["score"] = 1.0

# # 4. Calculer les vraies métriques globales
# g_prec, g_rec, g_f1 = calculate_global_metrics(all_raw_preds, train_annotations, confidence_threshold)


# gc1, gc2, gc3 = st.columns(3)
# gc1.metric("Précision Globale", f"{g_prec*100:.1f}%")
# gc2.metric("Rappel Global", f"{g_rec*100:.1f}%")
# gc3.metric("F1-Score Global", f"{g_f1*100:.1f}%")


##############################################


st.markdown("##### Statistiques Globales")

# Charger le DataFrame précalculé (instantané)
all_scored_preds = load_precomputed_global_preds()


if not all_scored_preds.empty:
    # Filtrer et calculer les métriques
    g_prec, g_rec, g_f1 = calculate_global_metrics(all_scored_preds, train_annotations, confidence_threshold)
    
    gc1, gc2, gc3 = st.columns(3)
    gc1.metric("Précision Globale", f"{g_prec*100:.1f}%")
    gc2.metric("Rappel Global", f"{g_rec*100:.1f}%")
    gc3.metric("F1-Score Global", f"{g_f1*100:.1f}%")
else:
    st.info("Les statistiques globales ne sont pas disponibles.")



    
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
else:
    render_note_panel(html_text, wide=True)
    col_tables = None

if col_tables is not None:
    with col_tables:
        st.subheader("Détail des prédictions")
        st.caption(TABLE_CLICK_CAPTION)

        # Colonnes standards à afficher
        standard_cols = [
            "mention", "section", "score", "seen_in_train", "score_2",
            "original_term", "type", "concept_id",
        ]

        tab1, tab2 = st.tabs(["🟨 Ajouts Dictionnaire (Jaune)", "🟥 Non Prédits / Manqués (Rouge)"])

        with tab1:
            st.write("**Entités détectées par le modèle mais absentes des annotations manuelles.**")
            if not yellow_spans:
                st.info("Aucun ajout détecté.")
            else:
                # Ne garder que les spans jaunes
                y_mask = pred_df.apply(lambda r: (int(r.start), int(r.end)) in yellow_spans, axis=1)
                yellow_df = pred_df[y_mask].copy()
            
                # Affichage aligné sur l'ancien layout, plus la raison de diagnostic
                display_cols = standard_cols
            
                # Garder start/end (cachés) pour remonter aux offsets depuis une ligne
                yellow_df = yellow_df.sort_values(by=["start", "end"])
                display_yellow_df = reset_clickable(
                    yellow_df[display_cols + ["start", "end"]].copy()
                )
                if 'score_2' in display_yellow_df.columns:
                    display_yellow_df = display_yellow_df.rename(columns={'score_2': 'score de levenchtein'})
                    display_cols = [c if c != 'score_2' else 'score de levenchtein' for c in display_cols]

                render_clickable_table(
                    display_yellow_df,
                    mention_col="mention",
                    category="yellow",
                    columns=display_cols,
                )

        with tab2:
            st.write("**Entités réelles (Ground Truth) que le pipeline final n'a pas retenues.**")
            if not red_spans:
                st.info("Aucun faux négatif détecté.")
            else:
                r_data = []
            
                # Calculer les métriques manquantes pour les entités de référence
                text_lc = str(raw_text).lower()
                lowercase_headers = [h.lower() for h in common_headers]
                h_positions, pos_header = get_sections(text_lc, lowercase_headers)
            
                for s, e in red_spans:
                    # Données de base depuis la référence
                    t_row = true_df[(true_df.start == s) & (true_df.end == e)].iloc[0]
                
                    # Compute section
                    section = get_header_by_pos(int(s), h_positions, pos_header, lowercase_headers) or "other"
                
                    # Résoudre le terme et le type via concept_map
                    raw_term = concept_map.get(t_row.concept_id, "Unknown")
                    val_str = str(raw_term).strip()
                    if val_str.endswith(")") and " (" in val_str:
                        idx = val_str.rfind(" (")
                        orig_term = val_str[:idx].strip()
                        c_type = val_str[idx + 2 : -1].strip()
                    else:
                        orig_term = val_str
                        c_type = "Unknown"
                    
                    # Compute Levenshtein distance score_2
                    score_2 = get_levenshtein_similarity(t_row.span, orig_term)
                
                    # Vu en prédiction brute mais filtré par le curseur ?
                    was_filtered = False
                    model_score = 0.0
                    seen = False
                
                    if not raw_pred_df.empty:
                        match = raw_pred_df[(raw_pred_df.start == s) & (raw_pred_df.end == e)]
                        if not match.empty:
                            was_filtered = True
                            model_score = match.iloc[0].score
                            seen = match.iloc[0].seen_in_train

                    # La mention exacte dans cette section existe-t-elle dans l'historique ?
                    span_orig = str(t_row.span).strip()
                    section_orig = str(section).strip()
                
                    span_lower = span_orig.lower()
                    section_lower = section_orig.lower()
                
                    seen = False
                    # Consulter directement le dictionnaire d'historique
                    if (
                        (section_lower, span_lower) in scores_by_mention or
                        (section_orig, span_orig) in scores_by_mention or
                        (section_orig, span_lower) in scores_by_mention or
                        (section_lower, span_orig) in scores_by_mention 
                        ) :
                        seen = True

                    else :
                        for dict_key in phrase_dict.keys():
                            dict_section, dict_term = dict_key
                        
                            # Vérifier si le terme correspond
                            if str(dict_term).lower() == span_lower:
                                # Si la section du dictionnaire est un tuple (sections autorisées), vérifier l'inclusion
                                if isinstance(dict_section, tuple):
                                    if section_lower in [str(s).lower() for s in dict_section]:
                                        seen = True
                                        break
                        
                    if was_filtered:
                        reason = f"Filtré (Score {model_score:.2f} < {confidence_threshold})"
                    else:
                        reason = "Absence Dictionnaire"
                    
                    r_data.append({
                        "section": section,
                        "concept_id": t_row.concept_id,
                        "mention": t_row.span,
                        "original_term": orig_term,
                        "type": c_type,
                        "score": model_score if was_filtered else None,
                        "seen_in_train": seen,
                        "score_2": score_2,
                        "cause_echec": reason,
                        "start": s,  # pour le tri et le clic → localisation
                        "end": e
                    })
                
                red_df = pd.DataFrame(r_data)
                display_cols_red = [
                    "mention", "section", "original_term", "type",
                    "seen_in_train", "score_2", "cause_echec", "concept_id",
                ]
                display_red_df = reset_clickable(
                    red_df.sort_values(by=["start", "end"])[display_cols_red + ["start", "end"]].copy()
                )
                if 'score_2' in display_red_df.columns:
                    display_red_df = display_red_df.rename(columns={'score_2': 'score de levenchtein'})
                    display_cols_red = [c if c != 'score_2' else 'score de levenchtein' for c in display_cols_red]

                render_clickable_table(
                    display_red_df,
                    mention_col="mention",
                    category="red",
                    columns=display_cols_red,
                )