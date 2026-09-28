"""Précalcule les prédictions dictionnaire pour l'interface."""
import pandas as pd
import pickle
import sys
import logging
import time
from pathlib import Path
from tqdm import tqdm

# --- JOURNALISATION ---
# Format avec horodatage pour suivre où part le temps
logging.basicConfig(
    level=logging.INFO,
    format="[%(asctime)s] %(levelname)s - %(message)s",
    datefmt="%H:%M:%S"
)

REPO = Path(__file__).resolve().parents[2]
sys.path.append(str(REPO / "dictionnaire" / "src"))

try:
    from mimic_common import common_headers, get_header_by_pos, get_sections
    from mimic_predict import make_predictions
except ImportError as e:
    logging.critical(f"Failed to import the dictionary engine. Error: {e}")
    sys.exit(1)

def main():
    start_time = time.time()
    base_dir = Path(__file__).parent
    
    # ---------------------------------------------------------
    # ÉTAPE 1 : CHARGER LES RESSOURCES
    # ---------------------------------------------------------
    logging.info("Starting offline precomputation pipeline...")
    
    try:
        # Charger les dictionnaires
        dict_path = base_dir / "assets/kiri_dicts.pkl"
        logging.info(f"Loading dictionaries from {dict_path}")
        with open(dict_path, "rb") as f:
            phrase_dict, uc_dict = pickle.load(f)
            
        # Charger les scores d'historique
        scores_path = base_dir / "debug/debug.pkl"
        logging.info(f"Loading historical scores from {scores_path}")
        with open(scores_path, "rb") as f:
            scores_by_mention = pickle.load(f).get("scores_by_mention", {})

        # Charger les notes d'apprentissage
        train_path = base_dir / "data/train_notes.csv"
        logging.info(f"Loading training notes from {train_path}")
        mimic_train = pd.read_csv(train_path)
        all_notes = mimic_train.set_index("note_id")["text"]
        
        logging.info(f"Successfully loaded {len(all_notes)} clinical notes.")
        
    except FileNotFoundError as e:
        logging.critical(f"File not found: {e.filename}. Please check your paths.")
        sys.exit(1)
    except Exception as e:
        logging.critical(f"Unexpected error during data loading: {e}")
        sys.exit(1)

# ---------------------------------------------------------
    # ÉTAPE 2 : PRÉDICTIONS BRUTES (BARRE DE PROGRESSION)
    # ---------------------------------------------------------
    logging.info("Step 2/4: Running raw predictions on all notes...")
    pred_start_time = time.time()
    
    all_preds_list = []
    
    # Une note à la fois pour mettre à jour la barre de progression
    for nid, text in tqdm(all_notes.items(), desc="Running Predictions", total=len(all_notes), unit="note"):
        # Petite Series pour une seule note, format attendu par l'API
        single_note_series = pd.Series({nid: text})
        
        try:
            # Prédire pour cette note
            pred_df = make_predictions(single_note_series, phrase_dict, uc_dict, submission=True)
            
            # Si le modèle a trouvé des entités, les garder
            if not pred_df.empty:
                all_preds_list.append(pred_df)
                
        except Exception as e:
            logging.error(f"Prediction failed for note_id {nid}: {e}")
            
    # Fusionner tous les DataFrames en un seul
    if all_preds_list:
        all_preds = pd.concat(all_preds_list, ignore_index=True)
    else:
        all_preds = pd.DataFrame()
        
    logging.info(f"Raw predictions finished in {time.time() - pred_start_time:.2f} seconds.")

    if all_preds.empty:
        logging.warning("Pipeline finished early: No predictions were generated.")
        return

    logging.info(f"Generated a total of {len(all_preds)} raw predictions.")

    # ---------------------------------------------------------
    # ÉTAPE 3 : EXTRACTION DES EN-TÊTES (CONTEXTE SPATIAL)
    # ---------------------------------------------------------
    logging.info("Step 3/4: Pre-computing spatial context (Headers) for all notes...")
    lowercase_headers = [h.lower() for h in common_headers]
    note_sections = {}
    
    # Barre de progression tqdm sur la boucle des notes
    for nid, text in tqdm(all_notes.items(), desc="Extracting Headers", total=len(all_notes), unit="note"):
        try:
            h_pos, pos_h = get_sections(str(text).lower(), lowercase_headers)
            note_sections[nid] = (h_pos, pos_h)
        except Exception as e:
            logging.error(f"Failed to extract headers for note_id {nid}: {e}")
            note_sections[nid] = ([], {})  # contexte vide par défaut

    # ---------------------------------------------------------
    # ÉTAPE 4 : SCORE RAPIDE (ITERTUPLES)
    # ---------------------------------------------------------
    logging.info("Step 4/4: Applying historical scoring logic with Tau penalty (Highly Optimized)...")
    TAU = 0.3
    scores_list = []
    
    # itertuples() au lieu d'apply(axis=1) : beaucoup plus rapide avec Pandas
    for row in tqdm(all_preds.itertuples(), desc="Calculating Scores", total=len(all_preds), unit="pred"):
        try:
            # Positions d'en-têtes en cache pour cette note
            h_pos, pos_h = note_sections.get(row.note_id, ([], {}))
            
            # Trouver la section clinique
            section = get_header_by_pos(int(row.start), h_pos, pos_h, lowercase_headers) or "other"
            
            # Normaliser pour coller aux clés du dictionnaire de debug
            sec_clean = str(section).lower().strip()
            dict_clean = str(row.dict_entry).lower().strip()
            
            # Chercher dans l'historique
            vals = scores_by_mention.get((sec_clean, dict_clean), [])
            if not vals:
                vals = scores_by_mention.get((section, row.dict_entry), [])
                
            # Calculer le score ajusté
            if vals:
                p_matches = sum(1 for v in vals if v == 1 or v == 1.0)
                m_additions = len(vals) - p_matches
                denominator = p_matches + (TAU * m_additions)
                
                score = round(p_matches / denominator, 2) if denominator > 0 else 0.0
            else:
                score = 1.0 # Default confidence for completely new combinations
                
            scores_list.append(score)
            
        except AttributeError as e:
            logging.error(f"Missing expected column in prediction DataFrame at index {row.Index}: {e}")
            scores_list.append(0.0)
        except Exception as e:
            logging.error(f"Unexpected error calculating score at index {row.Index}: {e}")
            scores_list.append(0.0)

    # Attacher la liste de scores au DataFrame
    all_preds["score"] = scores_list

    # ---------------------------------------------------------
    # ÉTAPE 5 : ENREGISTRER LES RÉSULTATS
    # ---------------------------------------------------------
    output_path = base_dir / "data/dict_global_predictions.csv"
    logging.info(f"Saving {len(all_preds)} scored predictions to disk...")
    
    try:
        # Créer le dossier si besoin
        output_path.parent.mkdir(parents=True, exist_ok=True)
        all_preds.to_csv(output_path, index=False)
        
        total_time = time.time() - start_time
        logging.info(f"SUCCESS! Pipeline completed in {total_time/60:.2f} minutes.")
        logging.info(f"File saved successfully at: {output_path}")
        
    except Exception as e:
        logging.critical(f"Failed to write output CSV: {e}")

if __name__ == "__main__":
    main()