"""Chemins des notes, dictionnaires, caches et sorties de l'étude."""
import os
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
DATA_ROOT = Path(os.environ.get("SNOMED_EL_HOME", REPO))

SNOBERT = REPO / "snobert"
DICTMETHOD = REPO / "dictionnaire"
DICTMETHOD_SRC = DICTMETHOD / "src"

COMPETITION = DATA_ROOT / "SnoBERT" / "data" / "competition_data"
PREPROCESS = DATA_ROOT / "SnoBERT" / "data" / "preprocess_data"

TRAIN_NOTES = COMPETITION / "cutmed_notes.csv"
TRAIN_ANN = COMPETITION / "cutmed_fixed_train_annotations.csv"
TEST_NOTES = COMPETITION / "test_notes.csv"
TEST_ANN = COMPETITION / "test_annotations.csv"

SNOBERT_FULL = DATA_ROOT / "SnoBERT" / "submission_baseline.csv"

DEMO_DICT = DATA_ROOT / "demo_dict"
DEMO_ASSETS = DEMO_DICT / "assets"
LEAKED_KIRI_DICTS = DEMO_ASSETS / "kiri_dicts.pkl"
ABBR_DICT = DEMO_ASSETS / "abbr_dict.pkl"
TERM_EXTENSION = DEMO_ASSETS / "term_extension.csv"
FLAT_SNOMED = DEMO_DICT / "data" / "flattened_terminology.csv"

OUTPUT_DIR = DATA_ROOT / "oracle_study" / "output"
KIRI_CACHE = OUTPUT_DIR / "kiri_core.pkl"
KIRI_FULL_CACHE = OUTPUT_DIR / "kiri_full_dicts.pkl"
KIRI_PRED = OUTPUT_DIR / "submission_kiri.csv"
SNOBERT_NODICT = OUTPUT_DIR / "submission_snobert_nodict.csv"
STATIC_DICT_PRED = OUTPUT_DIR / "submission_dict_only.csv"
TABLES_MD = OUTPUT_DIR / "tables_memoire.md"
DICT_CACHE = OUTPUT_DIR / "most_common_concept.pkl"
ALIGN_A = OUTPUT_DIR / "align_A.csv"
QUEUE_MD = OUTPUT_DIR / "table_file_relecture.md"
QUEUE_CSV = OUTPUT_DIR / "file_relecture_A.csv"
EXAMPLES_CSV = OUTPUT_DIR / "exemples_A.csv"
