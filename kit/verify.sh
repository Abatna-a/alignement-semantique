#!/usr/bin/env bash
# Vérifie que le kit a le code, les données attendues et les venv.
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
ok=0
ko=0

check() {
  local path="$1"
  local label="${2:-$1}"
  if [[ -e "$path" ]]; then
    echo "  [OK] $label"
    ok=$((ok + 1))
  else
    echo "  [KO] $label"
    ko=$((ko + 1))
  fi
}

echo "==> Code"
check "dictionnaire/src/mimic_predict.py"
check "snobert/src/main.py"
check "snobert/submission/main.py"
check "memoire/run_holdout.py"
check "parhaf/fetch_parhaf.py"
check "interfaces/dictionnaire/app6.py"

echo "==> Données SnoBERT"
check "snobert/data/competition_data/mimic-iv_notes_training_set.csv"
check "snobert/data/competition_data/train_annotations.csv"
check "snobert/data/competition_data/test_notes.csv"
check "snobert/data/competition_data/test_annotations.csv"
check "snobert/data/competition_data/SnomedCT_InternationalRF2_PRODUCTION_20260201T120000Z/Snapshot/Terminology"

echo "==> Données dictionnaire"
check "dictionnaire/data/raw/mimic-iv_notes_training_set.csv"
check "dictionnaire/data/raw/train_annotations.csv"
check "dictionnaire/data/raw/SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition/Snapshot/Terminology/sct2_Description_Snapshot-en_INT_20230531.txt"
check "dictionnaire/data/interim/flattened_terminology.csv" "interim (démarrage rapide)"
check "dictionnaire/assets/kiri_dicts.pkl" "dictionnaire déjà construit (démo)"

echo "==> Optionnels (reconstruire le dictionnaire from scratch)"
check "dictionnaire/data/raw/athena/CONCEPT.csv" "Athena CONCEPT.csv"
check "dictionnaire/data/raw/discharge.csv.gz"
check "dictionnaire/data/raw/medical_abbreviations.csv"

echo "==> PARHAF"
check "parhaf/csv/infectiology_spans.csv"
check "snobert/data/parhaf_infectiology/notes.csv"

echo "==> Environnements"
check ".venv_dict/bin/python"
check ".venv_snobert/bin/python"

echo
echo "Résumé : $ok OK, $ko manquants"
if [[ "$ko" -gt 0 ]]; then
  echo "Les [KO] optionnels (Athena, discharge, abréviations) n'empêchent pas"
  echo "d'utiliser l'interim / kiri_dicts déjà fournis ni SnoBERT / PARHAF."
  exit 1
fi
exit 0
