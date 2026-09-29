#!/usr/bin/env bash
# Assemble un dossier prêt à copier sur la VM entreprise.
# Usage :
#   ./kit/pack.sh
#   ./kit/pack.sh /chemin/vers/alignement_kit
#   ./kit/pack.sh /chemin/vers/alignement_kit --with-checkpoints
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
OUT="${1:-/Data/AMA/tmp/alignement_kit}"
WITH_CHECKPOINTS=0
for arg in "$@"; do
  case "$arg" in
    --with-checkpoints) WITH_CHECKPOINTS=1 ;;
  esac
done
# Si le premier argument est un flag, OUT par défaut
if [[ "${1:-}" == --* ]]; then
  OUT="/Data/AMA/tmp/alignement_kit"
fi

SRC_SNOMED_EL="${SNOMED_EL_HOME:-/Data/AMA/snomedEL}"
SRC_COMP="$SRC_SNOMED_EL/SnoBERT/data/competition_data"
SRC_RF2="$SRC_COMP/SnomedCT_InternationalRF2_PRODUCTION_20260201T120000Z"
SRC_PARHAF_CSV="${PARHAF_CSV:-/Data/AMA/parhaf/csv}"
SRC_PARHAF_NER="$SRC_SNOMED_EL/SnoBERT/data/parhaf_infectiology"
SRC_DEMO_DATA="$SRC_SNOMED_EL/demo_dict/data"
SRC_DEMO_ASSETS="$SRC_SNOMED_EL/demo_dict/assets"
SRC_PREPROCESS="$SRC_SNOMED_EL/SnoBERT/data/preprocess_data"
SRC_FIRST="$SRC_SNOMED_EL/SnoBERT/data/first_stage"
SRC_SECOND="$SRC_SNOMED_EL/SnoBERT/data/second_stage"

DEST="$OUT/Alignement-semantique"
DICT_RAW="$DEST/dictionnaire/data/raw"
DICT_INTERIM="$DEST/dictionnaire/data/interim"
SNO_COMP="$DEST/snobert/data/competition_data"
CHALLENGE_NAME="SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition"

echo "==> Kit -> $OUT"
rm -rf "$OUT"
mkdir -p "$DEST"

echo "==> Code"
rsync -a \
  --exclude '.git' \
  --exclude '__pycache__' \
  --exclude '.venv*' \
  --exclude '*/data/' \
  --exclude 'snobert/output*' \
  "$ROOT/" "$DEST/"

# Remettre kit/ et le reste du code (data exclus ci-dessus : on les reconstruit)
mkdir -p "$DICT_RAW" "$DICT_INTERIM" "$SNO_COMP" \
  "$DEST/snobert/data/preprocess_data" \
  "$DEST/snobert/data/parhaf_infectiology" \
  "$DEST/parhaf/csv" \
  "$DEST/interfaces/dictionnaire/data" \
  "$DEST/interfaces/dictionnaire/assets" \
  "$DEST/interfaces/snobert/data" \
  "$DEST/interfaces/parhaf/data"

echo "==> Données SnoBERT (notes + annotations)"
for f in \
  mimic-iv_notes_training_set.csv \
  train_annotations.csv \
  test_notes.csv \
  test_annotations.csv \
  cutmed_notes.csv \
  cutmed_fixed_train_annotations.csv
do
  if [[ -f "$SRC_COMP/$f" ]]; then
    cp -a "$SRC_COMP/$f" "$SNO_COMP/$f"
  else
    echo "  [manquant] $SRC_COMP/$f"
  fi
done

echo "==> RF2 pour SnoBERT"
if [[ -d "$SRC_RF2" ]]; then
  rsync -a "$SRC_RF2" "$SNO_COMP/"
else
  echo "  [manquant] $SRC_RF2"
fi

echo "==> Raw dictionnaire (notes + annotations partagées)"
for f in mimic-iv_notes_training_set.csv train_annotations.csv; do
  if [[ -f "$SRC_COMP/$f" ]]; then
    cp -a "$SRC_COMP/$f" "$DICT_RAW/$f"
  fi
done
# test notes pour l'inférence dictionnaire
if [[ -f "$SRC_COMP/test_notes.csv" ]]; then
  cp -a "$SRC_COMP/test_notes.csv" "$DEST/dictionnaire/data/test_notes.csv"
fi

echo "==> RF2 dictionnaire (copie + noms Challenge 20230531 attendus par process_data.py)"
if [[ -d "$SRC_RF2/Snapshot/Terminology" ]]; then
  TERM_SRC="$SRC_RF2/Snapshot/Terminology"
  TERM_DST="$DICT_RAW/$CHALLENGE_NAME/Snapshot/Terminology"
  mkdir -p "$TERM_DST"
  # Copier tout le Snapshot (portable sur la VM, pas de lien hors kit)
  rsync -a "$SRC_RF2/Snapshot/" "$DICT_RAW/$CHALLENGE_NAME/Snapshot/"
  # Alias de noms : le code dictionnaire cherche encore les fichiers *20230531*
  [[ -f "$TERM_SRC/sct2_Concept_Snapshot_INT_20260201.txt" ]] \
    && cp -a "$TERM_SRC/sct2_Concept_Snapshot_INT_20260201.txt" \
         "$TERM_DST/sct2_Concept_Snapshot_INT_20230531.txt"
  [[ -f "$TERM_SRC/sct2_Description_Snapshot-en_INT_20260201.txt" ]] \
    && cp -a "$TERM_SRC/sct2_Description_Snapshot-en_INT_20260201.txt" \
         "$TERM_DST/sct2_Description_Snapshot-en_INT_20230531.txt"
  [[ -f "$TERM_SRC/sct2_Relationship_Snapshot_INT_20260201.txt" ]] \
    && cp -a "$TERM_SRC/sct2_Relationship_Snapshot_INT_20260201.txt" \
         "$TERM_DST/sct2_Relationship_Snapshot_INT_20230531.txt"
else
  echo "  [manquant] RF2 source"
fi

echo "==> Interim dictionnaire (démarrage rapide, sans refaire process_data)"
if [[ -d "$SRC_DEMO_DATA" ]]; then
  for f in flattened_terminology.csv train_annotations_cln.csv; do
    [[ -f "$SRC_DEMO_DATA/$f" ]] && cp -a "$SRC_DEMO_DATA/$f" "$DICT_INTERIM/$f"
  done
fi
if [[ -d "$SRC_DEMO_ASSETS" ]]; then
  mkdir -p "$DEST/dictionnaire/assets"
  for f in abbr_dict.pkl kiri_dicts.pkl term_extension.csv; do
    [[ -f "$SRC_DEMO_ASSETS/$f" ]] && cp -a "$SRC_DEMO_ASSETS/$f" "$DEST/dictionnaire/assets/$f"
  done
  # Interfaces dictionnaire
  cp -a "$SRC_DEMO_ASSETS/." "$DEST/interfaces/dictionnaire/assets/" 2>/dev/null || true
  [[ -f "$SRC_DEMO_DATA/train_notes.csv" ]] && cp -a "$SRC_DEMO_DATA/train_notes.csv" "$DEST/interfaces/dictionnaire/data/"
  [[ -f "$SRC_DEMO_DATA/train_annotations_cln.csv" ]] && cp -a "$SRC_DEMO_DATA/train_annotations_cln.csv" "$DEST/interfaces/dictionnaire/data/"
  [[ -f "$SRC_DEMO_DATA/dict_global_predictions.csv" ]] && cp -a "$SRC_DEMO_DATA/dict_global_predictions.csv" "$DEST/interfaces/dictionnaire/data/"
  [[ -f "$SRC_DEMO_DATA/flattened_terminology.csv" ]] && cp -a "$SRC_DEMO_DATA/flattened_terminology.csv" "$DEST/interfaces/dictionnaire/data/"
fi

echo "==> PARHAF"
if [[ -d "$SRC_PARHAF_CSV" ]]; then
  rsync -a "$SRC_PARHAF_CSV/" "$DEST/parhaf/csv/"
fi
if [[ -d "$SRC_PARHAF_NER" ]]; then
  rsync -a "$SRC_PARHAF_NER/" "$DEST/snobert/data/parhaf_infectiology/"
fi

echo "==> Prétraitement SnoBERT (léger)"
if [[ -d "$SRC_PREPROCESS" ]]; then
  rsync -a "$SRC_PREPROCESS/" "$DEST/snobert/data/preprocess_data/"
fi

if [[ "$WITH_CHECKPOINTS" -eq 1 ]]; then
  echo "==> Checkpoints (--with-checkpoints)"
  [[ -d "$SRC_FIRST" ]] && rsync -a "$SRC_FIRST/" "$DEST/snobert/data/first_stage/"
  [[ -d "$SRC_SECOND" ]] && rsync -a "$SRC_SECOND/" "$DEST/snobert/data/second_stage/"
else
  mkdir -p "$DEST/snobert/data/first_stage" "$DEST/snobert/data/second_stage/sapbert"
  echo "  (checkpoints non inclus ; relancer pack.sh avec --with-checkpoints si besoin)"
fi

# Manifeste des manques connus
MANIFEST="$OUT/MANIFEST_DONNEES.txt"
{
  echo "Kit généré le $(date -Iseconds)"
  echo
  echo "Inclus si trouvés sur la machine source :"
  echo "  - notes / annotations challenge (SnoBERT + dictionnaire)"
  echo "  - RF2 20260201 (SnoBERT) + liens noms 20230531 (dictionnaire)"
  echo "  - PARHAF csv + tables infectiologie"
  echo "  - interim / assets démo dictionnaire"
  echo "  - preprocess_data SnoBERT"
  echo
  echo "À ajouter manuellement si absents (requis pour reconstruire le dictionnaire from scratch) :"
  echo "  dictionnaire/data/raw/athena/CONCEPT.csv"
  echo "  dictionnaire/data/raw/athena/CONCEPT_RELATIONSHIP.csv"
  echo "  dictionnaire/data/raw/discharge.csv.gz"
  echo "  dictionnaire/data/raw/medical_abbreviations.csv"
  echo
  echo "Sur la VM :"
  echo "  cd Alignement-semantique"
  echo "  bash kit/setup_envs.sh"
  echo "  bash kit/verify.sh"
  echo "  export SNOMED_EL_HOME=\"\$(pwd)\"   # ou pointer vers snomedEL si séparé"
} > "$MANIFEST"

cp -a "$DEST/kit/README.md" "$OUT/KIT.md" 2>/dev/null || true

echo
echo "OK. Bundle : $OUT"
du -sh "$OUT" "$DEST" 2>/dev/null
echo "Voir $MANIFEST"
