#!/usr/bin/env bash
# Prépare un dossier snomedEL portable (clé USB / disque externe → VM).
# Pas besoin d'accès réseau à la VM : tu copies le dossier, tu le poses sur la VM.
#
# Usage :
#   bash kit/pack_snomedel.sh
#   bash kit/pack_snomedel.sh /media/USB/snomedEL
#   bash kit/pack_snomedel.sh /media/USB/snomedEL --with-weights
#   bash kit/pack_snomedel.sh /media/USB/snomedEL --full
#
# Modes :
#   (défaut)     données utiles pour reprendre (notes, RF2, démos, preprocess, PARHAF)
#   --with-weights  + checkpoints NER + SapBERT (~6 Go de plus)
#   --full          presque tout snomedEL sauf les énormes output/ data107/ data123
set -euo pipefail

SRC="${SNOMED_EL_HOME:-/Data/AMA/snomedEL}"
OUT="${1:-/Data/AMA/tmp/snomedEL_portable}"
MODE="useful"
for arg in "$@"; do
  case "$arg" in
    --with-weights) MODE="weights" ;;
    --full) MODE="full" ;;
  esac
done
if [[ "${1:-}" == --* ]]; then
  OUT="/Data/AMA/tmp/snomedEL_portable"
fi

if [[ ! -d "$SRC" ]]; then
  echo "Source introuvable : $SRC"
  exit 1
fi

echo "==> snomedEL portable ($MODE) -> $OUT"
echo "    source : $SRC"
rm -rf "$OUT"
mkdir -p "$OUT"

copy_tree() {
  local from="$1"
  local to="$2"
  if [[ -e "$from" ]]; then
    mkdir -p "$(dirname "$to")"
    echo "  + $from"
    rsync -a "$from" "$to"
  else
    echo "  - absent : $from"
  fi
}

# --- toujours : squelette utile (chemins attendus par memoire/paths.py) ---
copy_tree "$SRC/SnoBERT/data/competition_data/" "$OUT/SnoBERT/data/competition_data/"
copy_tree "$SRC/SnoBERT/data/preprocess_data/" "$OUT/SnoBERT/data/preprocess_data/"
copy_tree "$SRC/SnoBERT/data/parhaf_infectiology/" "$OUT/SnoBERT/data/parhaf_infectiology/"
copy_tree "$SRC/demo_dict/" "$OUT/demo_dict/"
copy_tree "$SRC/demo_snobert/" "$OUT/demo_snobert/"
copy_tree "$SRC/demo_parhaf/" "$OUT/demo_parhaf/"
copy_tree "$SRC/dictmethod/" "$OUT/dictmethod/"
copy_tree "$SRC/oracle_study/" "$OUT/oracle_study/"

# prédictions / baselines utiles au mémoire
for f in submission_baseline.csv submission_mixup.csv; do
  if [[ -f "$SRC/SnoBERT/$f" ]]; then
    mkdir -p "$OUT/SnoBERT"
    cp -a "$SRC/SnoBERT/$f" "$OUT/SnoBERT/$f"
    echo "  + SnoBERT/$f"
  fi
done

if [[ "$MODE" == "weights" || "$MODE" == "full" ]]; then
  echo "==> poids"
  copy_tree "$SRC/SnoBERT/data/first_stage/" "$OUT/SnoBERT/data/first_stage/"
  copy_tree "$SRC/SnoBERT/data/second_stage/" "$OUT/SnoBERT/data/second_stage/"
fi

if [[ "$MODE" == "full" ]]; then
  echo "==> mode full (sans output/ ni data107/data123)"
  # autres dossiers data* légers éventuels, configs, src déjà hors git kit
  if [[ -d "$SRC/SnoBERT" ]]; then
    rsync -a \
      --exclude 'output/' \
      --exclude 'output_parhaf_ner/' \
      --exclude 'data107/' \
      --exclude 'data123/' \
      --exclude '__pycache__/' \
      --exclude '.cache/' \
      "$SRC/SnoBERT/" "$OUT/SnoBERT/"
  fi
fi

# PARHAF csv (vit à côté dans /Data/AMA/parhaf, utile avec prepare_parhaf)
if [[ -d /Data/AMA/parhaf/csv ]]; then
  mkdir -p "$OUT/_extra/parhaf"
  copy_tree "/Data/AMA/parhaf/csv/" "$OUT/_extra/parhaf/csv/"
fi

cat > "$OUT/LIRE_SUR_VM.txt" << EOF
Données snomedEL — transfert hors réseau
========================================

1) Copier CE DOSSIER entier sur une clé USB / disque externe.

2) Sur la VM, le placer par exemple :
     /home/USER/snomedEL
   (le contenu doit rester : SnoBERT/, demo_dict/, oracle_study/, …)

3) Dans le shell de la VM :
     export SNOMED_EL_HOME=/home/USER/snomedEL

   Les scripts du mémoire (Alignement-semantique) liront alors
   \$SNOMED_EL_HOME/SnoBERT/data/competition_data/ etc.

4) Vérifier :
     ls "\$SNOMED_EL_HOME/SnoBERT/data/competition_data"
     ls "\$SNOMED_EL_HOME/demo_dict/assets"

Mode de ce pack : $MODE
Généré le : $(date -Iseconds)
Source : $SRC
EOF

echo
echo "OK."
du -sh "$OUT" 2>/dev/null
echo
echo "Ensuite (hors réseau) :"
echo "  1. Copier $OUT sur une clé USB"
echo "  2. Sur la VM : coller le dossier, par ex. /home/USER/snomedEL"
echo "  3. export SNOMED_EL_HOME=/home/USER/snomedEL"
echo
echo "Voir $OUT/LIRE_SUR_VM.txt"
