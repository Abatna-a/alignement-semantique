#!/usr/bin/env bash
# Crée les environnements Python du kit.
# Usage (depuis Alignement-semantique/) :
#   bash kit/setup_envs.sh
#   bash kit/setup_envs.sh --cpu-only    # PyTorch CPU si pas de GPU
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"
CPU_ONLY=0
[[ "${1:-}" == "--cpu-only" ]] && CPU_ONLY=1

python3 --version

echo "==> .venv_dict (dictionnaire, CPU)"
python3 -m venv .venv_dict
# shellcheck disable=SC1091
source .venv_dict/bin/activate
pip install --upgrade pip
pip install -r dictionnaire/requirements.txt
pip install streamlit
deactivate

echo "==> .venv_snobert (entraînement / PARHAF / mixup)"
python3 -m venv .venv_snobert
# shellcheck disable=SC1091
source .venv_snobert/bin/activate
pip install --upgrade pip
if [[ "$CPU_ONLY" -eq 1 ]]; then
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
else
  echo "Installer PyTorch CUDA adapté à la machine, par ex. :"
  echo "  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121"
  echo "Puis relancer : pip install -r snobert/requirements.txt"
  # Tentative cu121 ; si échec, l'utilisateur ajuste
  pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121 \
    || pip install torch torchvision torchaudio
fi
pip install -r snobert/requirements.txt
pip install streamlit
deactivate

echo
echo "OK."
echo "  source .venv_dict/bin/activate      # dictionnaire + interfaces dict"
echo "  source .venv_snobert/bin/activate   # SnoBERT / PARHAF / mixup"
echo "  export SNOMED_EL_HOME=\"$ROOT\"    # chemins mémoire / données locales"
