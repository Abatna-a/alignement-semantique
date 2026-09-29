# Kit VM

Sur la VM : reprendre le travail avec le code, les données utiles, et deux environnements Python.

Sans accès réseau à la VM : on prépare des dossiers ici, on les met sur une **clé USB** (ou disque), on les recolle sur la VM.

Deux paquets :

1. **Code + données déjà rangées** → `pack.sh`  
2. **Arbre `snomedEL` d’origine** (chemins du mémoire) → `pack_snomedel.sh`

## 1) Code (Alignement-semantique)

```bash
cd /Data/AMA/Alignement-semantique
bash kit/pack.sh /Data/AMA/tmp/alignement_kit
# option : bash kit/pack.sh … --with-checkpoints
```

Mettre `/Data/AMA/tmp/alignement_kit` sur la clé USB.

## 2) Données snomedEL (layout d’origine)

```bash
cd /Data/AMA/Alignement-semantique
bash kit/pack_snomedel.sh /Data/AMA/tmp/snomedEL_portable
# avec poids NER + SapBERT :
# bash kit/pack_snomedel.sh /Data/AMA/tmp/snomedEL_portable --with-weights
```

Mettre `/Data/AMA/tmp/snomedEL_portable` sur la clé USB (dossier entier).

Sur la VM, le coller par exemple en `/home/USER/snomedEL` (il doit contenir `SnoBERT/`, `demo_dict/`, …).

## Sur la VM

```bash
cd /chemin/alignement_kit/Alignement-semantique
bash kit/setup_envs.sh          # ou : bash kit/setup_envs.sh --cpu-only
bash kit/verify.sh

# Pointer vers le snomedEL collé depuis la clé :
export SNOMED_EL_HOME=/home/USER/snomedEL
```

- Dictionnaire / interface dict : `source .venv_dict/bin/activate`
- SnoBERT / PARHAF / mixup : `source .venv_snobert/bin/activate`

Les commandes détaillées restent dans le `README.md` à la racine du dépôt.

## Ce que le pack inclut

| Élément | Rôle |
|---|---|
| Notes + annotations challenge | Entraînement / hold-out |
| RF2 SNOMED | Terminologie |
| PARHAF csv + tables infectio | Français |
| Interim + `kiri_dicts.pkl` | Dictionnaire utilisable sans tout reconstruire |
| `preprocess_data` | Synonymes / splits déjà calculés |

## Ce qu’il faut encore ajouter à la main (si on reconstruit le dictionnaire from scratch)

Placer sous `dictionnaire/data/raw/` :

- `athena/CONCEPT.csv` et `athena/CONCEPT_RELATIONSHIP.csv`
- `discharge.csv.gz`
- `medical_abbreviations.csv`

Sans ces fichiers, on s’appuie sur l’interim et les assets déjà fournis.

## Environnements

Deux venv séparés (dépendances différentes) :

1. `.venv_dict` — lookup CPU, Streamlit dictionnaire  
2. `.venv_snobert` — PyTorch, Hydra, Transformers, Streamlit SnoBERT/PARHAF  

On ne copie pas le gros conda `snoBERT` (~9 Go) : trop fragile d’une machine à l’autre. `setup_envs.sh` le recrée.
