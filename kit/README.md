# Kit VM

Objectif : sur une machine entreprise, reprendre le travail avec le code, les données utiles, et deux environnements Python.

## Contenu attendu après `pack.sh`

```
alignement_kit/
├── KIT.md
├── MANIFEST_DONNEES.txt
└── Alignement-semantique/
    ├── dictionnaire/data/raw/…
    ├── dictionnaire/data/interim/…     # démarrage rapide
    ├── dictionnaire/assets/            # kiri_dicts.pkl, etc.
    ├── snobert/data/competition_data/…
    ├── snobert/data/preprocess_data/…
    ├── snobert/data/parhaf_infectiology/…
    ├── parhaf/csv/…
    ├── kit/
    └── …
```

## Sur la machine source (ici)

```bash
cd /Data/AMA/Alignement-semantique
bash kit/pack.sh /Data/AMA/tmp/alignement_kit
# avec poids NER + SapBERT (~6 Go de plus) :
# bash kit/pack.sh /Data/AMA/tmp/alignement_kit --with-checkpoints
```

Puis copier vers la VM :

```bash
rsync -avh --progress /Data/AMA/tmp/alignement_kit/ user@vm:/chemin/alignement_kit/
```

## Sur la VM

```bash
cd /chemin/alignement_kit/Alignement-semantique
bash kit/setup_envs.sh          # ou : bash kit/setup_envs.sh --cpu-only
bash kit/verify.sh
export SNOMED_EL_HOME="$(pwd)"
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
