# Alignement sémantique

Deux façons de rattacher une mention clinique à un concept SNOMED CT, puis de mesurer le lien sur les notes d'origine. Un dictionnaire construit sur les notes annotées, un modèle en deux étages (spans, puis concept), un passage au français, un mélange des représentations pendant l'entraînement, et une comparaison qui dit quelles mentions relire.

---

## Vue d'ensemble

```
Compte rendu
     |
     v
+---------------------------+
| Dictionnaire              |  (section, mention) → code
|   ou                      |
| SnoBERT                   |  spans, puis SapBERT + dictionnaire statique
+---------------------------+
     |
     v
  note_id, start, end, concept_id
     |
     +-- lecture (Streamlit)
     +-- comparaison des deux sorties, file de relecture
```

Le dictionnaire et SnoBERT produisent la même table. Le français (PARHAF, CamemBERT-bio) s'arrête aux spans : les étiquettes sont Infection, Site, Bacterie, Bacteriemie.

---

## Installation

Python 3.10 ou plus. Un GPU NVIDIA sert à l'entraînement de SnoBERT et de PARHAF. Le dictionnaire tourne sur CPU.

```bash
git clone https://github.com/Abatna-a/alignement-semantique.git
cd alignement-semantique

python3 -m venv .venv
source .venv/bin/activate

pip install -r dictionnaire/requirements.txt
pip install -r snobert/requirements.txt
pip install streamlit
```

PyTorch s'installe à part, selon la version de CUDA, avant le reste de `snobert/requirements.txt`.

---

## Données

Le dépôt ne contient pas les notes, les annotations ni la terminologie. Il faut les placer soi-même. Trois familles de fichiers :

- des **comptes rendus** et leurs **annotations** (identifiant de note, bornes, code) ;
- une **release SNOMED CT** au format RF2 (libellés et relations) ;
- le **vocabulaire OMOP / Athena** (synonymes), pour le dictionnaire.

Les notes annotées viennent en pratique du jeu SNOMED CT Entity Linking (PhysioNet / MIMIC-IV Note). Le RF2 et Athena s'obtiennent auprès de SNOMED International et d'OHDSI. PARHAF, lui, se télécharge par script : corpus fictif français, déjà étiqueté en infectiologie.

Une fois rangés, les dossiers doivent ressembler à ceci.

**Dictionnaire** — `dictionnaire/data/`

```
dictionnaire/data/
├── raw/
│   ├── athena/
│   │   ├── CONCEPT.csv
│   │   └── CONCEPT_RELATIONSHIP.csv
│   ├── discharge.csv.gz
│   ├── medical_abbreviations.csv
│   ├── mimic-iv_notes_training_set.csv
│   ├── train_annotations.csv
│   └── SnomedCT_InternationalRF2_PRODUCTION_20230531T120000Z_Challenge_Edition/
│       └── Snapshot/Terminology/
├── interim/                  # rempli par process_data.py
└── test_notes.csv            # notes à annoter à l'inférence
```

**SnoBERT** — `snobert/data/`

```
snobert/data/
├── competition_data/
│   ├── mimic-iv_notes_training_set.csv
│   ├── train_annotations.csv
│   ├── test_notes.csv
│   ├── test_annotations.csv
│   └── SnomedCT_InternationalRF2_PRODUCTION_20260201T120000Z/
│       └── Snapshot/Terminology/
├── preprocess_data/          # rempli par preprocess.py
├── first_stage/              # checkpoints du reconnaisseur (à y copier)
├── second_stage/
│   └── sapbert/              # embeddings SapBERT
└── parhaf_infectiology/      # rempli par prepare_parhaf_ner.py
```

Le nom exact du dossier RF2 sous `competition_data/` est celui lu par `snobert/src/preprocess.py`. Pour le dictionnaire, c'est le nom du Challenge Edition sous `raw/`.

**PARHAF** — après `python parhaf/fetch_parhaf.py` :

```
parhaf/
├── csv/
└── flat/
```

**Mémoire et interfaces.** `memoire/paths.py` lit le dossier racine des données dans `SNOMED_EL_HOME` (notes, prédictions déjà calculées, caches). Les applications Streamlit attendent leurs CSV à côté d'elles, dans `interfaces/.../data/`.

Les chemins d'entraînement se règlent aussi dans `snobert/configs/snom.yaml` et `snobert/configs/parhaf_ner.yaml`.

---

## Dictionnaire

Depuis `dictionnaire/`. Les sources brutes sont sous `dictionnaire/data/raw/` (voir [Données](#données)).

Préparer les libellés, les synonymes, les annotations et les abréviations :

```bash
cd dictionnaire
python src/process_data.py make-flattened-terminology
python src/process_data.py make-synonyms
python src/process_data.py make-clean-annotations
python src/process_data.py make-abbreviations
python src/process_data.py make-abbr-dict
python src/process_data.py make-term-extension
python src/process_data.py make-unigrams
```

Construire le dictionnaire `(section, mention) → code` se fait en appelant `make_kiri_dicts` depuis `dictionnaire/src` :

```python
from pathlib import Path
import pandas as pd
from mimic_dev_main import make_kiri_dicts

texts = pd.read_csv("../data/raw/mimic-iv_notes_training_set.csv").set_index("note_id")["text"]
annotations = pd.read_csv("../data/interim/train_annotations_cln.csv")
make_kiri_dicts(texts, annotations, Path(".."))
```

Le pickle est écrit dans `dictionnaire/assets/kiri_dicts.pkl`. Le dossier `assets/` doit exister avant l'appel (`mkdir -p assets` depuis `dictionnaire/`).

Appliquer le dictionnaire. Le dossier courant doit contenir `assets/kiri_dicts.pkl` et `data/test_notes.csv`. La sortie est `submission.csv`.

```bash
cd dictionnaire
python src/mimic_submission_main.py
```

Mesurer la sortie contre les annotations de référence :

```bash
python src/scoring.py submission.csv data/interim/train_annotations_cln.csv
```

---

## SnoBERT

Depuis `snobert/`. Les notes, les annotations et le RF2 sont sous `snobert/data/competition_data/` (voir [Données](#données)).

Préparer les synonymes, les embeddings et le dictionnaire statique :

```bash
cd snobert
python src/preprocess.py
python src/preprocess.py --val
```

Entraîner le reconnaisseur de spans (BiomedBERT). Le réglage est `configs/snom.yaml`.

```bash
torchrun --nproc-per-node=1 src/main.py PARALLEL.DDP=false
```

Un exemple de reprise avec un split et un nombre d'époques :

```bash
torchrun --nproc-per-node=1 src/main.py split=0 epochs=80 PARALLEL.DDP=false
```

Les poids sont écrits sous `output/<date>/<heure>/models/`. Pour l'inférence, copier les dossiers de checkpoints dans `data/first_stage/`, et les embeddings SapBERT dans `data/second_stage/sapbert/`.

Lier les spans à un concept, puis ajouter le dictionnaire statique :

```bash
python submission/main.py
python submission/main.py --val
python submission/main.py --score-test
```

`--val` score un pli. `--score-test` score `data/competition_data/test_annotations.csv`. La sortie est `submission_baseline.csv`.

---

## Mélange à l'entraînement

Le mélange (Augmented Mixup Procedure for Privacy-Preserving Collaborative Training) ne s'applique que pendant l'entraînement du reconnaisseur. Le lieur ne change pas. Le score se calcule ensuite sur les notes d'origine.

```bash
cd snobert
torchrun --nproc-per-node=1 src/main.py mixup=true mixup_alpha=0.7 mixup_tau=1.0 PARALLEL.DDP=false
```

`mixup_tau` règle l'intensité du bruit. Pour scorer un dossier de checkpoints entraînés avec le mélange :

```bash
python submission/main.py --mixup --score-test --first-stage-dir data/first_stage
```

---

## Français (PARHAF)

Récupérer le corpus fictif, le mettre à plat, puis entraîner CamemBERT-bio.

```bash
python parhaf/fetch_parhaf.py
python parhaf/export_flat.py

cd snobert
python src/prepare_parhaf_ner.py
torchrun --nproc-per-node=1 src/main_parhaf.py PARALLEL.DDP=false
```

Le réglage est `configs/parhaf_ner.yaml` (`epochs`, `split`, `model`). Les notes préparées sont dans `snobert/data/parhaf_infectiology/`.

Écrire les spans prédits à partir d'un checkpoint :

```bash
python interfaces/parhaf/infer_submission.py --split both --ckpt /chemin/vers/le/checkpoint
```

Les CSV arrivent dans `interfaces/parhaf/data/` (`submission_val.csv`, `submission_test.csv`).

---

## Lecture

Streamlit sert à ouvrir une sortie déjà calculée. Les fichiers attendus sont à côté de chaque application.

Dictionnaire. D'abord le tableau global, puis l'application. Elle lit `interfaces/dictionnaire/assets/kiri_dicts.pkl`, `data/train_notes.csv`, `data/train_annotations_cln.csv` et `data/dict_global_predictions.csv`.

```bash
python interfaces/dictionnaire/precompute_dict_preds.py
streamlit run interfaces/dictionnaire/app6.py
```

SnoBERT. L'application lit `interfaces/snobert/data/test_notes.csv`, `test_annotations.csv` et `submission.csv`.

```bash
streamlit run interfaces/snobert/app_snobert.py
```

PARHAF. L'application lit `submission_val.csv` et `submission_test.csv` dans `interfaces/parhaf/data/`, et les notes dans `snobert/data/parhaf_infectiology/` (ou le dossier donné par `PARHAF_DATA`).

```bash
streamlit run interfaces/parhaf/app_parhaf.py
```

---

## Comparaison et file de relecture

Depuis la racine du dépôt. Les chemins des notes et des prédictions sont dans `memoire/paths.py`.

```bash
python -m memoire.run_holdout --pair both
python -m memoire.run_holdout --pair A --force
python -m memoire.run_review_queue
```

`--pair A` compare le dictionnaire au SnoBERT complet. `--pair B` compare le dictionnaire aux spans dont la surface n'est pas dans le dictionnaire statique. `--force` reconstruit le dictionnaire de la mesure et refait la prédiction sur les notes tenues de côté.

La file garde tout sauf l'accord des deux moteurs.

Vérifier le calcul sans relancer l'inférence :

```bash
python -m unittest memoire.tests.test_oracle memoire.tests.test_align memoire.tests.test_review_queue
```

---

## Arborescence

```
alignement-semantique/
├── dictionnaire/
│   ├── requirements.txt
│   └── src/
│       ├── process_data.py                  # libellés, synonymes, abréviations
│       ├── mimic_train.py                   # construction du dictionnaire
│       ├── mimic_dev_main.py                # make_kiri_dicts
│       ├── mimic_predict.py                 # application à une note
│       ├── mimic_submission_main.py         # écrit submission.csv
│       └── scoring.py                       # mIoU
├── snobert/
│   ├── configs/
│   │   ├── snom.yaml                        # entraînement BiomedBERT
│   │   └── parhaf_ner.yaml                  # entraînement CamemBERT-bio
│   ├── src/
│   │   ├── preprocess.py                    # synonymes, embeddings, dictionnaire statique
│   │   ├── main.py                          # entraînement des spans
│   │   ├── train.py                         # boucle, appelle le mélange
│   │   ├── mixup.py                         # mélange des représentations
│   │   ├── main_parhaf.py                   # entraînement français
│   │   └── prepare_parhaf_ner.py            # notes et annotations PARHAF
│   └── submission/
│       └── main.py                          # spans, SapBERT, dictionnaire statique
├── parhaf/
│   ├── fetch_parhaf.py
│   └── export_flat.py
├── interfaces/
│   ├── dictionnaire/app6.py
│   ├── dictionnaire/precompute_dict_preds.py
│   ├── snobert/app_snobert.py
│   └── parhaf/
│       ├── infer_submission.py
│       └── app_parhaf.py
├── memoire/
│   ├── run_holdout.py
│   ├── run_review_queue.py
│   └── paths.py
└── documents/
```
