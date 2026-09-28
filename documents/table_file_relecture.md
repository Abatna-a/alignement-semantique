# File de relecture (option A, sans gold)

Politique EDS : on **émet** KIRIs ∪ SnoBERT ; on **relit** tout sauf ACCORD.
L'oracle n'est pas utilisé à l'inférence : seulement `regime` (désaccord / unilatéral).

## Table 3 — efficacité de la file

| Mentions | 8599 |
| Erreurs (ni KIRIs ni SnoBERT justes) | 1282 |
| Taille de la file | 2431 (28.3 % des mentions) |
| Rappel des erreurs (file) | 71.1 % |
| Précision des erreurs (file) | 37.5 % |
| Erreurs restantes dans ACCORD (file aveugle) | 370 (28.9 % des erreurs) |

## Détail par régime

| Régime | Dans la file ? | n | Erreurs | P(erreur) | Part des erreurs |
|---|---|---:|---:|---:|---:|
| ACCORD | non | 6168 | 370 | 0.060 | 28.9 % |
| CONFLIT | oui | 506 | 58 | 0.115 | 4.5 % |
| D_SEUL | oui | 743 | 226 | 0.304 | 17.6 % |
| S_SEUL | oui | 653 | 99 | 0.152 | 7.7 % |
| AUCUN | oui | 529 | 529 | 1.000 | 41.3 % |
