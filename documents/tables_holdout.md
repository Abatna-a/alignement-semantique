# Tables hold-out — KIRIs × SnoBERT

L'oracle voit le gold : plafond, pas une méthode.
KIRIs ici = cœur `(section, mention)` sur le train cutmed, sans OMOP/abbréviations.

## Option A — KIRIs vs SnoBERT complet (2 étages + static dict)

Dictionnaire = KIRIs cœur `(header, mention)`. Neural = SnoBERT complet.

| Système | mIoU caractère | Rappel mention |
|---|---:|---:|
| KIRIs (cœur) | 0.3863 | 0.7572 |
| SnoBERT complet | 0.4903 | 0.7680 |
| Oracle | 0.7244 | 0.8509 |
| Δ vs meilleur des deux | 0.2340 | 0.0829 |

| Régime | n | % | P(erreur) | KIRIs juste / S faux | S juste / KIRIs faux | Les deux justes |
|---|---:|---:|---:|---:|---:|---:|
| ACCORD | 6168 | 71.7 | 0.060 | 0 | 0 | 5798 |
| CONFLIT | 506 | 5.9 | 0.115 | 196 | 252 | 0 |
| D_SEUL | 743 | 8.6 | 0.304 | 517 | 0 | 0 |
| S_SEUL | 653 | 7.6 | 0.152 | 0 | 554 | 0 |
| AUCUN | 529 | 6.2 | 1.000 | 0 | 0 | 0 |

## Option B — KIRIs vs résidu neural SnoBERT

Dictionnaire = KIRIs cœur `(header, mention)`. Neural = SnoBERT hors vocabulaire du static dict.

| Système | mIoU caractère | Rappel mention |
|---|---:|---:|
| KIRIs (cœur) | 0.3863 | 0.7572 |
| SnoBERT hors vocabulaire du static dict | 0.1365 | 0.0659 |
| Oracle | 0.7047 | 0.8150 |
| Δ vs meilleur des deux | 0.3184 | 0.0578 |

| Régime | n | % | P(erreur) | KIRIs juste / S faux | S juste / KIRIs faux | Les deux justes |
|---|---:|---:|---:|---:|---:|---:|
| ACCORD | 79 | 0.9 | 0.114 | 0 | 0 | 70 |
| CONFLIT | 246 | 2.9 | 0.138 | 73 | 139 | 0 |
| D_SEUL | 7092 | 82.5 | 0.102 | 6368 | 0 | 0 |
| S_SEUL | 413 | 4.8 | 0.133 | 0 | 358 | 0 |
| AUCUN | 769 | 8.9 | 1.000 | 0 | 0 | 0 |

Option B : pas de poids NER/SapBERT sur disque, donc pas de vrai rerun sans dict. On garde les spans SnoBERT dont la surface n'est pas dans le static dict train (longue traîne). Ce n'est pas le pipeline 2 étages entier sans `choose_concepts`.
