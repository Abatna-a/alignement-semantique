# Comparaison et relecture

Les deux moteurs sont comparés mention par mention. L'oracle est un plafond : il dit ce qu'on gagnerait si, pour chaque mention, on gardait la bonne réponse quand l'un des deux l'a. La file de relecture garde tout sauf l'accord.

```bash
python -m unittest memoire.tests.test_oracle memoire.tests.test_align memoire.tests.test_review_queue
python -m memoire.run_holdout --pair both
python -m memoire.run_review_queue
```

- `kiri.py` — dictionnaire construit sur les notes d'apprentissage.
- `kiri_full.py` — dictionnaire construit sur toutes les notes annotées.
- `run_holdout.py` — comparaison et oracle.
- `review_queue.py` — file : tout sauf l'accord.
