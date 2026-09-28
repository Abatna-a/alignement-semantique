"""Score mIoU caractère macro par concept."""
from pathlib import Path
from typing import List

import numpy as np
import pandas as pd
import scipy.sparse as sp
import typer


def iou_per_class(user_annotations: pd.DataFrame, target_annotations: pd.DataFrame) -> List[float]:
    """
    Calculate the IoU metric for each class in a set of annotations.
    """
    # Correspondance note_id → index dans le tableau
    docs = np.unique(np.concatenate([user_annotations.note_id, target_annotations.note_id]))
    doc_index_mapping = dict(zip(docs, range(len(docs))))

    # Union des catégories dans la référence et les prédictions
    cats = np.unique(np.concatenate([user_annotations.concept_id, target_annotations.concept_id]))

    # Index de caractère maximal dans la référence ou les prédictions
    max_end = np.max(np.concatenate([user_annotations.end, target_annotations.end]))

    # Remplir les matrices de catégorisation caractère par caractère
    def populate_char_mtx(n_rows, n_cols, annot_df):
        mtx = sp.lil_array((n_rows, n_cols), dtype=np.uint64)
        for row in annot_df.itertuples():
            doc_index = doc_index_mapping[row.note_id]
            mtx[doc_index, row.start : row.end] = row.concept_id  # noqa: E203
        return mtx.tocsr()

    gt_mtx = populate_char_mtx(docs.shape[0], max_end, target_annotations)
    pred_mtx = populate_char_mtx(docs.shape[0], max_end, user_annotations)

    # Calculer le IoU par catégorie
    ious = []
    for cat in cats:
        gt_cat = gt_mtx == cat
        pred_cat = pred_mtx == cat
        # Les matrices creuses n'acceptent pas les opérateurs bit à bit, mais les matrices _cat
        # sont booléennes : produit et somme donnent seulement Vrai/Faux
        intersection = gt_cat * pred_cat
        union = gt_cat + pred_cat
        iou = intersection.sum() / union.sum()
        ious.append(iou)

    return ious


def main(
    user_annotations_path: Path,
    target_annotations_path: Path,
):
    """
    Calcule le IoU caractère moyen macro par classe sur un jeu d'annotations.
    """
    user_annotations = pd.read_csv(user_annotations_path)
    target_annotations = pd.read_csv(target_annotations_path)
    ious = iou_per_class(user_annotations, target_annotations)
    print(f"macro-averaged character IoU metric: {np.mean(ious):0.4f}.")


if __name__ == "__main__":
    typer.run(main)
