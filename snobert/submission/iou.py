"""Score IoU caractère macro par concept."""
import numpy as np
import pandas as pd
import scipy.sparse as sp


def iou_per_class(
    user_annotations: pd.DataFrame, target_annotations: pd.DataFrame, mean: bool = False
):
    """
    Calculate the IoU metric for each class in a set of annotations.
    """
    user_annotations = user_annotations.copy()
    target_annotations = target_annotations.copy()
    for df in (user_annotations, target_annotations):
        df["start"] = df["start"].astype(int)
        df["end"] = df["end"].astype(int)
        if "concept_id" in df.columns:
            df["concept_id"] = df["concept_id"].astype(np.int64)

    # Correspondance note_id → index dans le tableau
    docs = np.unique(np.concatenate([user_annotations.note_id, target_annotations.note_id]))
    doc_index_mapping = dict(zip(docs, range(len(docs))))

    # Union des catégories dans la référence et les prédictions
    cats = np.unique(np.concatenate([user_annotations.concept_id, target_annotations.concept_id]))

    # Index de caractère maximal dans la référence ou les prédictions
    max_end = int(np.max(np.concatenate([user_annotations.end.to_numpy(), target_annotations.end.to_numpy()])))

    # Remplir les matrices de catégorisation caractère par caractère
    def populate_char_mtx(n_rows, n_cols, annot_df):
        mtx = sp.lil_array((int(n_rows), int(n_cols)), dtype=np.int64)
        for row in annot_df.itertuples():
            doc_index = doc_index_mapping[row.note_id]
            mtx[doc_index, int(row.start) : int(row.end)] = int(row.concept_id)  # noqa: E203
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
        if union.sum() == 0:
            ious.append(0)
            print(f"Category {cat} has no union")
            continue
        iou = intersection.sum() / union.sum()
        ious.append(iou)
    if mean:
        return np.mean(ious)
    return ious
