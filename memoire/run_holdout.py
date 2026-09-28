from __future__ import annotations

"""Lance la comparaison hold-out dictionnaire × SnoBERT × oracle."""
import argparse
from pathlib import Path

import pandas as pd

from .dictionary import (
    build_most_common_concept,
    load_concept_dict,
    save_concept_dict,
)
from .kiri_full import get_or_build_kiri_full, predict_kiri_full
from .oracle import (
    attach_system_hits,
    complementarity_delta,
    oracle_dominates,
    oracle_submission,
)
from .paths import (
    DICT_CACHE,
    KIRI_FULL_CACHE,
    KIRI_PRED,
    OUTPUT_DIR,
    SNOBERT_FULL,
    SNOBERT_NODICT,
    TABLES_MD,
    TEST_ANN,
    TEST_NOTES,
    TRAIN_ANN,
    TRAIN_NOTES,
)
from .regimes import add_regimes, regime_table
from .score import mean_char_iou, mention_recall
from .snobert_variants import snobert_neural_residual


def _load_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def _kiri_predictions(
    notes: pd.DataFrame,
    train_notes: pd.DataFrame,
    train_ann: pd.DataFrame,
    test_ann: pd.DataFrame,
    force: bool = False,
) -> pd.DataFrame:
    if KIRI_PRED.exists() and KIRI_FULL_CACHE.exists() and not force:
        return pd.read_csv(KIRI_PRED)
    d, uc_d = get_or_build_kiri_full(
        train_notes=train_notes,
        train_ann=train_ann,
        test_ann=test_ann,
        force=force,
    )
    pred = predict_kiri_full(notes=notes, d=d, uc_d=uc_d)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    pred.to_csv(KIRI_PRED, index=False)
    return pred


def _term_dict(train_notes: pd.DataFrame, train_ann: pd.DataFrame) -> dict[str, int]:
    if DICT_CACHE.exists():
        return load_concept_dict(DICT_CACHE)
    mapping = build_most_common_concept(notes=train_notes, annotations=train_ann)
    save_concept_dict(DICT_CACHE, mapping)
    return mapping


def _snobert_nodict(
    snobert_full: pd.DataFrame,
    notes: pd.DataFrame,
    term_to_cid: dict[str, int],
) -> pd.DataFrame:
    if SNOBERT_NODICT.exists():
        return pd.read_csv(SNOBERT_NODICT)
    pred = snobert_neural_residual(
        snobert_full=snobert_full, notes=notes, term_to_cid=term_to_cid
    )
    pred.to_csv(SNOBERT_NODICT, index=False)
    return pred


def _pair_metrics(
    gold: pd.DataFrame,
    dict_pred: pd.DataFrame,
    sno_pred: pd.DataFrame,
    align_path: Path,
    oracle_path: Path,
) -> tuple[dict[str, float], pd.DataFrame]:
    aligned = add_regimes(
        attach_system_hits(gold=gold, dict_pred=dict_pred, sno_pred=sno_pred)
    )
    aligned.to_csv(align_path, index=False)
    oracle_pred = oracle_submission(aligned)
    oracle_pred.to_csv(oracle_path, index=False)

    miou_d = mean_char_iou(pred=dict_pred, gold=gold)
    miou_s = mean_char_iou(pred=sno_pred, gold=gold)
    miou_o = mean_char_iou(pred=oracle_pred, gold=gold)
    rec_d = mention_recall(gold=gold, pred=dict_pred)
    rec_s = mention_recall(gold=gold, pred=sno_pred)
    rec_o = float(aligned["oracle_hit"].mean())
    if not oracle_dominates(dict_score=miou_d, sno_score=miou_s, oracle_score=miou_o):
        raise RuntimeError(f"oracle mIoU {miou_o} < max({miou_d}, {miou_s})")
    if not oracle_dominates(dict_score=rec_d, sno_score=rec_s, oracle_score=rec_o):
        raise RuntimeError(f"oracle recall {rec_o} < max({rec_d}, {rec_s})")
    scores = {
        "miou_kiri": miou_d,
        "miou_snobert": miou_s,
        "miou_oracle": miou_o,
        "recall_kiri": rec_d,
        "recall_snobert": rec_s,
        "recall_oracle": rec_o,
        "delta_miou": complementarity_delta(
            dict_score=miou_d, sno_score=miou_s, oracle_score=miou_o
        ),
        "delta_recall": complementarity_delta(
            dict_score=rec_d, sno_score=rec_s, oracle_score=rec_o
        ),
    }
    return scores, regime_table(aligned)


def _section_md(title: str, sno_label: str, scores: dict[str, float], regimes: pd.DataFrame) -> str:
    lines = [
        f"## {title}",
        "",
        f"Dictionnaire = KIRIs (train cutmed + OMOP/abbr, sans mentions hold-out). Neural = {sno_label}.",
        "",
        "| Système | mIoU caractère | Rappel mention |",
        "|---|---:|---:|",
        f"| KIRIs | {scores['miou_kiri']:.4f} | {scores['recall_kiri']:.4f} |",
        f"| {sno_label} | {scores['miou_snobert']:.4f} | {scores['recall_snobert']:.4f} |",
        f"| Oracle | {scores['miou_oracle']:.4f} | {scores['recall_oracle']:.4f} |",
        f"| Δ vs meilleur des deux | {scores['delta_miou']:.4f} | {scores['delta_recall']:.4f} |",
        "",
        "| Régime | n | % | P(erreur) | KIRIs juste / S faux | S juste / KIRIs faux | Les deux justes |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in regimes.itertuples(index=False):
        lines.append(
            f"| {row.regime} | {int(row.n)} | {100 * row.pct:.1f} | {row.p_error:.3f} | "
            f"{int(row.d_only_correct)} | {int(row.s_only_correct)} | {int(row.both_correct)} |"
        )
    lines.append("")
    return "\n".join(lines)


def run(pair: str = "both", force: bool = False) -> dict[str, dict[str, float]]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    notes = _load_csv(TEST_NOTES)
    gold = _load_csv(TEST_ANN)
    train_notes = _load_csv(TRAIN_NOTES)
    train_ann = _load_csv(TRAIN_ANN)
    sno_full = _load_csv(SNOBERT_FULL)
    kiri = _kiri_predictions(
        notes=notes,
        train_notes=train_notes,
        train_ann=train_ann,
        test_ann=gold,
        force=force,
    )

    results: dict[str, dict[str, float]] = {}
    sections: list[str] = [
        "# Tables hold-out — KIRIs × SnoBERT",
        "",
        "L'oracle voit le gold : plafond, pas une méthode.",
        "KIRIs ici = dictmethod sur le train cutmed, enrichi OMOP/abréviations, sans gold hold-out.",
        "",
    ]

    if pair in {"A", "both"}:
        scores_a, regimes_a = _pair_metrics(
            gold=gold,
            dict_pred=kiri,
            sno_pred=sno_full,
            align_path=OUTPUT_DIR / "align_A.csv",
            oracle_path=OUTPUT_DIR / "oracle_A.csv",
        )
        results["A"] = scores_a
        sections.append(
            _section_md(
                title="Option A — KIRIs vs SnoBERT complet (2 étages + static dict)",
                sno_label="SnoBERT complet",
                scores=scores_a,
                regimes=regimes_a,
            )
        )

    if pair in {"B", "both"}:
        term_to_cid = _term_dict(train_notes=train_notes, train_ann=train_ann)
        sno_b = _snobert_nodict(
            snobert_full=sno_full, notes=notes, term_to_cid=term_to_cid
        )
        scores_b, regimes_b = _pair_metrics(
            gold=gold,
            dict_pred=kiri,
            sno_pred=sno_b,
            align_path=OUTPUT_DIR / "align_B.csv",
            oracle_path=OUTPUT_DIR / "oracle_B.csv",
        )
        results["B"] = scores_b
        sections.append(
            _section_md(
                title="Option B — KIRIs vs résidu neural SnoBERT",
                sno_label="SnoBERT hors vocabulaire du static dict",
                scores=scores_b,
                regimes=regimes_b,
            )
        )
        sections.append(
            "Option B : pas de poids NER/SapBERT sur disque, donc pas de vrai "
            "rerun sans dict. On garde les spans SnoBERT dont la surface n'est "
            "pas dans le static dict train (longue traîne). Ce n'est pas le "
            "pipeline 2 étages entier sans `choose_concepts`."
        )

    TABLES_MD.write_text("\n".join(sections) + "\n", encoding="utf-8")
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--pair",
        choices=["A", "B", "both"],
        default="both",
        help="A: KIRIs vs SnoBERT full. B: KIRIs vs SnoBERT without added dict spans.",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Retrain the full KIRIs dict on cutmed and re-infer the hold-out.",
    )
    args = parser.parse_args()
    all_scores = run(pair=args.pair, force=args.force)
    for name, scores in all_scores.items():
        print(f"=== option {name} ===")
        for key, value in scores.items():
            print(f"{key}={value:.4f}")
