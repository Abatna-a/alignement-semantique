"""Enchaîne spans, liaison SapBERT et dictionnaire statique."""
import argparse
from pathlib import Path

import pandas as pd
from first_stage import fisrt_stage
from first_stage_postprocess import clean_by_header
from iou import iou_per_class
from loguru import logger
from second_stage import second_stage
from second_stage_postprocess import choose_concepts
from static_dict import add_static_dict


def cleanup(df):
    df = df[df.start < df.end]
    df = df[["note_id", "start", "end", "concept_id"]]
    return df


def main(
    test_notes_path: Path,
    first_stage_checkpoints: list[Path],
    second_stage_checkpoints: list[Path],
    train_notes_path: Path,
    train_annotations_path: Path,
    static_dict_path: Path,
    submission_path: Path,
):
    assert test_notes_path.exists(), f"test_notes_path: {test_notes_path} does not exist"
    assert all([i.exists() for i in first_stage_checkpoints]), [
        i for i in first_stage_checkpoints if not i.exists()
    ]
    assert all([i.exists() for i in second_stage_checkpoints]), [
        i for i in second_stage_checkpoints if not i.exists()
    ]
    assert train_notes_path.exists(), f"train_notes_path: {train_notes_path} does not exist"
    assert (
        train_annotations_path.exists()
    ), f"train_annotations_path: {train_annotations_path} does not exist"
    assert static_dict_path.exists(), f"static_dict_path: {static_dict_path} does not exist"

    note_df = pd.read_csv(test_notes_path)
    logger.debug(f"{note_df.shape=}")

    # Premier étage
    mentions_df = fisrt_stage(first_stage_checkpoints, note_df)
    logger.debug(f"{mentions_df.shape=}")

    # Post-traitement du premier étage
    mentions_df = clean_by_header(mentions_df, note_df)  # will add headers to annotations
    logger.debug(f"{mentions_df.shape=}")
    logger.debug(f"{mentions_df.head()}")

    # Second étage
    sap_checkpoint = second_stage_checkpoints[0]
    tdf = second_stage(mentions_df, sap_checkpoint, topk=1)
    logger.debug(f"{tdf.shape=}")
    logger.debug(f"{tdf.head()}")

    # Post-traitement du second étage
    tdf = choose_concepts(tdf, note_df, train_notes_path, train_annotations_path)
    logger.debug(f"{tdf.shape=}")
    logger.debug(f"{tdf.head()}")

    # Dictionnaire statique
    tdf = add_static_dict(tdf, static_dict_path, note_df)
    logger.debug(f"{tdf.shape=}")

    # Nettoyage et enregistrement
    tdf = cleanup(tdf)
    logger.debug(f"{tdf.shape=}")
    tdf.to_csv(submission_path, index=False)
    logger.info(f"Saved submission to {submission_path=}")


if __name__ == "__main__":
    args = argparse.ArgumentParser()
    args.add_argument("--val", action="store_true")
    args.add_argument(
        "--mixup",
        action="store_true",
        help="Use only first_stage checkpoints whose name contains 'mixup' "
        "(otherwise use only non-mixup / baseline checkpoints).",
    )
    args.add_argument(
        "--score-test",
        action="store_true",
        help="After inference, score vs data/competition_data/test_annotations.csv",
    )
    args.add_argument(
        "--first-stage-dir",
        type=Path,
        default=None,
        help="Checkpoint folder (default: data/first_stage). "
        "Use e.g. data107/first_stage_tau0.1 for a seed/tau cell.",
    )
    args = args.parse_args()
    ASSETS = Path("data")
    if args.val:
        SPLIT_PATH = ASSETS / "preprocess_data" / "splits"
        TRAIN_NOTES_PATH = ASSETS / "preprocess_data" / "splits" / "train_note_split_0.csv"
        TRAIN_ANNOTAIONS_PATH = ASSETS / "preprocess_data" / "splits" / "train_ann_split_0.csv"
        TEST_NOTES_PATH = ASSETS / "preprocess_data" / "splits" / "val_note_split_0.csv"
        TEST_ANNOTAIONS_PATH = ASSETS / "preprocess_data" / "splits" / "val_ann_split_0.csv"
        SUBMISSION_PATH = ASSETS / "submission.csv"
        STATIC_DICT_PATH = ASSETS / "preprocess_data" / "most_common_concept_val_0.pkl"
    else:
        TRAIN_NOTES_PATH = ASSETS / "competition_data" / "cutmed_notes.csv"
        TRAIN_ANNOTAIONS_PATH = ASSETS / "competition_data" / "cutmed_fixed_train_annotations.csv"
        TEST_NOTES_PATH = ASSETS / "competition_data" / "test_notes.csv"
        TEST_ANNOTAIONS_PATH = ASSETS / "competition_data" / "test_annotations.csv"
        SUBMISSION_PATH = Path("submission_mixup.csv" if args.mixup else "submission_baseline.csv")
        STATIC_DICT_PATH = ASSETS / "preprocess_data" / "most_common_concept.pkl"

    ckpt_root = args.first_stage_dir if args.first_stage_dir is not None else (ASSETS / "first_stage")
    all_ckpts = [p for p in ckpt_root.iterdir() if p.is_dir()]
    if args.mixup:
        FIRST_STAGE_CHECKPOINTS = [p for p in all_ckpts if "mixup" in p.name.lower()]
        if not FIRST_STAGE_CHECKPOINTS:
            # Les dossiers seed 107/123 sont mixup-only mais sans "mixup" dans le nom
            FIRST_STAGE_CHECKPOINTS = list(all_ckpts)
    else:
        FIRST_STAGE_CHECKPOINTS = [p for p in all_ckpts if "mixup" not in p.name.lower()]
    assert FIRST_STAGE_CHECKPOINTS, (
        f"No first_stage checkpoints for mode={'mixup' if args.mixup else 'baseline'} "
        f"in {ckpt_root}"
    )
    logger.info(
        f"mode={'mixup' if args.mixup else 'baseline'} "
        f"dir={ckpt_root} "
        f"checkpoints={[p.name for p in FIRST_STAGE_CHECKPOINTS]}"
    )

    SECOND_STAGE_CHECKPOINTS = [
        ASSETS / "second_stage" / "sapbert",
    ]

    main(
        TEST_NOTES_PATH,
        FIRST_STAGE_CHECKPOINTS,
        SECOND_STAGE_CHECKPOINTS,
        TRAIN_NOTES_PATH,
        TRAIN_ANNOTAIONS_PATH,
        STATIC_DICT_PATH,
        SUBMISSION_PATH,
    )
    if args.val:
        label = pd.read_csv(SUBMISSION_PATH)
        target = pd.read_csv(TEST_ANNOTAIONS_PATH)
        iou_score = iou_per_class(label, target, mean=True)
        logger.info(f"IoU: {iou_score}")
    elif args.score_test:
        label = pd.read_csv(SUBMISSION_PATH)
        target = pd.read_csv(TEST_ANNOTAIONS_PATH)
        iou_score = iou_per_class(label, target, mean=True)
        logger.info(f"IoU (hold-out test): {iou_score}")
