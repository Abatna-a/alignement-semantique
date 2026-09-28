from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memoire.align import span_iou
from memoire.oracle import (
    attach_system_hits,
    complementarity_delta,
    oracle_dominates,
    oracle_submission,
)
from memoire.score import mean_char_iou, mention_recall


def _spans(rows: list[tuple]) -> pd.DataFrame:
    return pd.DataFrame(rows, columns=["note_id", "start", "end", "concept_id"])


class SpanIouTests(unittest.TestCase):
    def test_identical_spans(self) -> None:
        self.assertEqual(
            span_iou(start_a=0, end_a=10, start_b=0, end_b=10),
            1.0,
        )

    def test_no_overlap(self) -> None:
        self.assertEqual(
            span_iou(start_a=0, end_a=4, start_b=10, end_b=14),
            0.0,
        )


class OracleSuperiorityTests(unittest.TestCase):
    """Preuves synthétiques : oracle ≥ max(D, S), et strictement meilleur en cas de complémentarité."""

    def setUp(self) -> None:
        self.gold = _spans(
            [
                ("n1", 0, 5, 111),
                ("n1", 10, 16, 222),
                ("n1", 20, 25, 333),
            ]
        )

    def test_complementarity_oracle_strictly_better_on_mention_recall(self) -> None:
        dictionary = _spans([("n1", 0, 5, 111), ("n1", 20, 25, 999)])
        snobert = _spans([("n1", 10, 16, 222)])
        aligned = attach_system_hits(
            gold=self.gold, dict_pred=dictionary, sno_pred=snobert
        )
        rec_d = mention_recall(gold=self.gold, pred=dictionary)
        rec_s = mention_recall(gold=self.gold, pred=snobert)
        rec_o = float(aligned["oracle_hit"].mean())
        self.assertTrue(
            oracle_dominates(dict_score=rec_d, sno_score=rec_s, oracle_score=rec_o)
        )
        self.assertGreater(
            complementarity_delta(dict_score=rec_d, sno_score=rec_s, oracle_score=rec_o),
            0.0,
        )
        self.assertAlmostEqual(rec_d, 1.0 / 3.0)
        self.assertAlmostEqual(rec_s, 1.0 / 3.0)
        self.assertAlmostEqual(rec_o, 2.0 / 3.0)

    def test_complementarity_oracle_strictly_better_on_character_miou(self) -> None:
        dictionary = _spans([("n1", 0, 5, 111)])
        snobert = _spans([("n1", 10, 16, 222)])
        aligned = attach_system_hits(
            gold=self.gold, dict_pred=dictionary, sno_pred=snobert
        )
        oracle_pred = oracle_submission(aligned)
        miou_d = mean_char_iou(pred=dictionary, gold=self.gold)
        miou_s = mean_char_iou(pred=snobert, gold=self.gold)
        miou_o = mean_char_iou(pred=oracle_pred, gold=self.gold)
        self.assertTrue(
            oracle_dominates(dict_score=miou_d, sno_score=miou_s, oracle_score=miou_o)
        )
        self.assertGreater(miou_o, miou_d)
        self.assertGreater(miou_o, miou_s)

    def test_identical_systems_oracle_not_worse(self) -> None:
        pred = _spans([("n1", 0, 5, 111), ("n1", 10, 16, 222)])
        aligned = attach_system_hits(gold=self.gold, dict_pred=pred, sno_pred=pred)
        rec = mention_recall(gold=self.gold, pred=pred)
        rec_o = float(aligned["oracle_hit"].mean())
        self.assertAlmostEqual(rec_o, rec)
        self.assertTrue(oracle_dominates(dict_score=rec, sno_score=rec, oracle_score=rec_o))

    def test_both_wrong_oracle_cannot_invent_a_code(self) -> None:
        dictionary = _spans([("n1", 0, 5, 999)])
        snobert = _spans([("n1", 0, 5, 999)])
        aligned = attach_system_hits(
            gold=self.gold, dict_pred=dictionary, sno_pred=snobert
        )
        self.assertFalse(bool(aligned.loc[0, "oracle_hit"]))
        self.assertTrue(oracle_submission(aligned).empty)


if __name__ == "__main__":
    unittest.main()
