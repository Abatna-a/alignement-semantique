from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memoire.align import best_alignment, is_mention_hit, span_iou
from memoire.dictionary import predict_with_dict
from memoire.kiri import build_kiri_core, predict_kiri
from memoire.regimes import add_regimes, mention_regime
from memoire.oracle import attach_system_hits
from memoire.snobert_variants import snobert_neural_residual


class AlignmentTests(unittest.TestCase):
    def test_prefers_higher_span_iou(self) -> None:
        gold = pd.DataFrame(
            [{"note_id": "n1", "start": 0, "end": 10, "concept_id": 1}]
        )
        pred = pd.DataFrame(
            [
                {"note_id": "n1", "start": 0, "end": 4, "concept_id": 9},
                {"note_id": "n1", "start": 0, "end": 10, "concept_id": 1},
            ]
        )
        aligned = best_alignment(gold=gold, pred=pred)
        self.assertEqual(int(aligned.loc[0, "pred_concept_id"]), 1)
        self.assertEqual(float(aligned.loc[0, "span_iou"]), 1.0)
        self.assertTrue(is_mention_hit(aligned.loc[0]))

    def test_below_threshold_is_a_miss(self) -> None:
        gold = pd.DataFrame(
            [{"note_id": "n1", "start": 0, "end": 10, "concept_id": 1}]
        )
        pred = pd.DataFrame(
            [{"note_id": "n1", "start": 0, "end": 3, "concept_id": 1}]
        )
        iou = span_iou(start_a=0, end_a=10, start_b=0, end_b=3)
        self.assertLess(iou, 0.5)
        aligned = best_alignment(gold=gold, pred=pred, min_span_iou=0.5)
        self.assertTrue(pd.isna(aligned.loc[0, "pred_concept_id"]))
        self.assertFalse(is_mention_hit(aligned.loc[0]))


class DictionaryAndRegimeTests(unittest.TestCase):
    def test_ngram_dict_finds_multiword_mention(self) -> None:
        notes = pd.DataFrame(
            [{"note_id": "n1", "text": "Patient with chest pain yesterday."}]
        )
        pred = predict_with_dict(
            notes=notes,
            most_common_concept={"chest pain": 279084009, "pain": 22253000},
        )
        long = pred[pred["concept_id"] == 279084009]
        self.assertEqual(len(long), 1)
        self.assertEqual(int(long.iloc[0]["end"] - long.iloc[0]["start"]), len("chest pain"))

    def test_four_regimes(self) -> None:
        gold = pd.DataFrame(
            [
                {"note_id": "n1", "start": 0, "end": 4, "concept_id": 1},
                {"note_id": "n1", "start": 10, "end": 14, "concept_id": 2},
                {"note_id": "n1", "start": 20, "end": 24, "concept_id": 3},
                {"note_id": "n1", "start": 30, "end": 34, "concept_id": 4},
            ]
        )
        dictionary = pd.DataFrame(
            [
                {"note_id": "n1", "start": 0, "end": 4, "concept_id": 1},
                {"note_id": "n1", "start": 10, "end": 14, "concept_id": 99},
                {"note_id": "n1", "start": 20, "end": 24, "concept_id": 3},
            ]
        )
        snobert = pd.DataFrame(
            [
                {"note_id": "n1", "start": 0, "end": 4, "concept_id": 1},
                {"note_id": "n1", "start": 10, "end": 14, "concept_id": 2},
                {"note_id": "n1", "start": 30, "end": 34, "concept_id": 4},
            ]
        )
        aligned = add_regimes(
            attach_system_hits(gold=gold, dict_pred=dictionary, sno_pred=snobert)
        )
        regimes = [mention_regime(row) for _, row in aligned.iterrows()]
        self.assertEqual(regimes, ["ACCORD", "CONFLIT", "D_SEUL", "S_SEUL"])


class KiriAndNodictTests(unittest.TestCase):
    def test_section_key_beats_any(self) -> None:
        pad = "x" * 100
        body = "Past Medical History:\n\nhypotension was recorded.\n"
        notes = pd.DataFrame([{"note_id": "n1", "text": pad + body}])
        start = (pad + body).lower().index("hypotension")
        gold = pd.DataFrame(
            [
                {
                    "note_id": "n1",
                    "start": start,
                    "end": start + len("hypotension"),
                    "concept_id": 45007003,
                }
            ]
        )
        mapping = build_kiri_core(notes=notes, annotations=gold)
        pred = predict_kiri(notes=notes, mapping=mapping, skip_prefix=0)
        self.assertGreaterEqual(len(pred), 1)
        self.assertEqual(int(pred.iloc[0]["concept_id"]), 45007003)

    def test_neural_residual_keeps_oov_mentions(self) -> None:
        notes = pd.DataFrame([{"note_id": "n1", "text": "aaaa bbbbbb"}])
        full = pd.DataFrame(
            [
                {"note_id": "n1", "start": 0, "end": 4, "concept_id": 1},
                {"note_id": "n1", "start": 5, "end": 11, "concept_id": 2},
            ]
        )
        residual = snobert_neural_residual(
            snobert_full=full,
            notes=notes,
            term_to_cid={"aaaa": 1},
        )
        self.assertEqual(len(residual), 1)
        self.assertEqual(int(residual.iloc[0]["start"]), 5)


if __name__ == "__main__":
    unittest.main()
