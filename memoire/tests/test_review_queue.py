from __future__ import annotations

import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from memoire.review_queue import queue_metrics


class ReviewQueueTests(unittest.TestCase):
    def test_queue_catches_disagreement_errors_not_accord(self) -> None:
        aligned = pd.DataFrame(
            [
                {"regime": "ACCORD", "oracle_hit": True},
                {"regime": "ACCORD", "oracle_hit": False},
                {"regime": "CONFLIT", "oracle_hit": False},
                {"regime": "S_SEUL", "oracle_hit": True},
                {"regime": "AUCUN", "oracle_hit": False},
            ]
        )
        m = queue_metrics(aligned)
        self.assertEqual(m["n_queue"], 3.0)
        self.assertAlmostEqual(m["error_recall"], 2.0 / 3.0)
        self.assertEqual(m["n_errors_in_accord"], 1.0)


if __name__ == "__main__":
    unittest.main()
