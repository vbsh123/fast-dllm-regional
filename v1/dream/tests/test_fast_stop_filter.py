from __future__ import annotations

import sys
import unittest
from pathlib import Path

import torch


MODEL_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(MODEL_ROOT))

from model.generation_utils import fast_stop_filter_eligibility  # noqa: E402


class FastStopFilterEligibilityTest(unittest.TestCase):
    def test_filters_only_guarded_region_while_left_region_is_unfinished(self):
        response_mask = torch.tensor(
            [True, False, False, False, True, True, True, True]
        )
        raw_predictions = torch.tensor([1, 1, 1, 1, 9, 2, 3, 4])
        proposed_predictions = torch.tensor([1, 1, 1, 1, 9, 2, 3, 4])
        proposed_probabilities = torch.tensor(
            [0.95, 0.0, 0.0, 0.0, 0.99, 0.60, 0.80, 0.90]
        )

        eligible, stats = fast_stop_filter_eligibility(
            response_mask,
            raw_predictions,
            proposed_predictions,
            proposed_probabilities,
            stop_token_ids={9},
            region_size=4,
            confidence_threshold=0.7,
        )

        self.assertEqual(
            eligible.tolist(),
            [True, False, False, False, False, False, True, True],
        )
        self.assertEqual(stats["guarded_region"], 1)
        self.assertEqual(stats["predicted_stop_position"], 4)
        self.assertEqual(stats["filtered_stop_candidates"], 1)
        self.assertEqual(stats["filtered_low_confidence_candidates"], 1)

    def test_releases_filter_after_regions_to_the_left_finish(self):
        response_mask = torch.tensor(
            [False, False, False, False, True, True, True, True]
        )
        raw_predictions = torch.tensor([1, 1, 1, 1, 9, 2, 3, 4])
        proposed_predictions = raw_predictions.clone()
        proposed_probabilities = torch.tensor(
            [0.0, 0.0, 0.0, 0.0, 0.99, 0.20, 0.30, 0.40]
        )

        eligible, stats = fast_stop_filter_eligibility(
            response_mask,
            raw_predictions,
            proposed_predictions,
            proposed_probabilities,
            stop_token_ids={9},
            region_size=4,
            confidence_threshold=0.7,
        )

        self.assertEqual(eligible.tolist(), response_mask.tolist())
        self.assertIsNone(stats["guarded_region"])
        self.assertEqual(stats["filtered_stop_candidates"], 0)
        self.assertEqual(stats["filtered_low_confidence_candidates"], 0)

    def test_never_guards_a_stop_predicted_in_the_first_region(self):
        response_mask = torch.ones(8, dtype=torch.bool)
        raw_predictions = torch.tensor([1, 9, 1, 1, 1, 1, 1, 1])
        proposed_predictions = raw_predictions.clone()
        proposed_probabilities = torch.full((8,), 0.1)

        eligible, stats = fast_stop_filter_eligibility(
            response_mask,
            raw_predictions,
            proposed_predictions,
            proposed_probabilities,
            stop_token_ids={9},
            region_size=4,
            confidence_threshold=0.7,
        )

        self.assertEqual(eligible.tolist(), response_mask.tolist())
        self.assertIsNone(stats["guarded_region"])


if __name__ == "__main__":
    unittest.main()
