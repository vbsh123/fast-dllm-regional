from __future__ import annotations

import sys
import unittest
from pathlib import Path


SCRIPT_ROOT = Path(__file__).resolve().parents[1] / "scripts"
sys.path.insert(0, str(SCRIPT_ROOT))

from summarize_regional_mechanism import summarize  # noqa: E402


class RegionalMechanismSummaryTest(unittest.TestCase):
    def test_summary_uses_event_weighted_rates(self):
        row = {
            "nfe": 10,
            "region_size": 32,
            "local_steps": 32,
            "max_progress_gap": 4,
            "deferral_threshold": 0.4,
            "deferral_until_revealed": 2,
            "max_region_deferrals": 4,
            "stop_mode": "filter",
            "startup_mechanism": {
                "candidate_update_events": 8,
                "deferred_update_events": 4,
                "maximum_deferral_events": 8,
                "committed_update_events": 4,
                "confidence_pass_commit_events": 2,
                "gap_forced_low_confidence_commit_events": 1,
                "region_limit_forced_low_confidence_commit_events": 0,
                "global_deadlock_forced_low_confidence_commit_events": 1,
                "bootstrap_tokens_committed": 4,
                "per_region": [
                    {
                        "region_index": 0,
                        "candidate_update_events": 8,
                        "deferred_update_events": 4,
                        "confidence_pass_commit_events": 2,
                        "gap_forced_commit_events": 1,
                        "region_limit_forced_commit_events": 0,
                        "global_deadlock_forced_commit_events": 1,
                        "bootstrap_tokens_committed": 4,
                        "first_commit_nfe": 2,
                        "startup_complete_nfe": 5,
                    }
                ],
            },
            "concurrency_mechanism": {
                "committing_region_events": 20,
                "forwards_with_no_scheduler_commit": 1,
                "forwards_with_multiple_committing_regions": 6,
            },
            "commit_confidence": {
                "all_committed_token_top1_probability": {
                    "count": 10,
                    "mean": 0.6,
                },
                "startup_committed_token_top1_probability": {
                    "count": 4,
                    "mean": 0.5,
                },
                "committed_tokens_below_deferral_threshold": 2,
                "committed_tokens_at_least_fast_reference_0_9": 3,
            },
            "progress_balance": {"maximum_absolute_adjacent_gap": 4},
        }

        result = summarize([row])

        self.assertEqual(result["examples"], 1)
        self.assertEqual(result["startup"]["deferral_rate"], 0.5)
        self.assertEqual(result["startup"]["maximum_deferral_events"], 8)
        self.assertEqual(
            result["startup"]["forced_rate_per_committed_update"], 0.5
        )
        self.assertEqual(
            result["concurrency"]["mean_committing_regions_per_forward"],
            2.0,
        )
        self.assertEqual(
            result["confidence"][
                "fraction_committed_at_least_fast_reference_0_9"
            ],
            0.3,
        )


if __name__ == "__main__":
    unittest.main()
