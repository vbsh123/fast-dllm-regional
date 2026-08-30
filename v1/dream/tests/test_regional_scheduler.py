from __future__ import annotations

import sys
import unittest
from pathlib import Path


DREAM_MODEL_ROOT = Path(__file__).resolve().parents[1] / "model"
sys.path.insert(0, str(DREAM_MODEL_ROOT))

from regional_scheduler import (  # noqa: E402
    build_regions,
    controlled_regions,
    linear_transfer_count,
)


class RegionalSchedulerTest(unittest.TestCase):
    def test_build_regions_keeps_partial_tail(self):
        regions = build_regions(100, 32)
        self.assertEqual(
            [(region.start, region.end) for region in regions],
            [(0, 32), (32, 64), (64, 96), (96, 100)],
        )

    def test_parent_at_gap_is_blocked_and_child_is_urgent(self):
        regions = build_regions(16, 8)
        active, blocked, urgent = controlled_regions(
            regions,
            remaining_masks=[4, 8],
            local_steps=8,
            max_progress_gap=4,
        )
        self.assertEqual(active, [1])
        self.assertEqual(blocked, {0})
        self.assertEqual(urgent, {1})

    def test_child_cannot_lead_parent(self):
        regions = build_regions(16, 8)
        active, blocked, urgent = controlled_regions(
            regions,
            remaining_masks=[8, 7],
            local_steps=8,
            max_progress_gap=4,
        )
        self.assertEqual(active, [0])
        self.assertEqual(blocked, {1})
        self.assertEqual(urgent, {0})

    def test_stalled_stop_child_temporarily_exempts_parent_gap(self):
        regions = build_regions(16, 8)
        active, blocked, urgent = controlled_regions(
            regions,
            remaining_masks=[4, 8],
            local_steps=8,
            max_progress_gap=4,
            progress_gap_exempt_children={1},
        )
        self.assertEqual(active, [0, 1])
        self.assertEqual(blocked, set())
        self.assertEqual(urgent, set())

    def test_zero_quota_does_not_imply_revealed_progress(self):
        regions = build_regions(32, 32)
        count = linear_transfer_count(
            32, schedule_step=0, local_steps=32, eps=1e-3
        )
        self.assertEqual(count, 0)
        self.assertEqual(regions[0].size - 32, 0)

    def test_last_local_step_commits_every_remaining_mask(self):
        self.assertEqual(
            linear_transfer_count(
                11, schedule_step=31, local_steps=32, eps=1e-3
            ),
            11,
        )


if __name__ == "__main__":
    unittest.main()
