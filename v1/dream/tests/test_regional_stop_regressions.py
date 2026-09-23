"""CPU full-loop regressions; no checkpoint downloads required."""

import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from model.generation_utils import _regional_sample


class SyntheticModel:
    def __init__(self, mode):
        self.mode = mode

    def __call__(self, x, *args):
        probabilities = torch.full((1, x.shape[1], 12), 1e-30)
        if self.mode in {"sampled_stop", "first_region_stop"}:
            # EOS is never raw top-1, but can survive top-p and be sampled.
            probabilities[:, :, 1] = 0.55
            probabilities[:, :, 9] = 0.45
            if self.mode == "first_region_stop":
                probabilities[0, 1, 1] = 0.05
                probabilities[0, 1, 9] = 0.95
        elif self.mode == "early_stop":
            probabilities.fill_(0.79 / 9)
            probabilities[:, :, 0] = 1e-30
            probabilities[:, :, 1] = 0.20
            probabilities[:, :, 9] = 0.01
            # Response position 1 is an easy EOS, position 0 is uncertain.
            probabilities[0, 2, :] = 0.05 / 10
            probabilities[0, 2, 0] = 1e-30
            probabilities[0, 2, 9] = 0.95
        else:
            probabilities[:, :, 1] = 1.0
        aligned = probabilities.log()
        # Undo the decoder's one-position logit alignment.
        return SimpleNamespace(
            logits=torch.cat([aligned[:, 1:], aligned[:, -1:]], dim=1)
        )


class RegionalStopRegressionTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.old_threads = torch.get_num_threads()
        torch.set_num_threads(1)

    @classmethod
    def tearDownClass(cls):
        torch.set_num_threads(cls.old_threads)

    def run_decoder(self, model_mode, stop_mode, length=64, **overrides):
        x = torch.zeros((1, length + 1), dtype=torch.long)
        x[0, 0] = 11
        histories = []
        options = dict(
            attention_mask="full", tok_idx=None, prompt_length=1,
            mask_token_id=0, eos_token_ids=[9], eps=0.001,
            temperature=0, top_p=None, top_k=None, alg_temp=0,
            commit_policy="entropy", region_size=32, local_steps=32,
            max_progress_gap=4, deferral_threshold=0.4,
            deferral_until_revealed=2, max_region_deferrals=1000000,
            max_global_deferrals=4, stop_mode=stop_mode,
            stop_filter_threshold=0.7,
            generation_tokens_hook_func=lambda step, x, logits: x,
            generation_logits_hook_func=lambda step, x, logits: logits,
            histories=histories, started_at=time.perf_counter(),
        )
        options.update(overrides)
        output, stats = _regional_sample(SyntheticModel(model_mode), x, **options)
        return output, stats, histories

    def test_ignored_suffix_does_not_close_startup_window(self):
        for mode in ("filter", "defer"):
            with self.subTest(mode=mode):
                output, stats, _ = self.run_decoder("early_stop", mode)
                self.assertEqual(stats["accepted_stop_position"], 1)
                self.assertEqual(output[0, :3].tolist(), [11, 1, 9])
                self.assertTrue(bool((output[0, 3:] == 0).all()))
                self.assertEqual(stats["post_startup_commit_events"], 0)
                startup = stats["startup_mechanism"]
                self.assertEqual(startup["bootstrap_tokens_committed"], 2)
                self.assertTrue(any(
                    attempt["revealed_before"] == 1
                    and attempt["decision"] == "deferred_low_confidence"
                    for attempt in startup["attempts"]
                ))

    def test_sampled_stop_waits_for_preceding_regions(self):
        self.check_stop_order("sampled_stop")

    def test_first_region_stop_does_not_hide_later_sampled_stop(self):
        self.check_stop_order("first_region_stop")

    def check_stop_order(self, model_mode):
        for mode in ("filter", "defer", "tail_guard"):
            with self.subTest(mode=mode), torch.random.fork_rng():
                torch.manual_seed(0)
                output, stats, histories = self.run_decoder(
                    model_mode, mode, temperature=0.1, top_p=0.9,
                )
                self.assertEqual(output[0, 0].item(), 11)
                self.assertGreater(stats["tail_guard_iterations"], 0)
                for snapshot in histories:
                    if bool((snapshot[0, 33:] == 9).any()):
                        self.assertFalse(bool((snapshot[0, 1:33] == 0).any()))

    def test_no_stop_completes_canvas_including_partial_region(self):
        for mode in ("none", "filter", "defer", "tail_guard"):
            with self.subTest(mode=mode):
                output, stats, _ = self.run_decoder("no_stop", mode, length=70)
                self.assertEqual(output.shape, (1, 71))
                self.assertEqual(output[0, 0].item(), 11)
                self.assertTrue(bool((output[0, 1:] == 1).all()))
                self.assertEqual(stats["tokens_committed"], 70)
                self.assertIsNone(stats["accepted_stop_position"])


if __name__ == "__main__":
    unittest.main()
