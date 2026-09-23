"""Exercise the adapter's actual preprocessing without importing lm-eval."""

import ast
import unittest
from pathlib import Path
from unittest.mock import MagicMock

import torch


class GenerationReached(Exception):
    pass


class PromptTruncationTest(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        path = Path(__file__).resolve().parents[1] / "eval.py"
        tree = ast.parse(path.read_text())
        dream = next(node for node in tree.body
                     if isinstance(node, ast.ClassDef) and node.name == "Dream")
        method = next(node for node in dream.body
                      if isinstance(node, ast.FunctionDef)
                      and node.name == "_generate_batch")
        namespace = {"List": list, "eval_logger": MagicMock()}
        exec(compile(ast.Module(body=[method], type_ignores=[]), str(path), "exec"),
             namespace)
        cls.generate_batch = staticmethod(namespace["_generate_batch"])

    def make_adapter(self, ids, budget):
        adapter = MagicMock()
        adapter.if_apply_chat_template = False
        adapter.add_bos_token = False
        adapter.max_length = budget + 256
        adapter.max_new_tokens = 256
        adapter.device = "cpu"
        adapter.tokenizer.return_value.input_ids = ids
        adapter.tokenizer.pad_token_id = 0
        adapter.tokenizer.eos_token_id = 9
        adapter.tokenizer.get_vocab.return_value = {}
        adapter.model.diffusion_generate.side_effect = GenerationReached
        return adapter

    def test_truncates_sequence_dimension_preserving_batch(self):
        ids = torch.arange(1, 3801).reshape(2, 1900)
        adapter = self.make_adapter(ids, 1792)
        with self.assertRaises(GenerationReached):
            self.generate_batch(adapter, ["first", "second"])
        actual = adapter.model.diffusion_generate.call_args.args[0]
        self.assertTrue(torch.equal(actual, ids[:, -1792:]))

    def test_short_prompt_and_padding_mask_are_preserved(self):
        ids = torch.tensor([[0, 0, 4, 5], [0, 6, 7, 8]])
        adapter = self.make_adapter(ids, 8)
        with self.assertRaises(GenerationReached):
            self.generate_batch(adapter, ["first", "second"])
        call = adapter.model.diffusion_generate.call_args
        self.assertTrue(torch.equal(call.args[0], ids))
        self.assertTrue(torch.equal(call.kwargs["attention_mask"], ids != 0))

    def test_no_prompt_budget_fails_before_generation(self):
        adapter = self.make_adapter(torch.ones((1, 4), dtype=torch.long), 0)
        with self.assertRaisesRegex(ValueError, "max_new_tokens"):
            self.generate_batch(adapter, ["prompt"])
        adapter.model.diffusion_generate.assert_not_called()
