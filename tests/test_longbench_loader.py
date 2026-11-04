"""LongBench zip loading, English task coverage, and the hardcoded token cut."""

import json
import zipfile
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from tokenpowerbench.data.loader import (
    LONGBENCH_EN_NOCODE_SUBTASKS,
    LONGBENCH_MAX_PROMPT_TOKENS,
    DatasetLoader,
    read_longbench_zip,
)


class _Tokenizer:
    def encode(self, text, add_special_tokens=False):
        return [0] * len(text.split())


class LongBenchLoaderTests(unittest.TestCase):
    def test_english_task_list_includes_the_added_tasks_and_drops_zh_and_code(self):
        added = {
            "musique", "gov_report", "qmsum", "multi_news", "trec", "triviaqa",
            "samsum", "passage_count", "passage_retrieval_en",
        }
        self.assertTrue(added <= set(LONGBENCH_EN_NOCODE_SUBTASKS))
        for dropped in ("multifieldqa_zh", "dureader", "vcsum", "lsht", "passage_retrieval_zh", "lcc", "repobench-p"):
            self.assertNotIn(dropped, LONGBENCH_EN_NOCODE_SUBTASKS)
        self.assertEqual(LONGBENCH_MAX_PROMPT_TOKENS, 12848)

    def test_zip_reader_keeps_english_tasks_and_builds_context_then_question(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            archive = root / "data.zip"
            with zipfile.ZipFile(archive, "w") as zf:
                zf.writestr("data/musique.jsonl", json.dumps({
                    "context": "A passage about power.", "input": "What is measured?",
                    "language": "en",
                }) + "\n")
                zf.writestr("data/lcc.jsonl", json.dumps({
                    "context": "def energy():", "input": "complete", "language": "python", "dataset": "lcc",
                }) + "\n")
                zf.writestr("data/multifieldqa_zh.jsonl", json.dumps({
                    "context": "中文", "input": "问题", "language": "zh",
                }) + "\n")
            rows = read_longbench_zip(archive, LONGBENCH_EN_NOCODE_SUBTASKS)
            self.assertEqual([row["dataset"] for row in rows], ["musique"])
            loader = DatasetLoader(seed=1)
            loader._longbench_tokenizer = _Tokenizer
            prompts = loader._longbench_rows_to_sampled(rows, n=10, min_w=2, max_w=50)
        self.assertEqual(prompts, ["A passage about power.\n\nWhat is measured?"])

    def test_token_limit_is_applied_after_the_word_window(self):
        short = "one two three four five"
        long = " ".join(["word"] * 20)
        rows = [
            {"context": short, "input": "question here", "language": "en", "dataset": "triviaqa"},
            {"context": long, "input": "question here", "language": "en", "dataset": "gov_report"},
            {"context": "zh passage", "input": "question", "language": "zh", "dataset": "dureader"},
        ]
        loader = DatasetLoader(seed=1)
        loader._longbench_tokenizer = _Tokenizer
        with patch("tokenpowerbench.data.loader.LONGBENCH_MAX_PROMPT_TOKENS", 12):
            prompts = loader._longbench_rows_to_sampled(rows, n=10, min_w=2, max_w=100)
        self.assertEqual(prompts, [f"{short}\n\nquestion here"])

    def test_unreadable_zip_stops_without_a_fallback(self):
        with tempfile.TemporaryDirectory() as directory:
            bad = Path(directory) / "data.zip"
            bad.write_text("not a zip")
            with self.assertRaises(RuntimeError) as caught:
                read_longbench_zip(bad, LONGBENCH_EN_NOCODE_SUBTASKS)
        self.assertIn("no synthetic fallback", str(caught.exception))

    def test_download_failure_stops_without_a_fallback(self):
        with patch("tokenpowerbench.data.loader.urllib.request.urlopen", side_effect=OSError("offline")):
            with self.assertRaises(RuntimeError) as caught:
                from tokenpowerbench.data.loader import download_longbench_zip
                with tempfile.TemporaryDirectory() as directory:
                    download_longbench_zip(directory)
        self.assertIn("no synthetic fallback", str(caught.exception))

    def test_missing_tokenizer_stops_the_longbench_load(self):
        loader = DatasetLoader(seed=1)
        with patch.dict("sys.modules", {"transformers": None}):
            with self.assertRaises(RuntimeError) as caught:
                loader._apply_longbench_token_limit(["one two three"])
        self.assertIn("no synthetic fallback", str(caught.exception))
        self.assertIsInstance(caught.exception.__cause__, ImportError)
