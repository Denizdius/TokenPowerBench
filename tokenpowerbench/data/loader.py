"""
Dataset loader supporting Alpaca, Dolly 15K, LongBench, and HumanEval.

Online: downloads from Hugging Face Hub (requires network on first run).

Offline: pass a local path via ``dataset_path`` (``--dataset-path`` on the CLI):

  1. Hugging Face ``save_to_disk`` directory (recommended)::

        from datasets import load_dataset
        load_dataset("tatsu-lab/alpaca").save_to_disk("/data/alpaca")

     Then: ``--dataset alpaca --dataset-path /data/alpaca``

  2. JSON / JSONL file with dataset-specific columns (see extractors below).

  3. Plain ``.txt`` file — one prompt per line (any ``--dataset`` name).
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Union

try:
    from datasets import load_dataset as hf_load_dataset
    from datasets import load_from_disk as hf_load_from_disk
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

_SUPPORTED = ("alpaca", "dolly", "longbench", "humaneval")


class DatasetLoader:
    """
    Load and pre-process prompts from standard LLM evaluation datasets.

    Parameters
    ----------
    cache_dir : str, optional
        HuggingFace dataset cache directory (online mode only).
    seed : int
        Random seed for reproducible sampling.
    """

    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42) -> None:
        self.cache_dir = cache_dir
        self.seed = seed
        random.seed(seed)
        if not _HF_AVAILABLE:
            print(
                "Warning: HuggingFace `datasets` library not installed. "
                "Online Hub download and save_to_disk loading are unavailable. "
                "Use --dataset-path with a .txt or .jsonl file, or: pip install datasets"
            )

    def load(
        self,
        dataset: str,
        num_samples: int = 1000,
        min_words: int = 5,
        max_words: int = 100,
        dataset_path: Optional[str] = None,
    ) -> List[str]:
        """
        Load, filter, and sample prompts.

        Parameters
        ----------
        dataset : str
            One of: "alpaca", "dolly", "longbench", "humaneval".
            Selects how rows are turned into prompt strings.
        num_samples : int
            Maximum number of prompts to return.
        min_words, max_words : int
            Filter by prompt length (word count).
        dataset_path : str, optional
            Local file or directory. When set, no Hub download is attempted.
            See module docstring for supported layouts.

        Returns
        -------
        List[str]
            Sampled prompts ready for inference.
        """
        name = dataset.lower().strip()
        loaders = {
            "alpaca": self._alpaca,
            "dolly": self._dolly,
            "longbench": self._longbench,
            "humaneval": self._humaneval,
        }
        if name not in loaders:
            raise ValueError(
                f"Unknown dataset {name!r}. Supported: {_SUPPORTED}"
            )

        if dataset_path:
            local = Path(dataset_path).expanduser().resolve()
            if not local.exists():
                print(f"[DatasetLoader] Error: dataset path does not exist: {local}")
                return []
            if local.is_file() and local.suffix.lower() == ".txt":
                return self._load_txt_file(local, num_samples, min_words, max_words, name)

        return loaders[name](num_samples, min_words, max_words, dataset_path)

    @staticmethod
    def supported_datasets() -> Dict[str, str]:
        return {
            "alpaca": "Stanford Alpaca — 52K instruction-following demos",
            "dolly": "Databricks Dolly 15K — high-quality instruction data",
            "longbench": "LongBench — long-context multi-task benchmark",
            "humaneval": "HumanEval — Python code completion tasks",
        }

    # ------------------------------------------------------------------
    # Local I/O
    # ------------------------------------------------------------------

    def _load_hf_local(self, path: Path) -> Any:
        """Load a Hugging Face dataset from disk (save_to_disk) or JSON/JSONL."""
        if not _HF_AVAILABLE:
            raise RuntimeError(
                "The `datasets` package is required to load this path. "
                "Install with: pip install datasets"
            )

        if path.is_file() and path.suffix.lower() in (".json", ".jsonl"):
            print(f"[DatasetLoader] Loading JSON/JSONL: {path}")
            return hf_load_dataset(
                "json",
                data_files=str(path),
                cache_dir=self.cache_dir,
                split="train",
            )

        if (path / "dataset_dict.json").exists() or (path / "state.json").exists():
            print(f"[DatasetLoader] Loading from disk: {path}")
            return hf_load_from_disk(str(path))

        json_files = sorted(path.glob("*.json")) + sorted(path.glob("*.jsonl"))
        json_files += sorted(path.glob("**/*.json")) + sorted(path.glob("**/*.jsonl"))
        # Prefer shallow files; avoid duplicates from ** glob
        seen = set()
        unique_json = []
        for f in json_files:
            if f.resolve() not in seen:
                seen.add(f.resolve())
                unique_json.append(f)
        if unique_json:
            data_file = str(unique_json[0])
            print(f"[DatasetLoader] Loading JSON/JSONL: {data_file}")
            return hf_load_dataset(
                "json",
                data_files=data_file,
                cache_dir=self.cache_dir,
                split="train",
            )

        raise FileNotFoundError(
            f"No Hugging Face dataset_dict.json or JSON/JSONL under {path}. "
            "Export with datasets.save_to_disk() or provide a .jsonl file."
        )

    def _load_txt_file(
        self,
        path: Path,
        n: int,
        min_w: int,
        max_w: int,
        name: str,
    ) -> List[str]:
        prompts = []
        with open(path, encoding="utf-8") as f:
            for line in f:
                text = line.strip()
                if text:
                    prompts.append(text)
        print(f"[DatasetLoader] Loaded {len(prompts)} lines from {path}")
        return self._filter_sample(prompts, n, min_w, max_w, f"{name} (txt)")

    @staticmethod
    def _iter_items(
        ds: Any, *split_names: str
    ) -> Iterable[Dict[str, Any]]:
        """Iterate rows from a DatasetDict or single Dataset."""
        if hasattr(ds, "keys"):
            for split in split_names:
                if split in ds:
                    for item in ds[split]:
                        yield item
            if not split_names:
                for split in ds.keys():
                    for item in ds[split]:
                        yield item
            return
        for item in ds:
            yield item

    # ------------------------------------------------------------------
    # Dataset-specific loaders
    # ------------------------------------------------------------------

    def _alpaca(
        self,
        n: int,
        min_w: int,
        max_w: int,
        local_path: Optional[str] = None,
    ) -> List[str]:
        try:
            if local_path:
                ds = self._load_hf_local(Path(local_path).expanduser().resolve())
            elif not _HF_AVAILABLE:
                return self._fallback()
            else:
                ds = hf_load_dataset("tatsu-lab/alpaca", cache_dir=self.cache_dir)

            prompts = []
            for item in self._iter_items(ds, "train"):
                instr = item.get("instruction", "").strip()
                ctx = item.get("input", "").strip()
                if not instr:
                    continue
                prompts.append(f"{instr}\n\nContext: {ctx}" if ctx else instr)
            return self._filter_sample(prompts, n, min_w, max_w, "Alpaca")
        except Exception as exc:
            print(f"[DatasetLoader] Alpaca load failed: {exc}")
            return [] if local_path else self._fallback()

    def _dolly(
        self,
        n: int,
        min_w: int,
        max_w: int,
        local_path: Optional[str] = None,
    ) -> List[str]:
        try:
            if local_path:
                ds = self._load_hf_local(Path(local_path).expanduser().resolve())
            elif not _HF_AVAILABLE:
                return self._fallback()
            else:
                ds = hf_load_dataset(
                    "databricks/databricks-dolly-15k", cache_dir=self.cache_dir
                )

            prompts = []
            for item in self._iter_items(ds, "train"):
                instr = item.get("instruction", "").strip()
                ctx = item.get("context", "").strip()
                if not instr:
                    continue
                prompts.append(f"{instr}\n\nContext: {ctx}" if ctx else instr)
            return self._filter_sample(prompts, n, min_w, max_w, "Dolly 15K")
        except Exception as exc:
            print(f"[DatasetLoader] Dolly load failed: {exc}")
            return [] if local_path else self._fallback()

    def _longbench(
        self,
        n: int,
        min_w: int,
        max_w: int,
        local_path: Optional[str] = None,
    ) -> List[str]:
        try:
            prompts: List[str] = []
            if local_path:
                root = Path(local_path).expanduser().resolve()
                subtasks = [
                    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"
                ]
                loaded_any = False
                for sub in subtasks:
                    sub_path = root / sub
                    if sub_path.is_dir():
                        ds = self._load_hf_local(sub_path)
                        loaded_any = True
                    else:
                        continue
                    for item in self._iter_items(ds, "test", "train"):
                        text = item.get("input", "").strip()
                        if text:
                            prompts.append(text)
                if not loaded_any:
                    ds = self._load_hf_local(root)
                    for item in self._iter_items(ds, "test", "train"):
                        text = item.get("input", "").strip()
                        if text:
                            prompts.append(text)
            elif not _HF_AVAILABLE:
                return self._longbench_fallback()
            else:
                subtasks = [
                    "narrativeqa", "qasper", "multifieldqa_en", "hotpotqa", "2wikimqa"
                ]
                for sub in subtasks:
                    try:
                        ds = hf_load_dataset(
                            "THUDM/LongBench", sub, cache_dir=self.cache_dir
                        )
                        for item in self._iter_items(ds, "test"):
                            text = item.get("input", "").strip()
                            if text:
                                prompts.append(text)
                    except Exception as exc:
                        print(f"[DatasetLoader] LongBench/{sub} failed: {exc}")

            if not prompts:
                return [] if local_path else self._longbench_fallback()
            return self._filter_sample(prompts, n, min_w, max_w, "LongBench")
        except Exception as exc:
            print(f"[DatasetLoader] LongBench load failed: {exc}")
            return [] if local_path else self._longbench_fallback()

    def _humaneval(
        self,
        n: int,
        min_w: int,
        max_w: int,
        local_path: Optional[str] = None,
    ) -> List[str]:
        try:
            if local_path:
                ds = self._load_hf_local(Path(local_path).expanduser().resolve())
            elif not _HF_AVAILABLE:
                return self._humaneval_fallback()
            else:
                ds = hf_load_dataset(
                    "openai/openai_humaneval", cache_dir=self.cache_dir
                )

            prompts = [
                f"Complete the following Python function:\n\n{item['prompt']}"
                for item in self._iter_items(ds, "test", "train")
                if item.get("prompt", "").strip()
            ]
            return self._filter_sample(prompts, n, min_w, max_w, "HumanEval")
        except Exception as exc:
            print(f"[DatasetLoader] HumanEval load failed: {exc}")
            return [] if local_path else self._humaneval_fallback()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_sample(
        self, prompts: List[str], n: int, min_w: int, max_w: int, name: str
    ) -> List[str]:
        filtered = [p for p in prompts if min_w <= len(p.split()) <= max_w]
        print(
            f"[DatasetLoader] {name}: {len(prompts)} raw → "
            f"{len(filtered)} after length filter ({min_w}–{max_w} words)"
        )
        if n < len(filtered):
            sampled = random.sample(filtered, n)
            print(f"[DatasetLoader] Sampled {n} from {len(filtered)}")
            return sampled
        print(f"[DatasetLoader] Using all {len(filtered)} prompts")
        return filtered

    @staticmethod
    def _fallback() -> List[str]:
        return [
            "Explain the concept of machine learning.",
            "What are the benefits of renewable energy?",
            "Describe the process of photosynthesis.",
            "How does artificial intelligence work?",
            "What is the difference between supervised and unsupervised learning?",
            "Explain quantum computing in simple terms.",
            "What are the main causes of climate change?",
            "How do neural networks learn?",
        ]

    @staticmethod
    def _longbench_fallback() -> List[str]:
        return [
            (
                "Given the following passage about climate change, analyze the main "
                "arguments and provide a comprehensive summary:\n\n"
                "Climate change refers to long-term changes in global and regional "
                "climate patterns. The primary driver of modern climate change is "
                "human activity, particularly the emission of greenhouse gases such "
                "as carbon dioxide and methane."
            ),
        ]

    @staticmethod
    def _humaneval_fallback() -> List[str]:
        return [
            "Complete the following Python function:\n\n"
            "def fibonacci(n: int) -> int:\n"
            '    """Return the nth Fibonacci number."""\n'
            "    # Your code here",
        ]
