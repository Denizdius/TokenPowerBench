"""
Dataset loader supporting Alpaca, Dolly 15K, LongBench, and HumanEval.

Online: downloads from Hugging Face Hub (requires ``datasets`` + network).

Offline: ``--dataset-path`` loads local data without Hub access.

  1. **No extra packages** — ``.txt``, ``.json``, or ``.jsonl`` (stdlib only).

  2. **``datasets``** — ``save_to_disk`` directories and Hub cache layout.

  3. **``pyarrow``** (optional) — ``.arrow`` / ``.parquet`` under ``save_to_disk``.
"""

from __future__ import annotations

import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

try:
    from datasets import load_dataset as hf_load_dataset
    from datasets import load_from_disk as hf_load_from_disk
    _HF_AVAILABLE = True
except ImportError:
    _HF_AVAILABLE = False

try:
    import pyarrow as pa
    import pyarrow.ipc as pa_ipc
    import pyarrow.parquet as pa_parquet
    _PYARROW_AVAILABLE = True
except ImportError:
    pa_parquet = None  # type: ignore
    _PYARROW_AVAILABLE = False

_SUPPORTED = ("alpaca", "dolly", "longbench", "humaneval")
_METADATA_JSON = frozenset(
    {"dataset_dict.json", "dataset_info.json", "state.json"}
)

# Official THUDM/LongBench configs (main suite; not LongBench-E ``*_e``).
LONGBENCH_SUBTASKS = (
    "narrativeqa",
    "qasper",
    "multifieldqa_en",
    "multifieldqa_zh",
    "hotpotqa",
    "2wikimqa",
    "musique",
    "dureader",
    "gov_report",
    "qmsum",
    "multi_news",
    "vcsum",
    "trec",
    "triviaqa",
    "samsum",
    "lsht",
    "passage_count",
    "passage_retrieval_en",
    "passage_retrieval_zh",
    "lcc",
    "repobench-p",
)

# Energy/long-context campaign defaults: English only, no code completion.
LONGBENCH_CODE_SUBTASKS = frozenset({"lcc", "repobench-p"})
LONGBENCH_ZH_SUBTASKS = frozenset(
    {
        "multifieldqa_zh",
        "dureader",
        "vcsum",
        "lsht",
        "passage_retrieval_zh",
    }
)
LONGBENCH_EN_NOCODE_SUBTASKS = tuple(
    s
    for s in LONGBENCH_SUBTASKS
    if s not in LONGBENCH_CODE_SUBTASKS and s not in LONGBENCH_ZH_SUBTASKS
)


class DatasetLoader:
    """
    Load and pre-process prompts from standard LLM evaluation datasets.
    """

    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42) -> None:
        self.cache_dir = cache_dir
        self.seed = seed
        random.seed(seed)
        if not _HF_AVAILABLE:
            print(
                "[DatasetLoader] HuggingFace `datasets` not installed — "
                "Hub download disabled. Local paths still work via "
                ".json/.jsonl/.txt, or Arrow/Parquet if pyarrow is installed."
            )

    def load(
        self,
        dataset: str,
        num_samples: int = 1000,
        min_words: int = 5,
        max_words: int = 100,
        dataset_path: Optional[str] = None,
    ) -> List[str]:
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
    # Local I/O (no HuggingFace `datasets` required)
    # ------------------------------------------------------------------

    @staticmethod
    def _companion_jsonl_paths(path: Path) -> List[Path]:
        """Common JSONL locations when ``path`` is a save_to_disk directory."""
        if path.is_file():
            return []
        names = [f"{path.name}.jsonl", "alpaca.jsonl", "data.jsonl"]
        candidates = [path.parent / n for n in names]
        candidates.append(path / f"{path.name}.jsonl")
        out: List[Path] = []
        seen: set[Path] = set()
        for c in candidates:
            key = c.resolve()
            if key not in seen and c.is_file():
                seen.add(key)
                out.append(c)
        return out

    def _load_local_records(self, path: Path) -> List[Dict[str, Any]]:
        """
        Load rows from a local path using stdlib JSON and optional pyarrow.

        Tried in order: direct file → companion .jsonl → JSON in tree → Arrow.
        """
        if path.is_file():
            records = self._read_json_file(path)
            if records:
                print(f"[DatasetLoader] Loaded {len(records)} rows (stdlib) from {path}")
                return records

        for jsonl_path in self._companion_jsonl_paths(path):
            records = self._read_json_file(jsonl_path)
            if records:
                print(
                    f"[DatasetLoader] Loaded {len(records)} rows (stdlib) from "
                    f"companion file {jsonl_path}"
                )
                return records

        records = self._collect_json_records_from_dir(path)
        if records:
            print(
                f"[DatasetLoader] Loaded {len(records)} rows (stdlib JSON) under {path}"
            )
            return records

        records = self._load_arrow_records(path)
        if records:
            print(
                f"[DatasetLoader] Loaded {len(records)} rows (pyarrow) under {path}"
            )
            return records

        if _HF_AVAILABLE:
            return list(self._iter_items(self._load_hf_local(path)))

        arrow_files = list(path.rglob("*.arrow")) if path.is_dir() else []
        hint = (
            "Your folder looks like HuggingFace save_to_disk (train/*.arrow). "
            "This Apptainer image has no `pyarrow` or `datasets` in pip, so convert "
            "once on the login node:\n"
            "  python3 scripts/arrow_shard_to_jsonl.py "
            f"--input {path} --output {path.parent / (path.name + '.jsonl')}\n"
            "Then rerun with:\n"
            f"  --dataset-path {path.parent / (path.name + '.jsonl')}"
        )
        if arrow_files and not _PYARROW_AVAILABLE:
            raise FileNotFoundError(hint)
        raise FileNotFoundError(
            f"Could not read data from {path}. Provide a .jsonl file, or see "
            "scripts/arrow_shard_to_jsonl.py to convert save_to_disk Arrow shards."
        )

    @staticmethod
    def _read_json_file(path: Path) -> List[Dict[str, Any]]:
        suffix = path.suffix.lower()
        if suffix not in (".json", ".jsonl"):
            return []

        records: List[Dict[str, Any]] = []
        with open(path, encoding="utf-8") as f:
            if suffix == ".jsonl":
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    row = json.loads(line)
                    if isinstance(row, dict):
                        records.append(row)
                return records

            data = json.load(f)
        if isinstance(data, list):
            records = [r for r in data if isinstance(r, dict)]
        elif isinstance(data, dict):
            for key in ("data", "train", "examples", "instances"):
                chunk = data.get(key)
                if isinstance(chunk, list):
                    records = [r for r in chunk if isinstance(r, dict)]
                    if records:
                        break
        return records

    def _collect_json_records_from_dir(self, root: Path) -> List[Dict[str, Any]]:
        candidates: List[Path] = []
        seen: set[Path] = set()
        for pattern in ("*.jsonl", "*.json", "**/*.jsonl", "**/*.json"):
            for p in sorted(root.glob(pattern)):
                if p.name in _METADATA_JSON:
                    continue
                key = p.resolve()
                if key in seen:
                    continue
                seen.add(key)
                candidates.append(p)

        # Prefer obvious Alpaca / split names
        def sort_key(p: Path) -> tuple:
            name = p.name.lower()
            if "alpaca" in name:
                return (0, name)
            if p.parent.name in ("train", "test", "validation"):
                return (1, name)
            return (2, name)

        candidates.sort(key=sort_key)

        all_records: List[Dict[str, Any]] = []
        for path in candidates:
            rows = self._read_json_file(path)
            if rows:
                all_records.extend(rows)
                # One good file is enough unless we need more (e.g. LongBench subdirs)
                if path.suffix.lower() == ".jsonl" or "alpaca" in path.name.lower():
                    break
        return all_records

    @staticmethod
    def _read_arrow_shard(data_file: Path) -> List[Dict[str, Any]]:
        """Read one Arrow shard (HF ``save_to_disk`` uses IPC *stream* format)."""
        if data_file.suffix == ".parquet":
            table = pa_parquet.read_table(data_file)
            return [r for r in table.to_pylist() if isinstance(r, dict)]

        with open(data_file, "rb") as f:
            # HuggingFace datasets.save_to_disk() writes stream format
            try:
                reader = pa_ipc.open_stream(f)
                table = reader.read_all()
            except Exception:
                f.seek(0)
                reader = pa_ipc.open_file(f)
                table = reader.read_all()
        return [r for r in table.to_pylist() if isinstance(r, dict)]

    def _load_arrow_records(self, root: Path) -> List[Dict[str, Any]]:
        if not _PYARROW_AVAILABLE:
            arrow_files = list(root.rglob("*.arrow"))
            if arrow_files:
                print(
                    f"[DatasetLoader] Found {len(arrow_files)} .arrow file(s) under "
                    f"{root} but pyarrow is not installed in this environment."
                )
            return []

        search_dirs = [root]
        for sub in ("train", "test", "validation"):
            d = root / sub
            if d.is_dir():
                search_dirs.append(d)

        records: List[Dict[str, Any]] = []
        for directory in search_dirs:
            for pattern in ("*.arrow", "*.parquet"):
                for data_file in sorted(directory.glob(pattern)):
                    try:
                        chunk = self._read_arrow_shard(data_file)
                        records.extend(chunk)
                    except Exception as exc:
                        print(f"[DatasetLoader] Skip {data_file}: {exc}")
        return records

    def _load_hf_local(self, path: Path) -> Any:
        """Load via HuggingFace ``datasets`` (requires package)."""
        if not _HF_AVAILABLE:
            raise RuntimeError("pip install datasets")

        if path.is_file() and path.suffix.lower() in (".json", ".jsonl"):
            print(f"[DatasetLoader] Loading JSON/JSONL (HF): {path}")
            return hf_load_dataset(
                "json",
                data_files=str(path),
                cache_dir=self.cache_dir,
                split="train",
            )

        if (path / "dataset_dict.json").exists() or (path / "state.json").exists():
            print(f"[DatasetLoader] Loading from disk (HF): {path}")
            return hf_load_from_disk(str(path))

        json_files = self._collect_json_records_from_dir(path)
        if json_files:
            return json_files

        raise FileNotFoundError(
            f"No readable dataset under {path} for HuggingFace loader."
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
        if isinstance(ds, list):
            for item in ds:
                if isinstance(item, dict):
                    yield item
            return
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

    def _rows_to_prompts(
        self, rows: Iterable[Dict[str, Any]], dataset: str
    ) -> List[str]:
        prompts: List[str] = []
        for item in rows:
            if dataset == "alpaca":
                instr = str(item.get("instruction", "")).strip()
                ctx = str(item.get("input", "")).strip()
                if not instr:
                    continue
                prompts.append(f"{instr}\n\nContext: {ctx}" if ctx else instr)
            elif dataset == "dolly":
                instr = str(item.get("instruction", "")).strip()
                ctx = str(item.get("context", "")).strip()
                if not instr:
                    continue
                prompts.append(f"{instr}\n\nContext: {ctx}" if ctx else instr)
            elif dataset == "longbench":
                # LongBench: ``input`` is the short question; ``context`` is the
                # long document. Concatenate so offline/online runs actually
                # stress long context (not just the query).
                question = str(item.get("input", "")).strip()
                context = str(item.get("context", "")).strip()
                if context and question:
                    prompts.append(f"{context}\n\n{question}")
                elif context:
                    prompts.append(context)
                elif question:
                    prompts.append(question)
            elif dataset == "humaneval":
                p = str(item.get("prompt", "")).strip()
                if p:
                    prompts.append(
                        f"Complete the following Python function:\n\n{p}"
                    )
        return prompts

    def _load_local_prompts(
        self, local_path: str, dataset: str, label: str,
        n: int, min_w: int, max_w: int,
    ) -> Optional[List[str]]:
        path = Path(local_path).expanduser().resolve()
        try:
            rows = self._load_local_records(path)
            if dataset == "longbench":
                return self._longbench_rows_to_sampled(rows, n, min_w, max_w)
            prompts = self._rows_to_prompts(rows, dataset)
            if prompts:
                return self._filter_sample(prompts, n, min_w, max_w, label)
        except Exception as exc:
            print(f"[DatasetLoader] {label} local load failed: {exc}")
        return None

    def _alpaca(
        self,
        n: int,
        min_w: int,
        max_w: int,
        local_path: Optional[str] = None,
    ) -> List[str]:
        if local_path:
            out = self._load_local_prompts(
                local_path, "alpaca", "Alpaca", n, min_w, max_w
            )
            return out if out is not None else []

        try:
            if not _HF_AVAILABLE:
                return self._fallback()
            ds = hf_load_dataset("tatsu-lab/alpaca", cache_dir=self.cache_dir)
            prompts = self._rows_to_prompts(self._iter_items(ds, "train"), "alpaca")
            return self._filter_sample(prompts, n, min_w, max_w, "Alpaca")
        except Exception as exc:
            print(f"[DatasetLoader] Alpaca load failed: {exc}")
            return self._fallback()

    def _dolly(
        self,
        n: int,
        min_w: int,
        max_w: int,
        local_path: Optional[str] = None,
    ) -> List[str]:
        if local_path:
            out = self._load_local_prompts(
                local_path, "dolly", "Dolly 15K", n, min_w, max_w
            )
            return out if out is not None else []

        try:
            if not _HF_AVAILABLE:
                return self._fallback()
            ds = hf_load_dataset(
                "databricks/databricks-dolly-15k", cache_dir=self.cache_dir
            )
            prompts = self._rows_to_prompts(self._iter_items(ds, "train"), "dolly")
            return self._filter_sample(prompts, n, min_w, max_w, "Dolly 15K")
        except Exception as exc:
            print(f"[DatasetLoader] Dolly load failed: {exc}")
            return self._fallback()

    def _longbench(
        self,
        n: int,
        min_w: int,
        max_w: int,
        local_path: Optional[str] = None,
    ) -> List[str]:
        if local_path:
            root = Path(local_path).expanduser().resolve()
            rows: List[Dict[str, Any]] = []
            subtasks = self._longbench_local_subtasks(root)
            for sub in subtasks:
                sub_path = root / sub
                if sub_path.is_dir():
                    try:
                        rows.extend(self._load_local_records(sub_path))
                    except Exception as exc:
                        print(f"[DatasetLoader] LongBench/{sub}: {exc}")
            if not rows:
                out = self._load_local_prompts(
                    local_path, "longbench", "LongBench", n, min_w, max_w
                )
                return out if out is not None else []
            return self._longbench_rows_to_sampled(rows, n, min_w, max_w)

        try:
            if not _HF_AVAILABLE:
                return self._longbench_fallback()
            rows: List[Dict[str, Any]] = []
            # Hub path: only English non-code configs (same campaign filter).
            for sub in LONGBENCH_EN_NOCODE_SUBTASKS:
                try:
                    ds = hf_load_dataset(
                        "THUDM/LongBench", sub, cache_dir=self.cache_dir
                    )
                    rows.extend(list(self._iter_items(ds, "test")))
                except Exception as exc:
                    print(f"[DatasetLoader] LongBench/{sub} failed: {exc}")
            if not rows:
                return self._longbench_fallback()
            return self._longbench_rows_to_sampled(rows, n, min_w, max_w)
        except Exception as exc:
            print(f"[DatasetLoader] LongBench load failed: {exc}")
            return self._longbench_fallback()

    def _longbench_rows_to_sampled(
        self,
        rows: List[Dict[str, Any]],
        n: int,
        min_w: int,
        max_w: int,
    ) -> List[str]:
        """Filter LongBench rows: language=en, drop code, then word filter + sample."""
        raw_n = len(rows)
        en_rows = [
            r
            for r in rows
            if str(r.get("language", "")).strip().lower() == "en"
        ]
        # Rows without a language tag stay only if dataset is known EN/non-code.
        if not en_rows and rows:
            en_rows = [
                r
                for r in rows
                if str(r.get("dataset", "")).strip().lower()
                in LONGBENCH_EN_NOCODE_SUBTASKS
            ]
        kept = [
            r
            for r in en_rows
            if str(r.get("dataset", "")).strip().lower()
            not in LONGBENCH_CODE_SUBTASKS
            and str(r.get("dataset", "")).strip().lower()
            not in LONGBENCH_ZH_SUBTASKS
        ]
        prompts = self._rows_to_prompts(kept, "longbench")
        print(
            f"[DatasetLoader] LongBench: {raw_n} raw → "
            f"{len(en_rows)} after language=en → "
            f"{len(kept)} after drop code/zh → "
            f"{len(prompts)} prompts"
        )
        return self._filter_sample(prompts, n, min_w, max_w, "LongBench")

    @staticmethod
    def _longbench_local_subtasks(root: Path) -> List[str]:
        """Known LongBench configs plus any extra subdirs under ``root``."""
        names = list(LONGBENCH_SUBTASKS)
        if root.is_dir():
            for child in sorted(root.iterdir()):
                if (
                    child.is_dir()
                    and not child.name.startswith(".")
                    and child.name not in names
                ):
                    names.append(child.name)
        return names

    def _humaneval(
        self,
        n: int,
        min_w: int,
        max_w: int,
        local_path: Optional[str] = None,
    ) -> List[str]:
        if local_path:
            out = self._load_local_prompts(
                local_path, "humaneval", "HumanEval", n, min_w, max_w
            )
            return out if out is not None else []

        try:
            if not _HF_AVAILABLE:
                return self._humaneval_fallback()
            ds = hf_load_dataset(
                "openai/openai_humaneval", cache_dir=self.cache_dir
            )
            prompts = self._rows_to_prompts(
                self._iter_items(ds, "test", "train"), "humaneval"
            )
            return self._filter_sample(prompts, n, min_w, max_w, "HumanEval")
        except Exception as exc:
            print(f"[DatasetLoader] HumanEval load failed: {exc}")
            return self._humaneval_fallback()

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _filter_sample(
        self, prompts: List[str], n: int, min_w: int, max_w: int, name: str
    ) -> List[str]:
        filtered = [p for p in prompts if min_w <= len(p.split()) <= max_w]
        print(
            f"[DatasetLoader] {name}: {len(prompts)} prompts → "
            f"{len(filtered)} after words ({min_w}–{max_w})"
        )
        if n < len(filtered):
            sampled = random.sample(filtered, n)
            print(f"[DatasetLoader] {name}: sampled {n} from {len(filtered)}")
            return sampled
        print(f"[DatasetLoader] {name}: using all {len(filtered)} prompts")
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
