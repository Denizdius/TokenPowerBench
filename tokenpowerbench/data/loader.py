"""
Dataset loader supporting Alpaca, Dolly 15K, LongBench, and HumanEval.

Alpaca, Dolly, and HumanEval download through Hugging Face ``datasets``.
LongBench does not: datasets 4 rejects the THUDM dataset script. The loader
downloads ``data.zip`` and reads the task jsonl files inside it.

Offline: ``--dataset-path`` loads local data without Hub access.

  1. **No extra packages** — ``.txt``, ``.json``, or ``.jsonl`` (stdlib only).

  2. **``datasets``** — ``save_to_disk`` directories and Hub cache layout.

  3. **``pyarrow``** (optional) — ``.arrow`` / ``.parquet`` under ``save_to_disk``.
"""

from __future__ import annotations

import io
import json
import random
import shutil
import urllib.request
import zipfile
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
# Public archive. The Hub dataset script is not loaded.
LONGBENCH_DATA_URL = "https://huggingface.co/datasets/THUDM/LongBench/resolve/main/data.zip"
# Leave room for a 2,000-token answer inside a 14,848-token context.
# Applied to every LongBench load, including the 500-token runs.
LONGBENCH_MAX_PROMPT_TOKENS = 12_848
LONGBENCH_TOKENIZER_ID = "Qwen/Qwen3.5-2B"
_LONGBENCH_LOCAL_TOKENIZER = Path("models/Qwen3.5-2B")


class DatasetLoader:
    """
    Load and pre-process prompts from standard LLM evaluation datasets.
    """

    def __init__(self, cache_dir: Optional[str] = None, seed: int = 42) -> None:
        self.cache_dir = cache_dir
        self.seed = seed
        self._random = random.Random(seed)
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

        if not _HF_AVAILABLE:
            raise RuntimeError("Alpaca/Dolly dataset unavailable; no synthetic fallback is allowed")
        try:
            ds = hf_load_dataset("tatsu-lab/alpaca", cache_dir=self.cache_dir)
            prompts = self._rows_to_prompts(self._iter_items(ds, "train"), "alpaca")
            return self._filter_sample(prompts, n, min_w, max_w, "Alpaca")
        except Exception as exc:
            print(f"[DatasetLoader] Alpaca load failed: {exc}")
            raise RuntimeError("Alpaca/Dolly dataset unavailable; no synthetic fallback is allowed") from exc

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

        if not _HF_AVAILABLE:
            raise RuntimeError("Alpaca/Dolly dataset unavailable; no synthetic fallback is allowed")
        try:
            ds = hf_load_dataset(
                "databricks/databricks-dolly-15k", cache_dir=self.cache_dir
            )
            prompts = self._rows_to_prompts(self._iter_items(ds, "train"), "dolly")
            return self._filter_sample(prompts, n, min_w, max_w, "Dolly 15K")
        except Exception as exc:
            print(f"[DatasetLoader] Dolly load failed: {exc}")
            raise RuntimeError("Alpaca/Dolly dataset unavailable; no synthetic fallback is allowed") from exc

    def _longbench(
        self,
        n: int,
        min_w: int,
        max_w: int,
        local_path: Optional[str] = None,
    ) -> List[str]:
        if local_path:
            root = Path(local_path).expanduser().resolve()
            if root.is_file() and root.suffix.lower() == ".zip":
                rows = read_longbench_zip(root, LONGBENCH_EN_NOCODE_SUBTASKS)
                return self._longbench_rows_to_sampled(rows, n, min_w, max_w)
            rows = self._longbench_local_rows(root)
            if rows:
                return self._longbench_rows_to_sampled(rows, n, min_w, max_w)
            out = self._load_local_prompts(
                local_path, "longbench", "LongBench", n, min_w, max_w
            )
            return out if out is not None else []

        zip_path = download_longbench_zip(self.cache_dir)
        rows = read_longbench_zip(zip_path, LONGBENCH_EN_NOCODE_SUBTASKS)
        return self._longbench_rows_to_sampled(rows, n, min_w, max_w)

    def _longbench_local_rows(self, root: Path) -> List[Dict[str, Any]]:
        """Read every English non-code task file under a local directory."""
        if not root.is_dir():
            return []
        rows: List[Dict[str, Any]] = []
        seen: set[Path] = set()
        for sub in LONGBENCH_EN_NOCODE_SUBTASKS:
            candidates = [root / sub, root / f"{sub}.jsonl"]
            candidates.extend(sorted(root.glob(f"**/{sub}.jsonl")))
            for path in candidates:
                key = path.resolve()
                if key in seen or not path.exists():
                    continue
                seen.add(key)
                try:
                    loaded = self._load_local_records(path) if path.is_dir() else self._read_json_file(path)
                except Exception as exc:
                    print(f"[DatasetLoader] LongBench/{sub}: {exc}")
                    continue
                for row in loaded:
                    row.setdefault("dataset", sub)
                    row.setdefault("language", "en")
                    rows.append(row)
        return rows

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
        filtered = self._word_filter(prompts, min_w, max_w, "LongBench")
        filtered = self._apply_longbench_token_limit(filtered)
        return self._take_samples(filtered, n, "LongBench")

    def _longbench_tokenizer(self):
        """Qwen3.5-2B tokenizer used for the hardcoded LongBench length cut."""
        cached = getattr(self, "_lb_tokenizer", None)
        if cached is not None:
            return cached
        try:
            from transformers import AutoTokenizer
        except ImportError as exc:
            raise RuntimeError(
                "LongBench token filter needs transformers to load the Qwen3.5-2B tokenizer; "
                "no synthetic fallback is allowed"
            ) from exc
        sources: List[str] = []
        local = _LONGBENCH_LOCAL_TOKENIZER
        if (local / "tokenizer.json").is_file() or (local / "tokenizer_config.json").is_file():
            sources.append(str(local))
        sources.append(LONGBENCH_TOKENIZER_ID)
        errors: List[str] = []
        for source in sources:
            try:
                tokenizer = AutoTokenizer.from_pretrained(source, trust_remote_code=True)
            except Exception as exc:
                errors.append(f"{source}: {exc}")
                continue
            self._lb_tokenizer = tokenizer
            print(f"[DatasetLoader] LongBench token filter: {source}")
            return tokenizer
        raise RuntimeError(
            "LongBench token filter could not load the Qwen3.5-2B tokenizer; "
            "no synthetic fallback is allowed: " + "; ".join(errors)
        )

    def _apply_longbench_token_limit(self, prompts: List[str]) -> List[str]:
        if not prompts:
            return prompts
        tokenizer = self._longbench_tokenizer()
        kept: List[str] = []
        dropped = 0
        for prompt in prompts:
            n_tokens = len(tokenizer.encode(prompt, add_special_tokens=False))
            if n_tokens <= LONGBENCH_MAX_PROMPT_TOKENS:
                kept.append(prompt)
            else:
                dropped += 1
        print(
            f"[DatasetLoader] LongBench: {len(prompts)} prompts → {len(kept)} "
            f"after <= {LONGBENCH_MAX_PROMPT_TOKENS} Qwen3.5-2B tokens ({dropped} dropped)"
        )
        return kept

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

        if not _HF_AVAILABLE:
            raise RuntimeError("HumanEval dataset unavailable; no synthetic fallback is allowed")
        try:
            ds = hf_load_dataset(
                "openai/openai_humaneval", cache_dir=self.cache_dir
            )
            prompts = self._rows_to_prompts(
                self._iter_items(ds, "test", "train"), "humaneval"
            )
            return self._filter_sample(prompts, n, min_w, max_w, "HumanEval")
        except Exception as exc:
            print(f"[DatasetLoader] HumanEval load failed: {exc}")
            raise RuntimeError("HumanEval dataset unavailable; no synthetic fallback is allowed") from exc

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _word_filter(
        self, prompts: List[str], min_w: int, max_w: int, name: str
    ) -> List[str]:
        filtered = [p for p in prompts if min_w <= len(p.split()) <= max_w]
        print(
            f"[DatasetLoader] {name}: {len(prompts)} prompts → "
            f"{len(filtered)} after words ({min_w}–{max_w})"
        )
        return filtered

    def _take_samples(self, prompts: List[str], n: int, name: str) -> List[str]:
        if n < len(prompts):
            sampled = self._random.sample(prompts, n)
            print(f"[DatasetLoader] {name}: sampled {n} from {len(prompts)}")
            return sampled
        print(f"[DatasetLoader] {name}: using all {len(prompts)} prompts")
        return prompts

    def _filter_sample(
        self, prompts: List[str], n: int, min_w: int, max_w: int, name: str
    ) -> List[str]:
        return self._take_samples(self._word_filter(prompts, min_w, max_w, name), n, name)


def download_longbench_zip(cache_dir: Optional[str] = None) -> Path:
    """Download THUDM LongBench ``data.zip`` once and reuse the cached file."""
    cache = Path(cache_dir).expanduser() if cache_dir else Path.home() / ".cache" / "tokenpowerbench" / "longbench"
    temporary = cache / "data.zip.partial"
    try:
        cache.mkdir(parents=True, exist_ok=True)
        dest = cache / "data.zip"
        if dest.is_file() and dest.stat().st_size > 0:
            print(f"[DatasetLoader] Using cached LongBench zip {dest}")
            return dest
        print(f"[DatasetLoader] Downloading {LONGBENCH_DATA_URL}")
        request = urllib.request.Request(
            LONGBENCH_DATA_URL, headers={"User-Agent": "tokenpowerbench"}
        )
        with urllib.request.urlopen(request, timeout=120) as response, open(temporary, "wb") as out:
            shutil.copyfileobj(response, out)
        temporary.replace(dest)
    except RuntimeError:
        raise
    except Exception as exc:
        if temporary.exists():
            temporary.unlink()
        raise RuntimeError(
            "LongBench dataset unavailable; no synthetic fallback is allowed"
        ) from exc
    return dest


def read_longbench_zip(
    zip_path: Path, tasks: Optional[Iterable[str]] = None
) -> List[Dict[str, Any]]:
    """Read task jsonl members from a LongBench ``data.zip`` archive."""
    wanted = list(tasks) if tasks is not None else None
    wanted_set = set(wanted) if wanted is not None else None
    rows: List[Dict[str, Any]] = []
    try:
        with zipfile.ZipFile(zip_path) as archive:
            members: Dict[str, str] = {}
            for name in archive.namelist():
                if name.endswith("/") or not name.endswith(".jsonl"):
                    continue
                stem = Path(name).stem
                if wanted_set is not None and stem not in wanted_set:
                    continue
                members.setdefault(stem, name)
            if wanted is not None and not members:
                raise RuntimeError(
                    "LongBench data.zip contains none of the English non-code tasks"
                )
            order = wanted if wanted is not None else sorted(members)
            missing = [stem for stem in order if stem not in members]
            if missing:
                print(f"[DatasetLoader] LongBench zip is missing {len(missing)} task files")
            for stem in order:
                name = members.get(stem)
                if name is None:
                    continue
                with archive.open(name) as raw:
                    for line in io.TextIOWrapper(raw, encoding="utf-8"):
                        line = line.strip()
                        if not line:
                            continue
                        row = json.loads(line)
                        if not isinstance(row, dict):
                            continue
                        row.setdefault("dataset", stem)
                        if stem in LONGBENCH_EN_NOCODE_SUBTASKS:
                            row.setdefault("language", "en")
                        rows.append(row)
    except RuntimeError:
        raise
    except Exception as exc:
        raise RuntimeError(
            "LongBench dataset unavailable; no synthetic fallback is allowed"
        ) from exc
    if not rows:
        raise RuntimeError("LongBench dataset unavailable; no synthetic fallback is allowed")
    return rows

