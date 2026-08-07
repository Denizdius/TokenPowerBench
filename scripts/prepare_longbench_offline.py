#!/usr/bin/env python3
"""
Download the full LongBench main suite (21 subtasks; same list as
DatasetLoader.LONGBENCH_SUBTASKS) and write an offline copy shaped like Alpaca:

  longbench/                 # optional save_to_disk per subtask
    narrativeqa/
    qasper/
    ...
  longbench.jsonl            # flat JSONL for Apptainer (stdlib-friendly)

Run once on a login node with HuggingFace ``datasets`` + network, then point
benchmarks at the JSONL (no Hub / no ``datasets`` needed inside the container):

  python3 scripts/prepare_longbench_offline.py \\
      --output-dir /gpfs/.../TokenPowerBench

  python3 run_single_node.py \\
      --dataset longbench \\
      --dataset-path /gpfs/.../TokenPowerBench/longbench.jsonl \\
      --min-words 500 --max-words 4000 \\
      --max-model-len 8192 \\
      ...
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

try:
    from tokenpowerbench.data.loader import LONGBENCH_SUBTASKS as SUBTASKS
except ImportError:
    # Fallback if package import path is unavailable on the login node
    SUBTASKS = (
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


def _prompt_from_row(row: Dict[str, Any]) -> str:
    question = str(row.get("input", "")).strip()
    context = str(row.get("context", "")).strip()
    if context and question:
        return f"{context}\n\n{question}"
    if context:
        return context
    return question


def _word_count(text: str) -> int:
    return len(text.split())


def _print_stats(word_counts: List[int], label: str) -> None:
    if not word_counts:
        print(f"[{label}] no prompts")
        return
    wc = sorted(word_counts)
    print(
        f"[{label}] n={len(wc)}  "
        f"min={wc[0]}  p25={wc[len(wc)//4]}  "
        f"median={statistics.median(wc):.0f}  "
        f"p75={wc[(3*len(wc))//4]}  max={wc[-1]}  "
        f"mean={statistics.mean(wc):.0f} words"
    )
    for lo, hi in ((2, 300), (500, 4000), (1000, 6000), (2000, 8000)):
        n = sum(1 for w in wc if lo <= w <= hi)
        print(f"  words {lo}–{hi}: {n} prompts")


def main() -> None:
    p = argparse.ArgumentParser(
        description="Download LongBench offline (save_to_disk + JSONL)"
    )
    p.add_argument(
        "--output-dir",
        type=Path,
        default=Path("."),
        help="Directory for longbench/ and longbench.jsonl (default: .)",
    )
    p.add_argument(
        "--cache-dir",
        default=None,
        help="HuggingFace datasets cache_dir (optional)",
    )
    p.add_argument(
        "--skip-save-to-disk",
        action="store_true",
        help="Only write longbench.jsonl (skip per-subtask save_to_disk folders)",
    )
    p.add_argument(
        "--repo",
        default="THUDM/LongBench",
        help="HF dataset repo (default: THUDM/LongBench)",
    )
    args = p.parse_args()

    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise SystemExit(
            "Needs HuggingFace `datasets` (run on login node):\n"
            "  pip install datasets"
        ) from exc

    out_root = args.output_dir.expanduser().resolve()
    disk_root = out_root / "longbench"
    jsonl_path = out_root / "longbench.jsonl"
    out_root.mkdir(parents=True, exist_ok=True)
    if not args.skip_save_to_disk:
        disk_root.mkdir(parents=True, exist_ok=True)

    all_rows: List[Dict[str, Any]] = []
    word_counts: List[int] = []

    for sub in SUBTASKS:
        print(f"Loading {args.repo} / {sub} …")
        ds = load_dataset(args.repo, sub, cache_dir=args.cache_dir)
        split = "test" if "test" in ds else next(iter(ds.keys()))
        split_ds = ds[split]

        if not args.skip_save_to_disk:
            sub_dir = disk_root / sub
            print(f"  save_to_disk → {sub_dir}")
            # DatasetLoader looks for subtask dirs; keep a DatasetDict with the
            # original split name so load_from_disk / arrow readers still work.
            ds.save_to_disk(str(sub_dir))

        n_sub = 0
        for row in split_ds:
            item = dict(row)
            item.setdefault("dataset", sub)
            all_rows.append(item)
            prompt = _prompt_from_row(item)
            if prompt:
                word_counts.append(_word_count(prompt))
            n_sub += 1
        print(f"  {n_sub} rows from split={split!r}")

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with open(jsonl_path, "w", encoding="utf-8") as f:
        for row in all_rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    print(f"\nWrote {len(all_rows)} rows → {jsonl_path}")
    if not args.skip_save_to_disk:
        print(f"Per-subtask folders → {disk_root}/{{{','.join(SUBTASKS)}}}")
    _print_stats(word_counts, "prompt = context + input")

    print(
        "\nUse offline:\n"
        f"  --dataset longbench --dataset-path {jsonl_path}\n"
        "  (or --dataset-path "
        f"{disk_root}  for per-subtask dirs)\n"
        "\nSuggested starters with max_model_len=8192:\n"
        "  --min-words 500 --max-words 4000\n"
        "Then widen toward 1000–6000 once jobs are stable."
    )


if __name__ == "__main__":
    main()
