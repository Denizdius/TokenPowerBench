#!/usr/bin/env python3
"""
Download LongBench ``data.zip`` (the Hub dataset script is not used) and write
an offline JSONL. The default is the English non-code task list. ``--all-tasks``
keeps the Chinese and code tasks as well. The benchmark loader still drops
prompts longer than 12,848 Qwen3.5-2B tokens at run time.

  longbench.jsonl
  longbench/<task>.jsonl

  python3 scripts/prepare_longbench_offline.py \\
      --output-dir /path/to/TokenPowerBench

  tokenpowerbench \\
      --dataset longbench \\
      --dataset-path /path/to/TokenPowerBench/longbench.jsonl \\
      --min-words 800 --max-words 6000 \\
      --max-model-len 14848 \\
      --language-model-only
"""

from __future__ import annotations

import argparse
import json
import statistics
from pathlib import Path
from typing import Any, Dict, List

from tokenpowerbench.data.loader import (
    LONGBENCH_EN_NOCODE_SUBTASKS,
    LONGBENCH_SUBTASKS,
    download_longbench_zip,
    read_longbench_zip,
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
        f"p75={wc[(3 * len(wc)) // 4]}  max={wc[-1]}  "
        f"mean={statistics.mean(wc):.0f} words"
    )
    for lo, hi in ((800, 6000), (2, 300), (500, 4000)):
        n = sum(1 for w in wc if lo <= w <= hi)
        print(f"  words {lo}–{hi}: {n} prompts")


def main() -> None:
    parser = argparse.ArgumentParser(description="Download LongBench data.zip and write JSONL")
    parser.add_argument("--output-dir", type=Path, default=Path("."), help="Directory for longbench.jsonl")
    parser.add_argument("--cache-dir", default=None, help="Directory for the cached data.zip")
    parser.add_argument("--all-tasks", action="store_true", help="Keep Chinese and code tasks in the JSONL")
    parser.add_argument("--flat-only", action="store_true", help="Write only the combined JSONL, not per-task files")
    args = parser.parse_args()

    tasks = list(LONGBENCH_SUBTASKS) if args.all_tasks else list(LONGBENCH_EN_NOCODE_SUBTASKS)
    zip_path = download_longbench_zip(args.cache_dir)
    rows = read_longbench_zip(zip_path, tasks)
    out_root = args.output_dir.expanduser().resolve()
    out_root.mkdir(parents=True, exist_ok=True)
    jsonl_path = out_root / "longbench.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    if not args.flat_only:
        task_dir = out_root / "longbench"
        task_dir.mkdir(parents=True, exist_ok=True)
        grouped: Dict[str, List[Dict[str, Any]]] = {task: [] for task in tasks}
        for row in rows:
            grouped.setdefault(str(row.get("dataset", "")), []).append(row)
        for task, task_rows in grouped.items():
            if not task or not task_rows:
                continue
            path = task_dir / f"{task}.jsonl"
            with path.open("w", encoding="utf-8") as handle:
                for row in task_rows:
                    handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"Per-task files → {task_dir}")

    word_counts = [_word_count(prompt) for row in rows if (prompt := _prompt_from_row(row))]
    print(f"\nWrote {len(rows)} rows → {jsonl_path}")
    _print_stats(word_counts, "prompt = context + blank line + input")
    print(
        "\nUse offline:\n"
        f"  --dataset longbench --dataset-path {jsonl_path}\n"
        "  --min-words 800 --max-words 6000 --max-model-len 14848\n"
        "Prompts over 12,848 Qwen3.5-2B tokens are dropped when the benchmark loads this file."
    )


if __name__ == "__main__":
    main()
