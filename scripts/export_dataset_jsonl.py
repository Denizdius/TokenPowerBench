#!/usr/bin/env python3
"""
One-time export: HuggingFace save_to_disk folder -> .jsonl (stdlib-friendly).

Run on a machine that has ``datasets`` (login node), then use the .jsonl offline
inside Apptainer without ``datasets`` or ``pyarrow``:

  python3 scripts/export_dataset_jsonl.py \\
      --input /gpfs/.../TokenPowerBench/alpaca \\
      --output /gpfs/.../TokenPowerBench/alpaca.jsonl

  python3 run_single_node.py --dataset alpaca --dataset-path .../alpaca.jsonl ...

For LongBench (download + JSONL in one step), prefer:

  python3 scripts/prepare_longbench_offline.py --output-dir /gpfs/.../TokenPowerBench
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    p = argparse.ArgumentParser(description="Export save_to_disk dataset to JSONL")
    p.add_argument("--input", required=True, help="save_to_disk directory")
    p.add_argument("--output", required=True, help="Output .jsonl path")
    p.add_argument("--split", default="train", help="Split name (default: train)")
    args = p.parse_args()

    try:
        from datasets import load_from_disk
    except ImportError as exc:
        raise SystemExit(
            "This script needs HuggingFace `datasets` (run on login node):\n"
            "  pip install datasets"
        ) from exc

    root = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    if not root.is_dir():
        raise SystemExit(f"Input not found: {root}")

    ds = load_from_disk(str(root))
    if args.split not in ds:
        raise SystemExit(f"Split {args.split!r} not in {list(ds.keys())}")

    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for row in ds[args.split]:
            f.write(json.dumps(dict(row), ensure_ascii=False) + "\n")
            n += 1
    print(f"Wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
