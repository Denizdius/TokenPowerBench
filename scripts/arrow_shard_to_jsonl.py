#!/usr/bin/env python3
"""
Convert HuggingFace save_to_disk Arrow shards to JSONL.

Needs **only pyarrow** (not the full ``datasets`` package). Run once on the
login node, then use ``--dataset-path .../alpaca.jsonl`` inside Apptainer.

  pip install pyarrow   # or an offline wheel

  python3 scripts/arrow_shard_to_jsonl.py \\
      --input /gpfs/.../TokenPowerBench/alpaca \\
      --output /gpfs/.../TokenPowerBench/alpaca.jsonl
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def _read_shard(path: Path) -> list:
    import pyarrow.ipc as ipc

    with open(path, "rb") as f:
        try:
            reader = ipc.open_stream(f)
            table = reader.read_all()
        except Exception:
            f.seek(0)
            reader = ipc.open_file(f)
            table = reader.read_all()
    return table.to_pylist()


def main() -> None:
    p = argparse.ArgumentParser(description="Arrow save_to_disk -> JSONL")
    p.add_argument("--input", required=True, help="Dataset root (has train/*.arrow)")
    p.add_argument("--output", required=True, help="Output .jsonl path")
    args = p.parse_args()

    try:
        import pyarrow  # noqa: F401
    except ImportError as exc:
        raise SystemExit(
            "Install pyarrow only (no datasets needed):\n"
            "  pip install pyarrow"
        ) from exc

    root = Path(args.input).expanduser().resolve()
    out = Path(args.output).expanduser().resolve()
    shards = sorted(root.rglob("*.arrow"))
    if not shards:
        raise SystemExit(f"No .arrow files under {root}")

    n = 0
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", encoding="utf-8") as f:
        for shard in shards:
            print(f"Reading {shard} …")
            for row in _read_shard(shard):
                if isinstance(row, dict):
                    f.write(json.dumps(row, ensure_ascii=False) + "\n")
                    n += 1
    print(f"Wrote {n} rows to {out}")


if __name__ == "__main__":
    main()
