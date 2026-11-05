#!/usr/bin/env python3
"""Compare kernel-gate runs against the committed baseline.

Usage:
    for i in 1 2 3; do ./build-vipera/bin/kernel-gate > /tmp/after.$i.jsonl; done
    python3 tools/kernel_gate/compare.py tools/kernel_gate/baseline.json /tmp/after.*.jsonl

The protocol is median of three runs, judged by tools/vipera/gate.h: the delta
must beat 5% strictly. Exit code: 0 when every row passes, 1 when a row fails,
2 on a malformed run file.
"""

from __future__ import annotations

import json
import statistics
import sys

MIN_DELTA = 0.05


def load_baseline(path: str) -> dict[str, dict]:
    with open(path, encoding="utf-8") as f:
        data = json.load(f)
    return {row["quant"]: row for row in data["rows"]}


def load_run(path: str) -> dict[str, float]:
    rows: dict[str, float] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                print(f"{path}: bad line {line!r} ({exc})", file=sys.stderr)
                raise SystemExit(2) from exc
            rows[row["quant"]] = float(row["gbs"])
    return rows


def median_runs(paths: list[str]) -> dict[str, float]:
    per_run = [load_run(p) for p in paths]
    quants = set(per_run[0])
    merged: dict[str, float] = {}
    for quant in quants:
        values = [run[quant] for run in per_run if quant in run]
        merged[quant] = statistics.median(values)
    return merged


def main(argv: list[str]) -> int:
    if len(argv) < 3:
        print(__doc__)
        return 2
    base = load_baseline(argv[1])
    after = median_runs(argv[2:])

    failed = False
    for quant, b in base.items():
        if quant not in after:
            print(f"{quant:<6} missing in run")
            failed = True
            continue
        delta = (after[quant] - b["gbs"]) / b["gbs"]
        verdict = "pass" if delta > MIN_DELTA else "fail"
        if verdict == "fail":
            failed = True
        print(
            f"{quant:<6} {b['gbs']:7.2f} -> {after[quant]:7.2f} GB/s  "
            f"{delta * 100:+6.1f}%  gate: {verdict}"
        )
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
