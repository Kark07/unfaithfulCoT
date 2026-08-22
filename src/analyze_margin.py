"""
Does hint-following concentrate at low confidence margin?

Motivation: the competence filter (plan §7, stage 2) keeps only questions the
model answers correctly. On math, "correct" means "computed", so those are
precisely the questions no hint can move -- which would explain GSM8K's 0/51
switch rate across three hint types. If switching concentrates at low margin,
the fix is to select correct-but-uncertain instances, not to change model scale.

Run:  python src/analyze_margin.py results/e0e_*.jsonl
"""

from __future__ import annotations

import argparse
import glob
import json
from collections import defaultdict
from pathlib import Path


def q(vals, p):
    if not vals:
        return float("nan")
    s = sorted(vals)
    return s[min(len(s) - 1, int(p * len(s)))]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    args = ap.parse_args()

    recs = []
    for pat in args.paths:
        for p in glob.glob(pat):
            recs += [json.loads(l) for l in
                     Path(p).read_text(encoding="utf-8").splitlines()]

    elig = [r for r in recs if r.get("margin") is not None]
    if not elig:
        print("No margin data. Re-run the pilot with the updated run_e0_pilot.py.")
        return

    print(f"{len(elig)} eligible instances with margin recorded\n")

    # --- margin distribution per task, switched vs not ---
    hdr = (f"{'task/hint':22} {'n':>4} {'switch':>7} "
           f"{'margin p25':>11} {'p50':>7} {'p75':>7} "
           f"{'mgn|switch':>11} {'mgn|no':>8}")
    print(hdr); print("-" * len(hdr))
    for key in sorted({f"{r['task']}/{r['hint_type']}" for r in elig}):
        g = [r for r in elig if f"{r['task']}/{r['hint_type']}" == key]
        sw = [r for r in g if r.get("switched")]
        no = [r for r in g if not r.get("switched")]
        m = [r["margin"] for r in g]
        ms = [r["margin"] for r in sw]
        mn = [r["margin"] for r in no]
        print(f"{key:22} {len(g):4} {100*len(sw)/len(g):6.0f}% "
              f"{q(m,.25):11.2f} {q(m,.50):7.2f} {q(m,.75):7.2f} "
              f"{(sum(ms)/len(ms) if ms else float('nan')):11.2f} "
              f"{(sum(mn)/len(mn) if mn else float('nan')):8.2f}")

    # --- switch rate by margin tertile, pooled ---
    print("\n--- switch rate by confidence margin (pooled across cells) ---")
    by_task = defaultdict(list)
    for r in elig:
        by_task[r["task"]].append(r)
    for task, g in sorted(by_task.items()):
        g = sorted(g, key=lambda r: r["margin"])
        k = max(1, len(g) // 3)
        bands = [("low  (least confident)", g[:k]),
                 ("mid ", g[k:2 * k]),
                 ("high (most confident)", g[2 * k:])]
        print(f"\n  {task}")
        for name, band in bands:
            if not band:
                continue
            s = sum(1 for r in band if r.get("switched"))
            lo, hi = band[0]["margin"], band[-1]["margin"]
            print(f"    {name:24} n={len(band):3}  switch={100*s/len(band):5.1f}%  "
                  f"margin [{lo:+.2f}, {hi:+.2f}]")

    print("\nReading: if switch rate falls monotonically as margin rises, the")
    print("competence filter is selecting away the instances we need, and the fix")
    print("is a margin cap -- not a bigger model.")


if __name__ == "__main__":
    main()
