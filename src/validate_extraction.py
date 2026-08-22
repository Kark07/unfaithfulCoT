"""
Audit answer extraction by hand. The switch test is

    switched := (a_u != X) and (a_h == X)

so it is only as trustworthy as extract_answer() on BOTH runs. A silent
extraction error fabricates or destroys switches invisibly -- and GSM8K's
0/51 result rests entirely on this. Math CoTs end in numbers and equations,
so they can fail differently from prose CoTs.

This prints the verbatim tail of each generation next to what was extracted,
so you can score it yourself and get a real accuracy number.

Run:  python src/validate_extraction.py results/e0d_*.jsonl --task gsm8k_mc --n 15
      python src/validate_extraction.py results/e0d_*.jsonl --score   # interactive
"""

from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_utils import extract_answer  # noqa: E402

BAR = "=" * 78


def load(paths: list[str]) -> list[dict]:
    recs = []
    for pat in paths:
        for p in glob.glob(pat):
            recs += [json.loads(l) for l in
                     Path(p).read_text(encoding="utf-8").splitlines()]
    return recs


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("paths", nargs="+")
    ap.add_argument("--task", default=None, help="filter to one task")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--which", choices=["unhinted", "hinted", "both"], default="both")
    ap.add_argument("--tail", type=int, default=420)
    ap.add_argument("--score", action="store_true",
                    help="prompt for the correct letter on each and report accuracy")
    args = ap.parse_args()

    recs = load(args.paths)
    if args.task:
        recs = [r for r in recs if r["task"] == args.task]

    have = [r for r in recs if r.get("unhinted_gen") or r.get("hinted_gen")]
    if not have:
        print("No verbatim generations stored in these files.\n"
              "They predate the audit patch -- re-run the pilot to capture them.")
        return

    print(f"{len(have)} records with stored generations"
          f"{f' (task={args.task})' if args.task else ''}\n")

    shown = 0
    agree = disagree = 0
    for r in have:
        if shown >= args.n:
            break
        panes = []
        if args.which in ("unhinted", "both") and r.get("unhinted_gen"):
            panes.append(("UNHINTED (P_0)", r["unhinted_gen"], r["a_u"]))
        if args.which in ("hinted", "both") and r.get("hinted_gen"):
            panes.append((f"HINTED (P_X -> {r['x_letter']})", r["hinted_gen"], r["a_h"]))
        if not panes:
            continue

        shown += 1
        print(BAR)
        print(f"[{shown}] {r['qid']}  {r['task']}/{r['hint_type']}  gold={r['gold']}  "
              f"a_u={r['a_u']}  X={r.get('x_letter')}  a_h={r.get('a_h')}  "
              f"switched={r.get('switched')}")
        for label, gen, extracted in panes:
            print("-" * 78)
            print(f"{label}   extracted -> {extracted!r}   "
                  f"(re-extract now: {extract_answer(gen)!r})")
            print("..." + gen[-args.tail:].strip())
        print(BAR)

        if args.score:
            truth = input("  what letter did the model ACTUALLY give? "
                          "[A/B/C/D / n=none / s=skip]: ").strip().upper()
            if truth == "S":
                continue
            got = panes[-1][2]
            want = None if truth == "N" else truth
            if got == want:
                agree += 1
            else:
                disagree += 1
                print(f"    MISMATCH: extractor said {got!r}, you said {want!r}")

    if args.score and (agree + disagree):
        n = agree + disagree
        print(f"\nextraction accuracy: {agree}/{n} ({100*agree/n:.0f}%)")
        if disagree:
            print("Mismatches mean the switch rates are not trustworthy. Add the "
                  "missed format to _ANS_PATTERNS in src/model_utils.py and re-run "
                  "src/rescore_verbalization.py-style re-analysis.")


if __name__ == "__main__":
    main()
