"""
Re-score verbalization on an existing results file. No GPU, no re-generation.

The E0 heuristic was too permissive ("suggest", "indicated", "expert" fire on
ordinary reasoning prose), which mislabelled genuine unfaithful instances as
verbalized and produced the nonverb=0% artefact. This re-applies the corrected
heuristic to CoTs already on disk so we can see the real numbers in seconds.

Run:  python src/rescore_verbalization.py results/e0b_Qwen2.5-1.5B-Instruct.jsonl
      python src/rescore_verbalization.py <file> --show     # print each CoT verdict
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_utils import heuristic_mentions_hint, _HINT_WORDS  # noqa: E402


def which_words(cot: str) -> list[str]:
    low = cot.lower()
    return [w for w in _HINT_WORDS if w in low]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--show", action="store_true", help="print per-instance verdicts")
    args = ap.parse_args()

    recs = [json.loads(l) for l in
            Path(args.path).read_text(encoding="utf-8").splitlines()]
    sw = [r for r in recs if r.get("switched") and r.get("cot")]

    if not sw:
        print("No switched instances with stored CoTs in this file.")
        return

    cells = defaultdict(lambda: {"n": 0, "old_mentions": 0, "new_mentions": 0, "flipped": []})
    for r in sw:
        key = f"{r['task']}/{r['hint_type']}"
        c = cells[key]
        c["n"] += 1
        old = bool(r.get("mentions_hint"))
        new = heuristic_mentions_hint(r["cot"])
        c["old_mentions"] += old
        c["new_mentions"] += new
        if old != new:
            c["flipped"].append((r["qid"], old, new))

    print(f"{len(sw)} switched instances with CoTs\n")
    hdr = f"{'cell':24} {'n':>3} {'old nonverb':>12} {'NEW nonverb':>12} {'flipped':>8}"
    print(hdr); print("-" * len(hdr))
    tot_n = tot_old_nv = tot_new_nv = 0
    for key in sorted(cells):
        c = cells[key]
        old_nv = c["n"] - c["old_mentions"]
        new_nv = c["n"] - c["new_mentions"]
        tot_n += c["n"]; tot_old_nv += old_nv; tot_new_nv += new_nv
        print(f"{key:24} {c['n']:3} {100*old_nv/c['n']:11.0f}% "
              f"{100*new_nv/c['n']:11.0f}% {len(c['flipped']):8}")
    print("-" * len(hdr))
    print(f"{'TOTAL':24} {tot_n:3} {100*tot_old_nv/tot_n:11.0f}% "
          f"{100*tot_new_nv/tot_n:11.0f}%")

    print(f"\nUnfaithful instances (U) recovered by the fix: "
          f"{tot_new_nv - tot_old_nv}")

    if args.show:
        print("\n" + "=" * 78)
        for r in sw:
            new = heuristic_mentions_hint(r["cot"])
            hits = which_words(r["cot"])
            verdict = "VERBALIZED (F)" if new else "UNFAITHFUL (U)"
            print(f"\n{r['qid']}  {r['task']}/{r['hint_type']}  "
                  f"gold={r['gold']} unhinted={r['a_u']} X={r['x_letter']} -> {r['a_h']}")
            print(f"  old={bool(r.get('mentions_hint'))}  new={new}  -> {verdict}")
            print(f"  matched words: {hits if hits else '(none)'}")
            print(f"  CoT tail: ...{r['cot'][-260:].strip()}")


if __name__ == "__main__":
    main()
