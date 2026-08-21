"""
Read CoTs by hand -- E0 step 3, and the seed for the 50 hand-labels in E1.

The two things only a human can decide at E0:
  1. Is the CoT real multi-step reasoning, or degenerate/looping?
  2. Does it mention the hint? (the heuristic is a placeholder for the E1 judge)

Run:  python src/inspect_cots.py results/e0_Qwen2.5-1.5B-Instruct.jsonl --n 10
      python src/inspect_cots.py <file> --n 20 --label   # write hand labels
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

BAR = "=" * 78


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=10)
    ap.add_argument("--only-switched", action="store_true", default=True)
    ap.add_argument("--all", dest="only_switched", action="store_false")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--label", action="store_true",
                    help="prompt for hand labels and write them alongside")
    args = ap.parse_args()

    p = Path(args.path)
    recs = [json.loads(l) for l in p.read_text(encoding="utf-8").splitlines()]
    pool = [r for r in recs if r.get("switched")] if args.only_switched else \
           [r for r in recs if r.get("cot")]
    if not pool:
        print("No matching records. Try --all.")
        return

    random.Random(args.seed).shuffle(pool)
    pool = pool[:args.n]
    labels = []

    for i, r in enumerate(pool, 1):
        print(f"\n{BAR}\n[{i}/{len(pool)}] {r['qid']}  task={r['task']}  hint={r['hint_type']}")
        print(f"gold={r['gold']}  unhinted={r['a_u']}  hint->X={r['x_letter']}  "
              f"hinted answer={r['a_h']}  switched={r['switched']}")
        if r.get("m_disc_hinted") is not None:
            d = r["m_disc_hinted"] - r["m_disc_neutral"]
            print(f"M_disc  hinted={r['m_disc_hinted']:+.3f}  "
                  f"neutral={r['m_disc_neutral']:+.3f}  delta={d:+.3f}")
        print(f"heuristic says mentions_hint = {r.get('mentions_hint')}")
        print("-" * 78)
        print(r["cot"][:2200])
        print(BAR)

        if args.label:
            coh = input("  coherent reasoning? [y/n/s(kip)]: ").strip().lower()
            men = input("  mentions the hint?  [y/n/s(kip)]: ").strip().lower()
            labels.append({"qid": r["qid"], "task": r["task"], "hint_type": r["hint_type"],
                           "coherent": coh, "mentions_hint_human": men,
                           "mentions_hint_heuristic": r.get("mentions_hint")})

    if args.label and labels:
        out = p.with_name(p.stem + "_handlabels.jsonl")
        with out.open("w", encoding="utf-8") as f:
            for l in labels:
                f.write(json.dumps(l) + "\n")
        agree = [l for l in labels
                 if l["mentions_hint_human"] in ("y", "n")
                 and (l["mentions_hint_human"] == "y") == bool(l["mentions_hint_heuristic"])]
        scored = [l for l in labels if l["mentions_hint_human"] in ("y", "n")]
        print(f"\nwrote {len(labels)} labels -> {out}")
        if scored:
            print(f"heuristic agreement: {len(agree)}/{len(scored)} "
                  f"({100*len(agree)/len(scored):.0f}%)")
        coh = [l for l in labels if l["coherent"] == "y"]
        print(f"coherent: {len(coh)}/{len(labels)}")


if __name__ == "__main__":
    main()
