"""
Show complete generations that extract_answer() failed to parse.

These are silently lost instances. Yield is the tightest constraint in the
project (plan section 7.2), so a parser gap is expensive: every unparsed
generation is an eligible question thrown away for no scientific reason.

Read the snippets, spot the format the model actually used, and add a pattern
to _ANS_PATTERNS in model_utils.py.

Run:  python src/inspect_unparsed.py results/e0_Qwen2.5-1.5B-Instruct.jsonl
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from model_utils import extract_answer  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("path")
    ap.add_argument("--n", type=int, default=15)
    args = ap.parse_args()

    recs = [json.loads(l) for l in
            Path(args.path).read_text(encoding="utf-8").splitlines()]
    bad = [r for r in recs if r.get("unparsed_snippet")]

    print(f"{len(bad)} unparsed of {len(recs)} records "
          f"({100*len(bad)/max(1,len(recs)):.0f}%)")
    if not bad:
        print("No parser gap. extract_answer() is keeping up.")
        return

    by_task = Counter(r["task"] for r in bad)
    print("by task:", dict(by_task))

    # Do the snippets contain a bare letter we could have caught?
    salvageable = 0
    for r in bad:
        if re.search(r"\(([ABCD])\)", r["unparsed_snippet"]):
            salvageable += 1
    print(f"snippets containing a bare (LETTER): {salvageable}/{len(bad)} "
          f"-- these are likely recoverable with a wider pattern\n")

    for i, r in enumerate(bad[:args.n], 1):
        print("=" * 78)
        print(f"[{i}] {r['qid']}  task={r['task']}  gold={r['gold']}  "
              f"samples={r['a_u_samples']}")
        print("-" * 78)
        print(r["unparsed_snippet"])
        print(f"--> extract_answer says: {extract_answer(r['unparsed_snippet'])!r}")
    print("=" * 78)
    print("\nIf a clear format repeats, add it to _ANS_PATTERNS in "
          "src/model_utils.py and re-run.")


if __name__ == "__main__":
    main()
