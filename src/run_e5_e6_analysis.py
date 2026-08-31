"""
E5 + E6 -- baselines and cross-task overlap.  research_plan_v2.md §8.

E5 is why this project has a result rather than a pile of numbers. A raw Jaccard
is uninterpretable on its own (plan §1 row 3: v1 preregistered J>0.5 = universal,
borrowed from Merullo's 78% NODE overlap; edge-level Jaccard is structurally far
lower, so that threshold would have forced the "task-specific" conclusion for
arithmetic reasons). Three reference quantities fix it:

  (a) split-half ceiling  -- max overlap the pipeline can produce (from run_circuits)
  (b) task-solving floor  -- overlap you get for free from plain MC machinery
  (c) score-matched null  -- permutation p-value that respects EAP-IG's skew,
                             unlike a hypergeometric test which assumes edges
                             are exchangeable

E6 then reports, for every pair:
    J_raw,  J_norm = J_raw / J_half,  p_perm

E6b is the main figure: a two-way decomposition. Within-hint-across-task vs
within-task-across-hint answers "is the mechanism organised by WHAT is reasoned
about, or by HOW the hint arrives?" -- sharper than the headline question, and
free from data already collected.

E6c tests separability: is the influence circuit the same as the suppression
circuit? If not, unfaithfulness is two mechanisms, and E9 can target one.

Run:
  python src/run_e5_e6_analysis.py --p 60
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from circuit_utils import (CIRCUITS, RESULTS, jaccard, load_circuit,  # noqa: E402
                           permutation_null)


def discover(p: int) -> dict:
    """Find every saved circuit for aggregation threshold p."""
    found = defaultdict(dict)
    for f in sorted(CIRCUITS.glob(f"*_p{p}.json")):
        edges, meta = load_circuit(f)
        arm = meta.get("arm", "?")
        key = (meta.get("task", "?"), meta.get("hint", "?"))
        found[arm][key] = {"edges": edges, "meta": meta, "path": f}
    return found


def ceiling_for(arm: str, task: str, hint: str) -> float:
    for name in (f"{arm}_{task}_{hint}_summary.json",
                 f"suppression_{task}_{hint}_summary.json"):
        f = CIRCUITS / name
        if f.exists():
            return json.loads(f.read_text(encoding="utf-8")).get("split_half_ceiling", float("nan"))
    return float("nan")


def scores_for(arm: str, task: str, hint: str):
    f = CIRCUITS / f"{arm}_{task}_{hint}_scores.pt"
    return torch.load(f) if f.exists() else None


def report_arm(arm: str, cells: dict, args) -> dict:
    print("\n" + "=" * 88)
    print(f"E6a -- cross-cell overlap, arm = {arm}")
    print("=" * 88)
    keys = sorted(cells)
    if len(keys) < 2:
        print(f"  only {len(keys)} cell(s); need >=2 to compare"); return {}

    rows = []
    hdr = (f"{'pair':38} {'J_raw':>7} {'J_half':>7} {'J_norm':>7} "
           f"{'null':>7} {'p95':>7} {'sig':>4}")
    print(hdr); print("-" * len(hdr))
    for (t1, h1), (t2, h2) in itertools.combinations(keys, 2):
        a, b = cells[(t1, h1)]["edges"], cells[(t2, h2)]["edges"]
        jr = jaccard(a, b)
        c1, c2 = ceiling_for(arm, t1, h1), ceiling_for(arm, t2, h2)
        half = (c1 + c2) / 2 if c1 == c1 and c2 == c2 else float("nan")
        jn = jr / half if half and half == half and half > 0 else float("nan")

        sa, sb = scores_for(arm, t1, h1), scores_for(arm, t2, h2)
        if sa is not None and sb is not None:
            null = permutation_null(sa, sb, len(a), len(b), n_draws=args.draws)
        else:
            null = {"mean": float("nan"), "p95": float("nan")}
        sig = "*" if (null["p95"] == null["p95"] and jr > null["p95"]) else ""

        label = f"{t1}/{h1} <-> {t2}/{h2}"
        print(f"{label:38} {jr:7.4f} {half:7.4f} {jn:7.3f} "
              f"{null['mean']:7.4f} {null['p95']:7.4f} {sig:>4}")
        rows.append({"a": f"{t1}/{h1}", "b": f"{t2}/{h2}", "J_raw": jr,
                     "J_half": half, "J_norm": jn, "null": null, "sig": bool(sig)})

    print("\ninterpretation (calibrated against E5, not v1's invented 0.5/0.2):")
    print("  J_norm > 0.7      approaches the within-cell ceiling -> strong sharing")
    print("  0.4 - 0.7         shared core + task-specific components")
    print("  < 0.4 but > floor weak sharing above the generic-MC baseline")
    print("  <= task-solving floor   NO unfaithfulness-specific shared mechanism")
    return {"pairs": rows}


def report_two_way(arm: str, cells: dict) -> dict:
    """E6b -- the main figure: does the mechanism track task or hint modality?"""
    print("\n" + "=" * 88)
    print(f"E6b -- two-way decomposition, arm = {arm}   [MAIN FIGURE]")
    print("=" * 88)
    within_hint, within_task = [], []
    for (t1, h1), (t2, h2) in itertools.combinations(sorted(cells), 2):
        j = jaccard(cells[(t1, h1)]["edges"], cells[(t2, h2)]["edges"])
        c1, c2 = ceiling_for(arm, t1, h1), ceiling_for(arm, t2, h2)
        half = (c1 + c2) / 2 if c1 == c1 and c2 == c2 else float("nan")
        jn = j / half if half and half == half and half > 0 else float("nan")
        if jn != jn:
            continue
        if h1 == h2 and t1 != t2:
            within_hint.append(jn)      # same hint, different domain
        elif t1 == t2 and h1 != h2:
            within_task.append(jn)      # same domain, different hint

    def m(v):
        return sum(v) / len(v) if v else float("nan")

    print(f"  within-hint, across-task  mean J_norm = {m(within_hint):.3f}  "
          f"(n={len(within_hint)})")
    print(f"  within-task, across-hint  mean J_norm = {m(within_task):.3f}  "
          f"(n={len(within_task)})")
    if within_hint and within_task:
        if m(within_task) > m(within_hint):
            print("  -> circuits cluster by HINT MODALITY: the mechanism survives a")
            print("     change of domain better than a change of hint format.")
        else:
            print("  -> circuits cluster by REASONING DOMAIN: the mechanism survives a")
            print("     change of hint format better than a change of domain.")
    else:
        print("  (need >=2 hint types AND >=2 tasks for this comparison)")
    return {"within_hint_across_task": m(within_hint),
            "within_task_across_hint": m(within_task)}


def report_separability(found: dict) -> dict:
    """E6c -- is influence the same circuit as suppression?"""
    print("\n" + "=" * 88)
    print("E6c -- separability: influence vs suppression (H3)")
    print("=" * 88)
    infl = found.get("influence", {})
    supp = found.get("suppression", {})
    shared = sorted(set(infl) & set(supp))
    if not shared:
        print("  no cell has both arms extracted yet"); return {}
    out = []
    print(f"{'cell':24} {'J(infl,supp)':>13} {'ceiling':>9} {'J_norm':>8}  verdict")
    print("-" * 70)
    for key in shared:
        t, h = key
        j = jaccard(infl[key]["edges"], supp[key]["edges"])
        c = ceiling_for("influence", t, h)
        jn = j / c if c == c and c > 0 else float("nan")
        verdict = "SEPARABLE" if (jn == jn and jn < 0.5) else "entangled"
        print(f"{t+'/'+h:24} {j:13.4f} {c:9.4f} {jn:8.3f}  {verdict}")
        out.append({"cell": f"{t}/{h}", "J": j, "ceiling": c, "J_norm": jn,
                    "separable": verdict == "SEPARABLE"})
    print("\n  SEPARABLE means unfaithfulness is TWO mechanisms, not one -- which is")
    print("  what makes E9 (ablate suppression, keep influence) meaningful.")
    return {"cells": out}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--p", type=int, default=60, help="aggregation threshold pct")
    ap.add_argument("--draws", type=int, default=1000)
    args = ap.parse_args()

    found = discover(args.p)
    if not found:
        print(f"No circuits found in {CIRCUITS}/ at p={args.p}. Run run_circuits.py first.")
        sys.exit(1)

    print(f"circuits found at p={args.p}:")
    for arm, cells in found.items():
        for (t, h), v in sorted(cells.items()):
            print(f"  {arm:12} {t:10} {h:10} {len(v['edges']):>8,} edges  "
                  f"(n={v['meta'].get('n_instances')})")

    # E5 floor
    print("\n" + "=" * 88)
    print("E5 -- task-solving floor")
    print("=" * 88)
    solve = found.get("solve", {})
    if len(solve) >= 2:
        for (t1, h1), (t2, h2) in itertools.combinations(sorted(solve), 2):
            j = jaccard(solve[(t1, h1)]["edges"], solve[(t2, h2)]["edges"])
            print(f"  {t1} <-> {t2}: J_solve = {j:.4f}")
        print("\n  Influence overlap must EXCEED this, or the 'shared unfaithfulness")
        print("  mechanism' is just the shared machinery for answering MC questions.")
    else:
        print("  not extracted yet -- run:  run_circuits.py --arm solve --cell <...>")

    out = {"p": args.p}
    for arm in ("influence", "suppression", "disclosure"):
        if arm in found and len(found[arm]) >= 2:
            out[arm] = report_arm(arm, found[arm], args)
            out[arm].update(report_two_way(arm, found[arm]))
    out["separability"] = report_separability(found)

    f = RESULTS / f"e5_e6_analysis_p{args.p}.json"
    f.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
    print(f"\nwrote {f}")


if __name__ == "__main__":
    main()
