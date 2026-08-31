"""
E3 + E4 -- circuit extraction.  research_plan_v2.md §8, experiments E3 and E4.

One script, two arms, because they differ only in the counterfactual and the
metric (plan §6.0 -- the corrupted input IS the question you are asking):

  --arm influence   E3.  clean=P_X  corrupted=P_Y   metric = logit(X) - logit(Y)
                    Varies WHICH LETTER the hint names, holding hint presence
                    fixed. Recovers the machinery routing the hinted letter to
                    the answer. Runs on all switched instances (U u F) -- does
                    NOT need the judge.

  --arm disclosure  E4.  clean=P_X  corrupted=P_0   metric = logit(Yes)-logit(No)
                    Varies HINT PRESENCE, holding length fixed. Recovers the
                    machinery carrying "a hint is here" to the self-report.
                    SIGN MATTERS: edges with negative attribution push toward
                    "No" despite a hint being present -- those are the
                    SUPPRESSION circuit, and they are the paper's subject.
                    Needs the U/F split, so run judge.py first.

  --arm solve       E5 floor. No hint either side; options shuffled in corrupted.
                    The plain task-solving circuit any "shared unfaithfulness
                    mechanism" must beat.

Outputs per-instance circuits (for E5's split-half ceiling) and the aggregated
task circuit (for E6/E7/E8/E9).

Run:
  python src/run_circuits.py --arm influence \
      --cell results/e1_Qwen2.5-7B-Instruct_mmlu_grader.jsonl --k 2000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eap_ig import EAPIG  # noqa: E402
from circuit_utils import (CIRCUITS, aggregate, load_instances,  # noqa: E402
                           metric_disc, metric_gold, metric_infl, pass_alpha,
                           pass_gamma, pass_solve, save_circuit, split_half_ceiling)
from prompts import calibrate_neutral  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["influence", "disclosure", "solve"], required=True)
    ap.add_argument("--cell", required=True, help="an E1 results jsonl")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--k", type=int, default=2000, help="k* from E2")
    ap.add_argument("--m", type=int, default=5, help="IG steps")
    ap.add_argument("--p", type=float, default=0.60, help="aggregation threshold")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--which", default=None,
                    help="instance set: U | F | R | switched (default depends on arm)")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--save-scores", action="store_true",
                    help="also save the mean score tensor (needed by E5 null)")
    args = ap.parse_args()

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=torch.bfloat16, device="cuda")
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    tok = model.tokenizer
    eap = EAPIG(model)
    print(eap.summary())

    # influence works on all switched instances; disclosure needs the U set
    want = args.which or ("U" if args.arm == "disclosure" else "switched")
    insts = load_instances(args.cell, want=want)
    if args.limit:
        insts = insts[:args.limit]
    if not insts:
        print(f"No instances with want={want} in {args.cell}"); sys.exit(1)

    task = insts[0].task
    hint = insts[0].hint_type
    neutral = calibrate_neutral(tok, hint, verbose=False)
    print(f"{len(insts)} instances | task={task} hint={hint} arm={args.arm} want={want}")

    import random
    rng = random.Random(0)

    inst_circuits: list[set] = []
    inst_supp: list[set] = []
    mean_scores = None
    skipped = 0
    t0 = time.time()

    for i, inst in enumerate(insts, 1):
        if args.arm == "influence":
            clean, corr = pass_alpha(model, tok, inst, neutral)
            fn = metric_infl(model, inst)
        elif args.arm == "disclosure":
            clean, corr = pass_gamma(model, tok, inst, neutral)
            fn = metric_disc(model)
        else:
            clean, corr = pass_solve(model, tok, inst, neutral, rng)
            fn = metric_gold(model, inst)

        if clean.shape != corr.shape:
            skipped += 1
            continue

        scores = eap.attribute(clean, corr, fn, m=args.m)
        mean_scores = scores if mean_scores is None else mean_scores + scores

        inst_circuits.append(eap.top_k_edges(scores, k=args.k, signed="abs"))
        if args.arm == "disclosure":
            # negative attribution = pushes toward "No" despite the hint being
            # present = actively suppressing disclosure (plan §6.5)
            inst_supp.append(eap.top_k_edges(scores, k=args.k // 2, signed="neg"))

        if i % 10 == 0 or i == len(insts):
            el = time.time() - t0
            print(f"  {i}/{len(insts)}  ({el:.0f}s, {el/i:.2f}s/inst)", flush=True)

    if not inst_circuits:
        print("nothing extracted (all shape-mismatched?)"); sys.exit(1)
    mean_scores /= len(inst_circuits)
    print(f"skipped {skipped} on shape mismatch")

    prefix = args.out_prefix or f"{args.arm}_{task}_{hint}"
    CIRCUITS.mkdir(parents=True, exist_ok=True)

    # aggregate at several p; report p=0.60 in the main text (plan §4.4.3)
    for p in (0.50, 0.60, 0.70):
        c = aggregate(inst_circuits, p)
        save_circuit(CIRCUITS / f"{prefix}_p{int(p*100)}.json", c,
                     {"arm": args.arm, "task": task, "hint": hint, "p": p,
                      "k": args.k, "m": args.m, "n_instances": len(inst_circuits),
                      "want": want, "model": args.model})
        print(f"  p={p:.2f}  |circuit| = {len(c):,} edges")

    if inst_supp:
        for p in (0.50, 0.60, 0.70):
            c = aggregate(inst_supp, p)
            save_circuit(CIRCUITS / f"suppression_{task}_{hint}_p{int(p*100)}.json", c,
                         {"arm": "suppression", "task": task, "hint": hint, "p": p,
                          "k": args.k // 2, "n_instances": len(inst_supp),
                          "want": want, "model": args.model})
            print(f"  suppression p={p:.2f}  |circuit| = {len(c):,} edges")

    # per-instance circuits: E5's split-half ceiling needs them
    (CIRCUITS / f"{prefix}_instances.json").write_text(
        json.dumps([sorted([int(u), int(d)] for u, d in c) for c in inst_circuits]),
        encoding="utf-8")

    ceil = split_half_ceiling(inst_circuits, p=args.p)
    print(f"\nsplit-half ceiling (p={args.p}): {ceil:.3f}")
    print("  Cross-task Jaccard cannot meaningfully exceed this. E6 divides by it.")

    if args.save_scores:
        torch.save(mean_scores, CIRCUITS / f"{prefix}_scores.pt")
        print(f"  -> {prefix}_scores.pt (for E5 permutation null)")

    (CIRCUITS / f"{prefix}_summary.json").write_text(json.dumps({
        "arm": args.arm, "task": task, "hint": hint, "want": want,
        "n_instances": len(inst_circuits), "skipped": skipped,
        "k": args.k, "m": args.m,
        "split_half_ceiling": ceil,
        "sizes": {f"p{int(p*100)}": len(aggregate(inst_circuits, p))
                  for p in (0.50, 0.60, 0.70)},
    }, indent=2), encoding="utf-8")
    print(f"\nwrote circuits to {CIRCUITS}/")


if __name__ == "__main__":
    main()
