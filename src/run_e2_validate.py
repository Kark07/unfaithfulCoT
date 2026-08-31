"""
E2 -- Pipeline validation.  research_plan_v2.md §8, experiment E2.

Prove EAP-IG is correctly implemented BEFORE trusting it on novel behaviour.
Replaces v1's ACDC plan (§1 row 8: ACDC's ~5 h/query was GPT-2 Large; at 7B it is
a time sink, not a validation).

Two checks:

  1. IOI SANITY on GPT-2 small. The canonical circuit is known. If the name-mover
     heads (9.6, 9.9, 10.0) and S-inhibition heads (7.3, 7.9, 8.6, 8.10) do not
     surface in the top edges, the implementation is WRONG and every circuit
     result downstream would be silently garbage. This is the single most
     important test in the project.

  2. FAITHFULNESS-vs-SIZE on the real model. Sweep edge budget k, mean-ablate
     everything outside the circuit, measure the fraction of the metric
     recovered. Pick k* = smallest k reaching >= 85% (plan §4.4.3).

Run:
  python src/run_e2_validate.py --stage ioi
  python src/run_e2_validate.py --stage curve --model Qwen/Qwen2.5-7B-Instruct \
      --cell results/e1_Qwen2.5-7B-Instruct_mmlu_grader.jsonl --n 20
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eap_ig import EAPIG  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
RESULTS = ROOT / "results"

# Wang et al. 2023, IOI in GPT-2 small
IOI_NAME_MOVERS = {(9, 6), (9, 9), (10, 0)}
IOI_S_INHIBITION = {(7, 3), (7, 9), (8, 6), (8, 10)}
IOI_INDUCTION = {(5, 5), (6, 9), (5, 8), (5, 9)}

IOI_TEMPLATES = [
    ("When John and Mary went to the store, Mary gave a drink to", " John", " Mary"),
    ("When Alice and Bob went to the park, Bob gave a ball to", " Alice", " Bob"),
    ("After Tom and Sara arrived at the office, Sara handed a file to", " Tom", " Sara"),
    ("While Emma and Jack were at the cafe, Jack passed a menu to", " Emma", " Jack"),
    ("When Anna and Paul visited the museum, Paul showed a map to", " Anna", " Paul"),
    ("After Kate and Mike left the party, Mike gave a coat to", " Kate", " Mike"),
    ("When Lucy and Sam went to the beach, Sam brought a towel to", " Lucy", " Sam"),
    ("While Nina and Dave sat in the hall, Dave read a note to", " Nina", " Dave"),
]


def stage_ioi(args) -> None:
    from transformer_lens import HookedTransformer

    print("=" * 74)
    print("E2 stage 1 -- IOI sanity on GPT-2 small")
    print("=" * 74)
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    model = HookedTransformer.from_pretrained("gpt2-small", device=dev)
    model.set_use_attn_result(True)      # required: per-head residual writes
    model.set_use_hook_mlp_in(True)
    eap = EAPIG(model)
    print(eap.summary())

    all_scores = None
    for prompt, io_tok, s_tok in IOI_TEMPLATES:
        clean = model.to_tokens(prompt, prepend_bos=True)
        # corrupted: swap the subject so the correct answer flips (ABC pattern)
        corr_prompt = prompt.replace(s_tok.strip(), "Alex")
        corr = model.to_tokens(corr_prompt, prepend_bos=True)
        if corr.shape != clean.shape:
            continue
        t_io = model.to_single_token(io_tok)
        t_s = model.to_single_token(s_tok)

        def metric(logits):
            return logits[0, -1, t_io] - logits[0, -1, t_s]

        s = eap.attribute(clean, corr, metric, m=args.m)
        all_scores = s if all_scores is None else all_scores + s

    if all_scores is None:
        print("no usable IOI templates (tokenisation mismatch)"); return
    all_scores /= len(IOI_TEMPLATES)

    circuit = eap.top_k_edges(all_scores, k=args.k)
    heads_in_circuit = set()
    for (u, d) in circuit:
        for node in (eap.writers[u], eap.readers[d]):
            if node.kind == "head":
                heads_in_circuit.add((node.layer, node.head))

    def report(name, expected):
        hit = expected & heads_in_circuit
        print(f"  {name:16} {len(hit)}/{len(expected)}  {sorted(hit)}")
        return len(hit)

    print(f"\ntop-{args.k} edges touch {len(heads_in_circuit)} distinct heads")
    nm = report("name movers", IOI_NAME_MOVERS)
    si = report("S-inhibition", IOI_S_INHIBITION)
    report("induction", IOI_INDUCTION)

    ok = nm >= 2 and si >= 2
    print("\nVERDICT:", "PASS -- implementation recovers the known circuit" if ok
          else "FAIL -- EAP-IG is not finding IOI. DO NOT trust any circuit result.")
    if not ok:
        print("  Check: set_use_attn_result(True); hook names; the metric sign;")
        print("  whether clean/corrupted are token-aligned.")

    out = RESULTS / "e2_ioi_scores.pt"
    torch.save({"scores": all_scores, "heads": sorted(heads_in_circuit)}, out)
    print(f"  -> {out}")


def stage_curve(args) -> None:
    from transformer_lens import HookedTransformer
    from circuit_utils import (load_instances, metric_infl, pass_alpha)
    from prompts import calibrate_neutral

    print("=" * 74)
    print(f"E2 stage 2 -- faithfulness vs circuit size on {args.model}")
    print("=" * 74)
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=torch.bfloat16, device="cuda")
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    tok = model.tokenizer
    eap = EAPIG(model)
    print(eap.summary())

    insts = load_instances(args.cell, want="switched")[:args.n]
    print(f"{len(insts)} switched instances from {Path(args.cell).name}")
    neutral = calibrate_neutral(tok, insts[0].hint_type, verbose=False)

    ks = [int(x) for x in args.ks.split(",")]
    per_k: dict[int, list[float]] = {k: [] for k in ks}
    for i, inst in enumerate(insts, 1):
        clean, corr = pass_alpha(model, tok, inst, neutral)
        if clean.shape != corr.shape:
            continue
        fn = metric_infl(model, inst)
        scores = eap.attribute(clean, corr, fn, m=args.m)
        for k in ks:
            c = eap.top_k_edges(scores, k=k)
            r = eap.faithfulness(clean, corr, fn, c)
            if r["faithfulness"] == r["faithfulness"]:
                per_k[k].append(r["faithfulness"])
        if i % 5 == 0:
            print(f"  {i}/{len(insts)}", flush=True)

    print(f"\n{'k':>8} {'faithfulness':>14} {'n':>5}")
    print("-" * 30)
    kstar = None
    for k in ks:
        v = per_k[k]
        mean = sum(v) / len(v) if v else float("nan")
        print(f"{k:8} {mean:13.3f} {len(v):5}")
        if kstar is None and mean >= 0.85:
            kstar = k
    print(f"\nk* (smallest k reaching >=85%) = {kstar}")
    if kstar is None:
        print("  No k reached 85%. Widen --ks, or the metric/pass may be wrong.")

    out = RESULTS / "e2_faithfulness_curve.json"
    out.write_text(json.dumps(
        {"model": args.model, "cell": str(args.cell), "m": args.m,
         "curve": {str(k): (sum(v)/len(v) if v else None) for k, v in per_k.items()},
         "k_star": kstar}, indent=2), encoding="utf-8")
    print(f"  -> {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--stage", choices=["ioi", "curve"], required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--cell", default=None)
    ap.add_argument("--n", type=int, default=20)
    ap.add_argument("--m", type=int, default=5, help="IG integration steps")
    ap.add_argument("--k", type=int, default=2000, help="edge budget for IOI check")
    ap.add_argument("--ks", default="50,100,250,500,1000,2500,5000,10000")
    args = ap.parse_args()

    if args.stage == "ioi":
        stage_ioi(args)
    else:
        if not args.cell:
            print("--cell is required for the curve stage"); sys.exit(1)
        stage_curve(args)


if __name__ == "__main__":
    main()
