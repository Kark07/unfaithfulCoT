"""
E7 -- cross-task faithfulness transfer.  research_plan_v2.md §8, experiment E7.

Overlap is a SET statistic and does not prove sufficiency. Hanna et al. (2024)
showed circuits can overlap heavily and still fail to transfer -- which is the
whole reason their paper exists and why plan §1 row 3 insists on measuring
transfer directly rather than inferring it from Jaccard.

Procedure: take the circuit found on task A, apply it to task B's instances
(mean-ablate everything outside it), and measure the fraction of task B's metric
it recovers. Fill an asymmetric NxN matrix; the diagonal is the within-task
faithfulness and is the upper bound for that column.

ASYMMETRY IS A FINDING. If the knowledge circuit explains math instances but not
the reverse, the knowledge circuit is a superset -- evidence for a core mechanism
plus task-specific extras. Discuss it; do not average it away.

Run:
  python src/run_e7_transfer.py --arm influence --p 60 \
      --cells results/e1_Qwen2.5-7B-Instruct_mmlu_grader.jsonl \
              results/e1_Qwen2.5-7B-Instruct_logiqa_grader.jsonl
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eap_ig import EAPIG  # noqa: E402
from circuit_utils import (CIRCUITS, RESULTS, load_circuit, load_instances,  # noqa: E402
                           metric_disc, metric_infl, pass_alpha, pass_gamma)
from prompts import calibrate_neutral  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["influence", "suppression"], default="influence")
    ap.add_argument("--cells", nargs="+", required=True, help="E1 jsonl files")
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--p", type=int, default=60)
    ap.add_argument("--n", type=int, default=25, help="test instances per cell")
    ap.add_argument("--tag", default="")
    args = ap.parse_args()

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=torch.bfloat16, device="cuda")
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    tok = model.tokenizer
    eap = EAPIG(model)
    print(eap.summary())

    # load circuits + test instances per cell
    cells = {}
    for path in args.cells:
        want = "U" if args.arm == "suppression" else "switched"
        insts = load_instances(path, want=want)[:args.n]
        if not insts:
            print(f"  skip {Path(path).name}: no instances"); continue
        t, h = insts[0].task, insts[0].hint_type
        cfile = CIRCUITS / f"{args.arm}_{t}_{h}_p{args.p}.json"
        if not cfile.exists():
            print(f"  skip {t}/{h}: no circuit at {cfile.name}"); continue
        edges, meta = load_circuit(cfile)
        cells[(t, h)] = {"insts": insts, "edges": edges,
                         "neutral": calibrate_neutral(tok, h, verbose=False)}
        print(f"  {t}/{h}: {len(insts)} test instances, {len(edges):,} edges")

    if len(cells) < 2:
        print("need >=2 cells with circuits"); sys.exit(1)

    keys = sorted(cells)
    matrix = {}
    for src in keys:
        for tgt in keys:
            vals = []
            for inst in cells[tgt]["insts"]:
                neutral = cells[tgt]["neutral"]
                if args.arm == "influence":
                    clean, corr = pass_alpha(model, tok, inst, neutral)
                    fn = metric_infl(model, inst)
                else:
                    clean, corr = pass_gamma(model, tok, inst, neutral)
                    fn = metric_disc(model)
                if clean.shape != corr.shape:
                    continue
                r = eap.faithfulness(clean, corr, fn, cells[src]["edges"])
                if r["faithfulness"] == r["faithfulness"]:
                    vals.append(r["faithfulness"])
            matrix[(src, tgt)] = sum(vals) / len(vals) if vals else float("nan")
            tag = " (diag)" if src == tgt else ""
            print(f"  circuit={src[0]:9} -> tested on {tgt[0]:9} "
                  f"faithfulness={matrix[(src,tgt)]:.3f}  n={len(vals)}{tag}", flush=True)

    print("\n" + "=" * 74)
    print(f"E7 transfer matrix -- arm={args.arm}  (rows = circuit, cols = tested on)")
    print("=" * 74)
    w = max(len(k[0]) for k in keys) + 2
    print(" " * w + "".join(f"{k[0]:>12}" for k in keys))
    for src in keys:
        row = "".join(f"{matrix[(src,tgt)]:12.3f}" for tgt in keys)
        print(f"{src[0]:<{w}}{row}")

    print("\nRead the OFF-DIAGONAL against the diagonal of its own column:")
    print("  near-diagonal  -> the source circuit is sufficient for the target task")
    print("  much lower     -> task-specific mechanism")
    print("  ASYMMETRIC     -> one circuit is a superset of the other. This is a")
    print("                    finding (core mechanism + task-specific extras).")

    out = RESULTS / f"e7_transfer_{args.arm}{args.tag}.json"
    out.write_text(json.dumps(
        {"arm": args.arm, "p": args.p, "model": args.model,
         "matrix": {f"{s[0]}/{s[1]}->{t[0]}/{t[1]}": v for (s, t), v in matrix.items()}},
        indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
