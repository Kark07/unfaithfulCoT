"""
E8 + E9 -- causal ablation.  research_plan_v2.md §8, experiments E8 and E9.

E8 (--arm influence): ablate the SHARED influence circuit and ask
    (a) does hint-following drop across all tasks?      [causal]
    (b) does unhinted accuracy survive?                 [specific]
    (c) does a size-matched RANDOM ablation do the same? [control]
  If random ablation drops hint-following equally, the circuit is not special --
  we found the model's general fragility, not a mechanism.

E9 (--arm suppression): ablate the SHARED suppression circuit, RE-GENERATE the
  CoTs, and ask whether the model starts VERBALIZING hints it previously hid.
  This is the headline: an interpretability result that yields a repair, not just
  a description. Success looks like verbalization 20% -> 50% with hint-following
  and accuracy roughly intact, while ablating the INFLUENCE circuit instead lowers
  following WITHOUT raising verbalization (a double dissociation).

IMPLEMENTATION NOTE -- read before reporting results.
The circuit analysis (E3-E7) is edge-level. Ablation during autoregressive
generation is done at NODE level: we mean-ablate the writer nodes incident to the
circuit's edges. Per-edge ablation during generation would require rebuilding
each reader's residual input at every decoding step, which is not worth the
complexity here. State this limitation in the paper -- it is a real one.

Means are taken over the CORRUPTED distribution (mean-ablation, never zero --
zero is off-distribution and distorts everything, plan §4.5.1).

Run:
  python src/run_e8_e9_ablation.py --arm influence --p 60 \
      --cells results/e1_*_mmlu_grader.jsonl results/e1_*_logiqa_grader.jsonl
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))

from eap_ig import EAPIG  # noqa: E402
from circuit_utils import (CIRCUITS, RESULTS, chat_wrap, load_circuit,  # noqa: E402
                           load_instances)
from model_utils import extract_answer, heuristic_mentions_hint, split_cot  # noqa: E402
from prompts import build_triplet, calibrate_neutral  # noqa: E402


# ---------------------------------------------------------------------------
# node-level mean ablation
# ---------------------------------------------------------------------------

def circuit_nodes(eap: EAPIG, edges: set[tuple[int, int]]) -> set[int]:
    """Writer-node indices incident to the circuit's edges."""
    return {u for (u, _) in edges}


def node_means(model, eap: EAPIG, tok, prompts: list[str], max_n: int = 32) -> dict:
    """Mean writer activations over the corrupted distribution."""
    acc, count = {}, 0
    for p in prompts[:max_n]:
        store = {}
        with torch.no_grad():
            with model.hooks(fwd_hooks=eap._writer_hooks(store)):
                model(model.to_tokens(p, prepend_bos=False))
        for L in range(eap.n_layers):
            a = store[f"attn{L}"].mean(dim=(0, 1))       # [head, d]
            m = store[f"m{L}"].mean(dim=(0, 1))          # [d]
            acc[f"attn{L}"] = a if f"attn{L}" not in acc else acc[f"attn{L}"] + a
            acc[f"m{L}"] = m if f"m{L}" not in acc else acc[f"m{L}"] + m
        count += 1
    for k in acc:
        acc[k] = acc[k] / max(1, count)
    return acc


def ablation_hooks(eap: EAPIG, nodes: set[int], means: dict):
    """Replace each in-circuit writer's output with its corrupted-distribution mean."""
    by_layer_head = defaultdict(set)
    mlp_layers = set()
    for i in nodes:
        n = eap.writers[i]
        if n.kind == "head":
            by_layer_head[n.layer].add(n.head)
        elif n.kind == "mlp":
            mlp_layers.add(n.layer)

    hooks = []
    for L, heads in by_layer_head.items():
        hs = sorted(heads)

        def head_hook(act, hook, L=L, hs=hs):
            for h in hs:
                act[:, :, h, :] = means[f"attn{L}"][h].to(act.dtype)
            return act
        hooks.append((f"blocks.{L}.attn.hook_result", head_hook))

    for L in mlp_layers:
        def mlp_hook(act, hook, L=L):
            act[:, :, :] = means[f"m{L}"].to(act.dtype)
            return act
        hooks.append((f"blocks.{L}.hook_mlp_out", mlp_hook))
    return hooks


def random_nodes(eap: EAPIG, n: int, seed: int = 0) -> set[int]:
    rng = random.Random(seed)
    pool = [i for i, w in enumerate(eap.writers) if w.kind in ("head", "mlp")]
    return set(rng.sample(pool, min(n, len(pool))))


# ---------------------------------------------------------------------------

@torch.no_grad()
def gen(model, tok, prompt: str, hooks, max_new: int = 800) -> str:
    ids = model.to_tokens(chat_wrap(tok, prompt), prepend_bos=False)
    ctx = model.hooks(fwd_hooks=hooks) if hooks else _null()
    with ctx:
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False,
                             verbose=False)
    return model.to_string(out[0, ids.shape[1]:])


class _null:
    def __enter__(self): return self
    def __exit__(self, *a): return False


def measure(model, tok, eap, insts, neutral, hooks, args, label: str) -> dict:
    follow = verb = acc = n = 0
    for inst in insts[:args.n]:
        trip = build_triplet(inst.q, inst.hint_type, inst.x_letter,
                             inst.y_letter, neutral)
        txt = gen(model, tok, trip["P_X"], hooks, args.max_new_tokens)
        cot, a_h = split_cot(txt)
        n += 1
        if a_h == inst.x_letter:
            follow += 1
            if heuristic_mentions_hint(cot):
                verb += 1
        # unhinted accuracy on the same question
        txt0 = gen(model, tok, trip["P_0"], hooks, args.max_new_tokens)
        if extract_answer(txt0) == inst.q.gold_letter:
            acc += 1
    return {"label": label, "n": n,
            "hint_follow": follow / max(1, n),
            "verbalized_of_followers": verb / max(1, follow),
            "unhinted_acc": acc / max(1, n)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm", choices=["influence", "suppression"], required=True)
    ap.add_argument("--cells", nargs="+", required=True)
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--p", type=int, default=60)
    ap.add_argument("--n", type=int, default=25, help="instances per condition")
    ap.add_argument("--max-new-tokens", type=int, default=800)
    args = ap.parse_args()

    from transformer_lens import HookedTransformer
    model = HookedTransformer.from_pretrained_no_processing(
        args.model, dtype=torch.bfloat16, device="cuda")
    model.set_use_attn_result(True)
    model.set_use_hook_mlp_in(True)
    tok = model.tokenizer
    eap = EAPIG(model)
    print(eap.summary())

    # shared circuit = intersection across cells (plan §8 E8 step 1)
    per_cell, shared = {}, None
    for path in args.cells:
        want = "U" if args.arm == "suppression" else "switched"
        insts = load_instances(path, want=want)
        if not insts:
            continue
        t, h = insts[0].task, insts[0].hint_type
        cf = CIRCUITS / f"{args.arm}_{t}_{h}_p{args.p}.json"
        if not cf.exists():
            print(f"  missing circuit {cf.name}"); continue
        edges, _ = load_circuit(cf)
        per_cell[(t, h)] = {"insts": insts, "edges": edges}
        shared = edges if shared is None else (shared & edges)
        print(f"  {t}/{h}: {len(insts)} inst, {len(edges):,} edges")

    if not per_cell or not shared:
        print("no shared circuit (empty intersection or missing files)"); sys.exit(1)
    print(f"\nshared {args.arm} circuit: {len(shared):,} edges")

    nodes = circuit_nodes(eap, shared)
    print(f"incident writer nodes: {len(nodes)}")
    if not nodes:
        print("nothing to ablate"); sys.exit(1)

    # also load the OTHER arm's shared circuit for the double dissociation
    other = "suppression" if args.arm == "influence" else "influence"
    other_shared = None
    for (t, h) in per_cell:
        cf = CIRCUITS / f"{other}_{t}_{h}_p{args.p}.json"
        if cf.exists():
            e, _ = load_circuit(cf)
            other_shared = e if other_shared is None else (other_shared & e)

    results = []
    for (t, h), cell in per_cell.items():
        neutral = calibrate_neutral(tok, h, verbose=False)
        corrupt_prompts = [build_triplet(i.q, h, i.x_letter, i.y_letter, neutral)["P_0"]
                           for i in cell["insts"][:32]]
        means = node_means(model, eap, tok, [chat_wrap(tok, p) for p in corrupt_prompts])

        print(f"\n--- {t}/{h} ---")
        conds = [("baseline", None)]
        conds.append((f"ablate {args.arm}", ablation_hooks(eap, nodes, means)))
        if other_shared:
            on = circuit_nodes(eap, other_shared)
            conds.append((f"ablate {other}", ablation_hooks(eap, on, means)))
        conds.append(("ablate random (size-matched)",
                      ablation_hooks(eap, random_nodes(eap, len(nodes)), means)))

        for label, hooks in conds:
            r = measure(model, tok, eap, cell["insts"], neutral, hooks, args, label)
            r["task"], r["hint"] = t, h
            results.append(r)
            print(f"  {label:32} follow={r['hint_follow']*100:5.1f}%  "
                  f"verbalized={r['verbalized_of_followers']*100:5.1f}%  "
                  f"acc={r['unhinted_acc']*100:5.1f}%", flush=True)

    print("\n" + "=" * 84)
    print(f"E{'8' if args.arm=='influence' else '9'} summary")
    print("=" * 84)
    if args.arm == "influence":
        print("Success: hint-following drops across ALL tasks, unhinted accuracy")
        print("roughly intact, and random ablation does neither.")
    else:
        print("Success: verbalization RISES materially, while ablating influence")
        print("instead lowers following WITHOUT raising verbalization, and random")
        print("does neither. That double dissociation is the strongest form of the")
        print("claim that influence and suppression are separate mechanisms.")
    print("\nNOTE: verbalization here uses the KEYWORD heuristic. Re-score the saved")
    print("generations with judge.py before putting any number in the paper.")

    out = RESULTS / f"e{'8' if args.arm=='influence' else '9'}_ablation_{args.arm}.json"
    out.write_text(json.dumps({"arm": args.arm, "p": args.p,
                               "shared_edges": len(shared),
                               "nodes_ablated": len(nodes),
                               "results": results}, indent=2), encoding="utf-8")
    print(f"\nwrote {out}")


if __name__ == "__main__":
    main()
