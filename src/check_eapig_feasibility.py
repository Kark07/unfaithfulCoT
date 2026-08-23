"""
Can we actually do circuit analysis on Qwen2.5-7B? Run this BEFORE E1.

This is the project's sharpest fork. E0 chose 7B because 1.5B does not yield
unfaithful instances (GSM8K 0%, LogiQA 0%). But EAP-IG memory scales with
(activations x edges), and 7B is ~4.5x the model. If EAP-IG cannot run at 7B we
are caught between a model that gives behaviour and a model that gives circuits,
and the design has to change -- so find out now, not on day 4.

Checks, cheapest first:
  1. does TransformerLens load Qwen2.5-7B at all
  2. how much VRAM the weights take, and what is left
  3. graph size (nodes, edges) -- drives EAP-IG memory
  4. a real forward+backward with gradients (the actual EAP-IG requirement)
  5. cache one full activation set and measure the peak

Run:  python src/check_eapig_feasibility.py --model Qwen/Qwen2.5-7B-Instruct
"""

from __future__ import annotations

import argparse
import sys
import traceback

import torch


def gb(x: float) -> str:
    return f"{x/1e9:.1f} GB"


def vram(tag: str) -> None:
    if torch.cuda.is_available():
        free, total = torch.cuda.mem_get_info()
        peak = torch.cuda.max_memory_allocated()
        print(f"    [{tag}] free={gb(free)} / {gb(total)}  peak_alloc={gb(peak)}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="Qwen/Qwen2.5-7B-Instruct")
    ap.add_argument("--dtype", default="bfloat16")
    ap.add_argument("--seq-len", type=int, default=256,
                    help="representative prompt length (E0 medians were 94-217)")
    args = ap.parse_args()

    if not torch.cuda.is_available():
        print("No CUDA. Run this on the GPU server.")
        sys.exit(1)

    print("=" * 70)
    print(f"EAP-IG FEASIBILITY -- {args.model}")
    print("=" * 70)
    free, total = torch.cuda.mem_get_info()
    print(f"GPU: {torch.cuda.get_device_name(0)}  free={gb(free)} / {gb(total)}")
    if free / 1e9 < 20:
        print("  !! under 20 GB free -- another user may be on this GPU. "
              "Check nvidia-smi and pick a free device before trusting these numbers.")

    # ---- 1. does TransformerLens load it -------------------------------
    print("\n[1] loading via TransformerLens ...")
    try:
        from transformer_lens import HookedTransformer
    except Exception as e:
        print(f"  FAIL: transformer_lens not importable ({type(e).__name__}: {e})")
        print("  -> pip install transformer_lens")
        sys.exit(1)

    dtype = {"bfloat16": torch.bfloat16, "float32": torch.float32}[args.dtype]
    try:
        model = HookedTransformer.from_pretrained_no_processing(
            args.model, dtype=dtype, device="cuda")
    except Exception:
        print("  from_pretrained_no_processing failed; trying from_pretrained ...")
        try:
            model = HookedTransformer.from_pretrained(args.model, dtype=dtype, device="cuda")
        except Exception:
            print("  FAIL: TransformerLens cannot load this model.")
            traceback.print_exc(limit=3)
            print("\n  -> Options: (a) use the HF model with a hand-rolled EAP-IG,")
            print("               (b) drop to a TL-supported model of similar scale,")
            print("               (c) check the TL version supports Qwen2.")
            sys.exit(1)
    print("  OK")
    vram("after load")

    # ---- 2/3. graph size ------------------------------------------------
    cfg = model.cfg
    n_layers, n_heads = cfg.n_layers, cfg.n_heads
    n_nodes = n_layers * n_heads + n_layers + 1          # heads + MLPs + embed
    # EAP edges: every upstream node writes to every downstream node's input
    n_edges = 0
    upstream = 1
    for _ in range(n_layers):
        n_edges += upstream * n_heads * 3   # q, k, v inputs
        upstream += n_heads
        n_edges += upstream                 # mlp input
        upstream += 1
    n_edges += upstream                     # unembed
    print(f"\n[2] graph: {n_layers} layers x {n_heads} heads, d_model={cfg.d_model}")
    print(f"    nodes ~ {n_nodes:,}   edges ~ {n_edges:,}")
    scores_gb = n_edges * 4 / 1e9
    print(f"    edge-score tensor (fp32): {scores_gb:.3f} GB  (small -- not the bottleneck)")

    # ---- 4. forward + backward with grads -------------------------------
    print("\n[4] forward + backward with gradients (the real EAP-IG requirement) ...")
    torch.cuda.reset_peak_memory_stats()
    ids = torch.randint(0, cfg.d_vocab, (1, args.seq_len), device="cuda")
    try:
        logits = model(ids)
        loss = logits[0, -1].sum()
        loss.backward()
        print("  OK -- gradients flow")
        vram("after backward")
    except torch.cuda.OutOfMemoryError:
        print(f"  OOM at seq_len={args.seq_len}")
        print("  -> try shorter prompts, gradient checkpointing, or a smaller model")
        sys.exit(1)
    except Exception as e:
        print(f"  FAIL: {type(e).__name__}: {e}")
        traceback.print_exc(limit=3)
        sys.exit(1)
    model.zero_grad(set_to_none=True)

    # ---- 5. full activation cache --------------------------------------
    print("\n[5] caching a full activation set (EAP-IG holds clean + corrupted) ...")
    torch.cuda.reset_peak_memory_stats()
    try:
        _, cache = model.run_with_cache(ids)
        n_act = sum(v.numel() for v in cache.values())
        bytes_per = 2 if dtype is torch.bfloat16 else 4
        print(f"  OK -- {len(cache)} tensors, {n_act:,} elements "
              f"({gb(n_act*bytes_per)} per cache)")
        print(f"  EAP-IG needs ~2 caches + grads: est. {gb(n_act*bytes_per*3)}")
        vram("after cache")
        del cache
    except torch.cuda.OutOfMemoryError:
        print("  OOM caching activations.")
        print("  -> EAP-IG at this scale needs offloading, shorter prompts, or a smaller model")
        sys.exit(1)

    torch.cuda.empty_cache()
    free_after, _ = torch.cuda.mem_get_info()
    print("\n" + "=" * 70)
    print("VERDICT: TransformerLens loads, gradients flow, activations cache.")
    print(f"         {gb(free_after)} free after teardown.")
    print("         EAP-IG at 7B looks feasible -- proceed to E2 proper (IOI sanity")
    print("         on GPT-2 small, then faithfulness-vs-size curve on this model).")
    print("=" * 70)


if __name__ == "__main__":
    main()
