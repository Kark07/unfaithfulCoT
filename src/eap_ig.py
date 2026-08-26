"""
EAP-IG: Edge Attribution Patching with Integrated Gradients (Hanna et al., COLM 2024).

Implements research_plan_v2.md §4.4 / §6.5.

THE KEY IDENTITY
----------------
In a residual-stream transformer every downstream component reads the SUM of all
upstream writes. So the gradient of the metric w.r.t. "the edge from upstream u
into downstream d" is just the gradient w.r.t. d's INPUT -- identical for every
u. That collapses a per-edge computation into an outer product:

    score[u, d] = (z_u^clean - z_u^corr) . g_d

with z_u the upstream node's output (d_model) and g_d the gradient w.r.t. the
downstream node's input (d_model). One cache of deltas, one cache of gradients,
one matmul -- roughly a million edges scored in a few passes.

EAP-IG replaces the single gradient with an average taken along a straight line
from corrupted to clean activations, which fixes the gradient saturation that
makes plain EAP unfaithful:

    score[u, d] = (z_u^clean - z_u^corr) . (1/m) sum_k g_d(x_corr + (k/m)(x_clean - x_corr))

Causal masking: a node only reads from nodes strictly before it. We build an
upper-triangular mask over (upstream, downstream) in topological order.

Usage
-----
    from eap_ig import EAPIG
    eap = EAPIG(model)
    scores = eap.attribute(clean_toks, corr_toks, metric_fn, m=5)
    circuit = eap.top_k_edges(scores, k=2000)
    faith  = eap.faithfulness(clean_toks, corr_toks, metric_fn, circuit)
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class Node:
    """One writer into the residual stream, or one reader from it."""
    name: str          # "input", "a3.h7", "m12", "logits"
    kind: str          # "input" | "head" | "mlp" | "logits"
    layer: int
    head: int | None
    order: int         # topological position; u may feed d iff u.order < d.order


class EAPIG:
    def __init__(self, model):
        """model: a TransformerLens HookedTransformer."""
        self.model = model
        cfg = model.cfg
        self.n_layers, self.n_heads = cfg.n_layers, cfg.n_heads
        self.d_model = cfg.d_model
        self._build_nodes()

    # -- graph -----------------------------------------------------------
    def _build_nodes(self) -> None:
        writers: list[Node] = [Node("input", "input", -1, None, 0)]
        order = 1
        for L in range(self.n_layers):
            for H in range(self.n_heads):
                writers.append(Node(f"a{L}.h{H}", "head", L, H, order)); order += 1
            writers.append(Node(f"m{L}", "mlp", L, None, order)); order += 1
        self.writers = writers

        readers: list[Node] = []
        order = 1
        for L in range(self.n_layers):
            for H in range(self.n_heads):
                readers.append(Node(f"a{L}.h{H}", "head", L, H, order)); order += 1
            readers.append(Node(f"m{L}", "mlp", L, None, order)); order += 1
        readers.append(Node("logits", "logits", self.n_layers, None, order))
        self.readers = readers

        # causal mask: writer w can feed reader r iff w.order < r.order
        W, R = len(self.writers), len(self.readers)
        mask = torch.zeros(W, R, dtype=torch.bool)
        for i, w in enumerate(self.writers):
            for j, r in enumerate(self.readers):
                if w.order < r.order:
                    mask[i, j] = True
        self.edge_mask = mask
        self.n_edges = int(mask.sum())

    def summary(self) -> str:
        return (f"{self.n_layers}L x {self.n_heads}H d={self.d_model} | "
                f"{len(self.writers)} writers, {len(self.readers)} readers, "
                f"{self.n_edges:,} edges")

    # -- hooks -----------------------------------------------------------
    def _writer_hooks(self, store: dict):
        """Cache each writer's contribution to the residual stream."""
        hooks = []

        def emb_hook(act, hook):
            store["input"] = act.detach()
            return act
        hooks.append(("hook_embed", emb_hook))

        for L in range(self.n_layers):
            def head_hook(act, hook, L=L):
                # act: [batch, pos, head, d_model] -- per-head result
                store[f"attn{L}"] = act.detach()
                return act
            hooks.append((f"blocks.{L}.attn.hook_result", head_hook))

            def mlp_hook(act, hook, L=L):
                store[f"m{L}"] = act.detach()
                return act
            hooks.append((f"blocks.{L}.hook_mlp_out", mlp_hook))
        return hooks

    def _reader_grad_hooks(self, store: dict):
        """Cache gradients w.r.t. each reader's INPUT (its residual-stream read)."""
        hooks = []
        for L in range(self.n_layers):
            def pre_hook(act, hook, L=L):
                act.retain_grad()
                store[f"resid_pre{L}"] = act
                return act
            hooks.append((f"blocks.{L}.hook_resid_pre", pre_hook))

            def mid_hook(act, hook, L=L):
                act.retain_grad()
                store[f"resid_mid{L}"] = act
                return act
            hooks.append((f"blocks.{L}.hook_resid_mid", mid_hook))

        def post_hook(act, hook):
            act.retain_grad()
            store["resid_final"] = act
            return act
        hooks.append((f"blocks.{self.n_layers-1}.hook_resid_post", post_hook))
        return hooks

    # -- flatten caches into [n_writers, d] / [n_readers, d] --------------
    def _stack_writers(self, store: dict, pos: int) -> torch.Tensor:
        rows = [store["input"][0, pos]]
        for L in range(self.n_layers):
            attn = store[f"attn{L}"][0, pos]          # [head, d_model]
            rows.extend(attn[H] for H in range(self.n_heads))
            rows.append(store[f"m{L}"][0, pos])
        return torch.stack(rows)                       # [W, d_model]

    def _stack_reader_grads(self, store: dict, pos: int) -> torch.Tensor:
        rows = []
        for L in range(self.n_layers):
            gpre = store[f"resid_pre{L}"].grad
            gmid = store[f"resid_mid{L}"].grad
            gpre = torch.zeros(self.d_model, device=self._dev) if gpre is None else gpre[0, pos]
            gmid = torch.zeros(self.d_model, device=self._dev) if gmid is None else gmid[0, pos]
            for _ in range(self.n_heads):
                rows.append(gpre)                      # all heads in L read resid_pre
            rows.append(gmid)                          # MLP reads resid_mid
        gfin = store["resid_final"].grad
        rows.append(torch.zeros(self.d_model, device=self._dev) if gfin is None else gfin[0, pos])
        return torch.stack(rows)                       # [R, d_model]

    # -- main entry point -------------------------------------------------
    def attribute(self, clean_toks, corr_toks, metric_fn, m: int = 5,
                  pos: int = -1) -> torch.Tensor:
        """Return an [n_writers, n_readers] edge-score tensor.

        metric_fn(logits) -> scalar. Convention: HIGHER means more of the clean
        behaviour, so a positive score means the edge carries clean->corrupted
        difference toward the behaviour.
        """
        model = self.model
        self._dev = next(model.parameters()).device

        # 1. writer activations, clean and corrupted (no grad needed)
        clean_store, corr_store = {}, {}
        with torch.no_grad():
            with model.hooks(fwd_hooks=self._writer_hooks(corr_store)):
                model(corr_toks)
            with model.hooks(fwd_hooks=self._writer_hooks(clean_store)):
                model(clean_toks)
        z_clean = self._stack_writers(clean_store, pos)
        z_corr = self._stack_writers(corr_store, pos)
        delta = (z_clean - z_corr).float()              # [W, d]
        del clean_store, corr_store

        # 2. cache the corrupted embedding once -- it anchors the IG path
        emb_corr = {}

        def _grab_emb(act, hook):
            emb_corr["e"] = act.detach()
            return act

        with torch.no_grad():
            with model.hooks(fwd_hooks=[("hook_embed", _grab_emb)]):
                model(corr_toks)

        # 3. integrated gradients along corrupted -> clean.
        #    Interpolating at the EMBEDDING makes every downstream activation a
        #    smooth function of alpha, which is what removes the saturation that
        #    makes plain EAP (alpha=1 only) unfaithful.
        grads = torch.zeros(len(self.readers), self.d_model,
                            device=self._dev, dtype=torch.float32)
        for k in range(1, m + 1):
            alpha = k / m
            gstore: dict = {}

            def emb_interp(act, hook, alpha=alpha):
                return emb_corr["e"] + alpha * (act - emb_corr["e"])

            model.zero_grad(set_to_none=True)
            with model.hooks(fwd_hooks=[("hook_embed", emb_interp)]
                             + self._reader_grad_hooks(gstore)):
                logits = model(clean_toks)
                metric_fn(logits).backward()
            grads += self._stack_reader_grads(gstore, pos).float()
            del gstore
        grads /= m

        # 4. outer product, causal mask
        scores = delta @ grads.T                        # [W, R]
        scores = scores * self.edge_mask.to(scores.device)
        return scores.detach().cpu()

    # -- circuit construction --------------------------------------------
    def top_k_edges(self, scores: torch.Tensor, k: int,
                    signed: str = "abs") -> set[tuple[int, int]]:
        """Top-k edges. signed='abs' | 'pos' | 'neg' (neg = suppression, §6.5)."""
        s = scores.clone()
        if signed == "abs":
            s = s.abs()
        elif signed == "pos":
            s = s.clamp(min=0)
        elif signed == "neg":
            s = (-s).clamp(min=0)
        else:
            raise ValueError(signed)
        flat = s.flatten()
        k = min(k, int((flat > 0).sum()))
        if k == 0:
            return set()
        idx = torch.topk(flat, k).indices
        R = scores.shape[1]
        return {(int(i // R), int(i % R)) for i in idx}

    def edge_name(self, u: int, d: int) -> str:
        return f"{self.writers[u].name}->{self.readers[d].name}"

    # -- faithfulness -----------------------------------------------------
    def faithfulness(self, clean_toks, corr_toks, metric_fn,
                     circuit: set[tuple[int, int]], pos: int = -1) -> dict:
        """Fraction of the clean-vs-corrupted metric gap the circuit recovers.

        Sufficiency test: run on the CORRUPTED input, then add back only the
        in-circuit writers' clean contributions to each reader's residual input.
        Everything outside the circuit stays at its corrupted value, which is
        mean-ablation against the corrupted distribution (plan §4.5.1 -- never
        zero-ablation, which is off-distribution).

            faithfulness = (m_circuit - m_corr) / (m_clean - m_corr)

        1.0 = the circuit fully explains the behaviour; 0.0 = it explains none.
        """
        model = self.model
        self._dev = next(model.parameters()).device

        clean_store, corr_store = {}, {}
        with torch.no_grad():
            with model.hooks(fwd_hooks=self._writer_hooks(corr_store)):
                logits_corr = model(corr_toks)
            with model.hooks(fwd_hooks=self._writer_hooks(clean_store)):
                logits_clean = model(clean_toks)
            m_corr = float(metric_fn(logits_corr))
            m_clean = float(metric_fn(logits_clean))

        z_clean = self._stack_writers(clean_store, pos)
        z_corr = self._stack_writers(corr_store, pos)
        delta = z_clean - z_corr                        # [W, d]

        # per-reader patch vector: sum of in-circuit upstream deltas
        patch = torch.zeros(len(self.readers), self.d_model,
                            device=delta.device, dtype=delta.dtype)
        for (u, d) in circuit:
            patch[d] += delta[u]

        # readers grouped by which residual point they read
        def reader_idx(L: int, head: bool) -> int:
            base = L * (self.n_heads + 1)
            return base if head else base + self.n_heads

        hooks = []
        for L in range(self.n_layers):
            def pre(act, hook, L=L):
                # all heads in L read resid_pre; they share one patch slot group
                add = patch[reader_idx(L, True): reader_idx(L, True) + self.n_heads].sum(0)
                act[0, pos] = act[0, pos] + add
                return act
            hooks.append((f"blocks.{L}.hook_resid_pre", pre))

            def mid(act, hook, L=L):
                act[0, pos] = act[0, pos] + patch[reader_idx(L, False)]
                return act
            hooks.append((f"blocks.{L}.hook_resid_mid", mid))

        def post(act, hook):
            act[0, pos] = act[0, pos] + patch[-1]
            return act
        hooks.append((f"blocks.{self.n_layers-1}.hook_resid_post", post))

        with torch.no_grad():
            with model.hooks(fwd_hooks=hooks):
                m_circ = float(metric_fn(model(corr_toks)))

        gap = m_clean - m_corr
        return {
            "m_clean": m_clean, "m_corr": m_corr, "m_circuit": m_circ,
            "faithfulness": (m_circ - m_corr) / gap if abs(gap) > 1e-8 else float("nan"),
            "n_edges": len(circuit),
        }

    @staticmethod
    def jaccard(a: set, b: set) -> float:
        if not a and not b:
            return float("nan")
        return len(a & b) / len(a | b)
