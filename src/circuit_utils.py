"""
Shared plumbing for the circuit experiments (E2-E9).

Everything the E-scripts need in common:
  * load E1 records and rebuild the token-aligned prompt triplets
  * build the two scoring passes and their metrics (plan §6.5)
  * save / load circuits as JSON
  * aggregate per-instance circuits into a task circuit (plan §4.4.3)
  * Jaccard, normalized overlap, score-matched permutation null (plan §8 E5)
"""

from __future__ import annotations

import json
import random
from dataclasses import dataclass
from pathlib import Path

import torch

from prompts import (DISCLOSURE_PROBE, Question, build_triplet,  # noqa: E402
                     calibrate_neutral)

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data" / "processed"
RESULTS = ROOT / "results"
CIRCUITS = ROOT / "circuits"
ANSWER_CUE = "Answer: ("


# ---------------------------------------------------------------------------
# E1 records
# ---------------------------------------------------------------------------

@dataclass
class Instance:
    """One switched instance, with everything needed to rebuild its prompts."""
    qid: str
    task: str
    hint_type: str
    q: Question
    x_letter: str
    y_letter: str
    a_h: str
    cot: str
    verbalized: bool | None      # from judge.py if present, else heuristic
    label: str                   # "U" | "F" | "R"


def load_questions(task: str) -> dict[str, Question]:
    rows = [json.loads(l) for l in
            (DATA / f"{task}.jsonl").read_text(encoding="utf-8").splitlines()]
    return {r["qid"]: Question(r["qid"], r["task"], r["stem"],
                               r["options"], r["gold_idx"], r.get("meta", {}))
            for r in rows}


def load_instances(path: str | Path, want: str = "switched",
                   prefer_judge: bool = True) -> list[Instance]:
    """Load E1 records.

    want: "switched" (U u F) | "U" | "F" | "R"
    prefer_judge: use judge.py's label when present, else the keyword heuristic.
                  Heuristic labels are provisional (F12, F21) -- never publish them.
    """
    recs = [json.loads(l) for l in
            Path(path).read_text(encoding="utf-8").splitlines()]
    qcache: dict[str, dict[str, Question]] = {}
    out = []
    for r in recs:
        if r.get("switched") is None:
            continue
        task = r["task"]
        if task not in qcache:
            qcache[task] = load_questions(task)
        q = qcache[task].get(r["qid"])
        if q is None:
            continue

        if r["switched"]:
            j = r.get("judge")
            if prefer_judge and isinstance(j, dict) and "verbalized" in j:
                verb = bool(j["verbalized"])
            else:
                verb = r.get("mentions_hint_heuristic")
            label = "F" if verb else "U"
        else:
            verb, label = None, "R"

        if want == "switched" and label == "R":
            continue
        if want in ("U", "F", "R") and label != want:
            continue
        if not r.get("cot") or not r.get("x_letter"):
            continue

        out.append(Instance(r["qid"], task, r["hint_type"], q,
                            r["x_letter"], r["y_letter"], r.get("a_h") or r["x_letter"],
                            r["cot"], verb, label))
    return out


# ---------------------------------------------------------------------------
# Scoring passes (plan §6.5)
# ---------------------------------------------------------------------------

def chat_wrap(tok, prompt: str) -> str:
    return tok.apply_chat_template([{"role": "user", "content": prompt}],
                                   tokenize=False, add_generation_prompt=True)


def _toks(model, text: str) -> torch.Tensor:
    return model.to_tokens(text, prepend_bos=False)


def pass_alpha(model, tok, inst: Instance, neutral: str
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Influence pass. clean = P_X, corrupted = P_Y. Both + 'Answer: ('."""
    trip = build_triplet(inst.q, inst.hint_type, inst.x_letter, inst.y_letter, neutral)
    clean = chat_wrap(tok, trip["P_X"]) + ANSWER_CUE
    corr = chat_wrap(tok, trip["P_Y"]) + ANSWER_CUE
    return _toks(model, clean), _toks(model, corr)


def pass_gamma(model, tok, inst: Instance, neutral: str
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Disclosure pass. clean = P_X, corrupted = P_0. CoT and answer held fixed."""
    trip = build_triplet(inst.q, inst.hint_type, inst.x_letter, inst.y_letter, neutral)
    tail = f"{inst.cot}\n\nAnswer: ({inst.a_h})\n\n{DISCLOSURE_PROBE}"
    clean = chat_wrap(tok, trip["P_X"]) + tail
    corr = chat_wrap(tok, trip["P_0"]) + tail
    return _toks(model, clean), _toks(model, corr)


def pass_solve(model, tok, inst: Instance, neutral: str, rng: random.Random
               ) -> tuple[torch.Tensor, torch.Tensor]:
    """Plain task-solving pass -- the E5 floor. No hint on either side.

    clean = P_0 with the real options; corrupted = P_0 with options shuffled.
    Measures the machinery for answering an MC question at all, which is the
    baseline any 'shared unfaithfulness circuit' must beat (plan §8 E5b).
    """
    trip = build_triplet(inst.q, inst.hint_type, inst.x_letter, inst.y_letter, neutral)
    shuffled = list(inst.q.options)
    rng.shuffle(shuffled)
    qs = Question(inst.q.qid, inst.q.task, inst.q.stem, shuffled, inst.q.gold_idx)
    trip_s = build_triplet(qs, inst.hint_type, inst.x_letter, inst.y_letter, neutral)
    clean = chat_wrap(tok, trip["P_0"]) + ANSWER_CUE
    corr = chat_wrap(tok, trip_s["P_0"]) + ANSWER_CUE
    return _toks(model, clean), _toks(model, corr)


def metric_letters(model, a: str, b: str):
    """M = logit(a) - logit(b) at the final position."""
    ta = model.to_single_token(a)
    tb = model.to_single_token(b)

    def fn(logits):
        return logits[0, -1, ta] - logits[0, -1, tb]
    return fn


def metric_infl(model, inst: Instance):
    """M_infl = logit(X) - logit(Y)."""
    return metric_letters(model, inst.x_letter, inst.y_letter)


def metric_disc(model):
    """M_disc = logit('A'=Yes) - logit('B'=No)."""
    return metric_letters(model, "A", "B")


def metric_gold(model, inst: Instance):
    """Task-solving metric: logit(gold) - logit(best distractor)."""
    wrong = [L for L in inst.q.incorrect_letters()]
    return metric_letters(model, inst.q.gold_letter, wrong[0])


# ---------------------------------------------------------------------------
# Circuit I/O
# ---------------------------------------------------------------------------

def save_circuit(path: str | Path, edges: set[tuple[int, int]], meta: dict) -> None:
    p = Path(path); p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps({"meta": meta,
                             "edges": sorted([int(u), int(d)] for u, d in edges)}),
                 encoding="utf-8")


def load_circuit(path: str | Path) -> tuple[set[tuple[int, int]], dict]:
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return {(int(u), int(v)) for u, v in d["edges"]}, d.get("meta", {})


def aggregate(instance_circuits: list[set[tuple[int, int]]], p: float = 0.60
              ) -> set[tuple[int, int]]:
    """Task circuit = edges present in at least p of the instance circuits.

    plan §4.4.3. Sweep p in {0.5, 0.6, 0.7}; report 0.60 in the main text and do
    NOT pick p after seeing overlap results.
    """
    if not instance_circuits:
        return set()
    counts: dict[tuple[int, int], int] = {}
    for c in instance_circuits:
        for e in c:
            counts[e] = counts.get(e, 0) + 1
    need = p * len(instance_circuits)
    return {e for e, c in counts.items() if c >= need}


# ---------------------------------------------------------------------------
# Overlap statistics (plan §8 E5/E6)
# ---------------------------------------------------------------------------

def jaccard(a: set, b: set) -> float:
    if not a and not b:
        return float("nan")
    return len(a & b) / len(a | b)


def split_half_ceiling(instance_circuits: list[set], p: float = 0.60,
                       n_splits: int = 5, seed: int = 0) -> float:
    """Noise ceiling: two circuits built from disjoint halves of the SAME cell.

    Cross-task overlap cannot meaningfully exceed this. Without it a raw Jaccard
    is uninterpretable (plan §1 row 3).
    """
    rng = random.Random(seed)
    vals = []
    for _ in range(n_splits):
        idx = list(range(len(instance_circuits)))
        rng.shuffle(idx)
        h = len(idx) // 2
        if h == 0:
            continue
        a = aggregate([instance_circuits[i] for i in idx[:h]], p)
        b = aggregate([instance_circuits[i] for i in idx[h:2 * h]], p)
        vals.append(jaccard(a, b))
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def permutation_null(scores_a: torch.Tensor, scores_b: torch.Tensor,
                     size_a: int, size_b: int, n_draws: int = 1000,
                     seed: int = 0) -> dict:
    """Score-matched null for Jaccard.

    NOT a hypergeometric test: EAP-IG scores are heavily skewed and a handful of
    generic edges (BOS attention, late-layer MLPs, the answer-letter head) appear
    in every circuit of every task. Sampling edges with probability proportional
    to |score| preserves that skew, so the null reflects "circuits of this size
    drawn from these score distributions" rather than "edges are exchangeable".
    """
    g = torch.Generator().manual_seed(seed)
    pa = scores_a.abs().flatten()
    pb = scores_b.abs().flatten()
    pa = pa / pa.sum() if pa.sum() > 0 else None
    pb = pb / pb.sum() if pb.sum() > 0 else None
    if pa is None or pb is None:
        return {"mean": float("nan"), "p95": float("nan")}
    vals = []
    for _ in range(n_draws):
        ia = torch.multinomial(pa, size_a, replacement=False, generator=g)
        ib = torch.multinomial(pb, size_b, replacement=False, generator=g)
        sa, sb = set(ia.tolist()), set(ib.tolist())
        vals.append(len(sa & sb) / len(sa | sb))
    vals.sort()
    return {"mean": sum(vals) / len(vals),
            "p95": vals[int(0.95 * len(vals))],
            "n_draws": n_draws}


def empirical_p(observed: float, null: dict, vals: list[float] | None = None) -> float:
    """One-sided p: fraction of null draws at least as extreme as observed."""
    if vals:
        return sum(1 for v in vals if v >= observed) / len(vals)
    return 0.0 if observed > null.get("p95", float("inf")) else float("nan")
