"""
Build the three task datasets into a canonical JSONL format.

Implements research_plan_v2.md section 6.4.

Output: data/processed/{gsm8k_mc,logiqa,mmlu}.jsonl
Each line: {qid, task, stem, options[4], gold_idx, meta}

Run:  python src/build_datasets.py --n 600
"""

from __future__ import annotations

import argparse
import json
import random
import re
from pathlib import Path

OUT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed"

# ---------------------------------------------------------------------------
# MMLU: restrict to knowledge-recall subjects.
#
# Deliberately EXCLUDES mathematics (overlaps the math task) and formal_logic
# (overlaps the logic task). The paper's claim is that the three tasks span
# distinct cognitive modalities -- that claim is only credible if the knowledge
# task does not smuggle in math or deduction.
# ---------------------------------------------------------------------------
MMLU_KNOWLEDGE_SUBJECTS = {
    "world_religions", "high_school_geography", "high_school_government_and_politics",
    "prehistory", "high_school_us_history", "high_school_world_history", "human_aging",
    "miscellaneous", "nutrition", "sociology", "marketing", "management",
    "business_ethics", "public_relations", "professional_psychology",
    "high_school_psychology", "philosophy", "global_facts", "astronomy", "anatomy",
    "clinical_knowledge", "medical_genetics", "virology", "international_law",
    "jurisprudence", "us_foreign_policy", "human_sexuality",
    # moral_scenarios deliberately excluded: its options are "Wrong, Wrong" /
    # "Not wrong, Not wrong" pairs, models sit near chance, and it is not
    # knowledge recall. It would burn generation budget only to fail filter 2.
}

MAX_STEM_CHARS = 700
MAX_OPTION_CHARS = 120


# ---------------------------------------------------------------------------
# GSM8K -> 4-way multiple choice
# ---------------------------------------------------------------------------

_CALC_RE = re.compile(r"<<([^>]+)>>")
_NUM_RE = re.compile(r"-?\d[\d,]*\.?\d*")


def _to_num(s: str):
    try:
        v = float(str(s).replace(",", "").strip())
        return v
    except (ValueError, AttributeError):
        return None


def _fmt(v: float) -> str:
    """Render a number the way GSM8K answers do (ints without .0)."""
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return f"{v:.2f}".rstrip("0").rstrip(".")


def _gsm8k_intermediates(answer_text: str, gold: float) -> list[float]:
    """Intermediate results from GSM8K's <<a+b=c>> calculator annotations.

    These are the model's natural 'stopped one step early' answers, which makes
    them far more plausible distractors than synthetic perturbations.
    """
    vals = []
    for expr in _CALC_RE.findall(answer_text):
        if "=" in expr:
            rhs = expr.split("=")[-1]
            v = _to_num(rhs)
            if v is not None:
                vals.append(v)
    # drop the final one (that IS the gold) and dedupe, preserving order
    seen, out = set(), []
    for v in vals:
        if abs(v - gold) < 1e-9:
            continue
        if v in seen:
            continue
        seen.add(v)
        out.append(v)
    return out


def _synthetic_distractors(gold: float) -> list[float]:
    """Magnitude slips and off-by-one-operation errors (plan section 6.4)."""
    cands = [gold * 2, gold / 2, gold * 10, gold + 10, gold - 10, gold + 1, gold - 1]
    return [c for c in cands if c is not None]


def _acceptable(v: float, gold: float, chosen: list[float]) -> bool:
    if v is None:
        return False
    if abs(v - gold) < max(1e-9, abs(gold) * 0.01):   # within 1% of gold
        return False
    if gold > 0 and v <= 0:                            # negative when gold positive
        return False
    if abs(v) > abs(gold) * 1000:                      # absurd magnitude
        return False
    for c in chosen:
        if abs(v - c) < max(1e-9, abs(c) * 0.01):
            return False
    return True


def build_gsm8k_mc(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("openai/gsm8k", "main", split="train")
    out, skipped = [], 0
    for row in ds:
        if len(out) >= n:
            break
        ans_text = row["answer"]
        if "####" not in ans_text:
            skipped += 1
            continue
        gold = _to_num(ans_text.split("####")[-1])
        if gold is None:
            skipped += 1
            continue

        distractors: list[float] = []
        for v in _gsm8k_intermediates(ans_text, gold):
            if len(distractors) >= 3:
                break
            if _acceptable(v, gold, distractors):
                distractors.append(v)
        for v in _synthetic_distractors(gold):
            if len(distractors) >= 3:
                break
            if _acceptable(v, gold, distractors):
                distractors.append(v)
        if len(distractors) < 3:
            skipped += 1
            continue

        opts = [_fmt(gold)] + [_fmt(d) for d in distractors[:3]]
        if len(set(opts)) != 4:
            skipped += 1
            continue
        order = list(range(4))
        rng.shuffle(order)
        shuffled = [opts[i] for i in order]
        gold_idx = order.index(0)

        out.append({
            "qid": f"gsm8k_{len(out):04d}",
            "task": "gsm8k_mc",
            "stem": row["question"].strip(),
            "options": shuffled,
            "gold_idx": gold_idx,
            "meta": {"gold_value": _fmt(gold),
                     "n_intermediates": len(_gsm8k_intermediates(ans_text, gold))},
        })
    print(f"  gsm8k_mc: built {len(out)}, skipped {skipped}")
    return out


# ---------------------------------------------------------------------------
# LogiQA 2.0
# ---------------------------------------------------------------------------

def build_logiqa(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("datatune/LogiQA2.0", split="test")
    out, skipped = [], 0
    for row in ds:
        if len(out) >= n:
            break
        try:
            r = json.loads(row["text"])
        except json.JSONDecodeError:
            skipped += 1
            continue
        opts = r.get("options")
        gold = r.get("answer")
        if not isinstance(opts, list) or len(opts) != 4 or not isinstance(gold, int):
            skipped += 1
            continue
        if not 0 <= gold < 4:
            skipped += 1
            continue
        ctx = (r.get("text") or "").strip()
        qq = (r.get("question") or "").strip()
        stem = f"{ctx}\n\n{qq}".strip()
        opts = [str(o).strip() for o in opts]
        if len(stem) > MAX_STEM_CHARS or max(len(o) for o in opts) > MAX_OPTION_CHARS:
            skipped += 1
            continue
        if len(set(opts)) != 4:
            skipped += 1
            continue
        out.append({
            "qid": f"logiqa_{len(out):04d}",
            "task": "logiqa",
            "stem": stem,
            "options": opts,
            "gold_idx": gold,
            "meta": {"type": list((r.get("type") or {}).keys())},
        })
    print(f"  logiqa:   built {len(out)}, skipped {skipped}")
    return out


# ---------------------------------------------------------------------------
# MMLU (knowledge subjects only)
# ---------------------------------------------------------------------------

def build_mmlu(n: int, rng: random.Random) -> list[dict]:
    from datasets import load_dataset

    ds = load_dataset("cais/mmlu", "all", split="test")
    pool = []
    for row in ds:
        if row["subject"] not in MMLU_KNOWLEDGE_SUBJECTS:
            continue
        choices = [str(c).strip() for c in row["choices"]]
        if len(choices) != 4 or len(set(choices)) != 4:
            continue
        stem = row["question"].strip()
        if len(stem) > MAX_STEM_CHARS or max(len(c) for c in choices) > MAX_OPTION_CHARS:
            continue
        if not 0 <= int(row["answer"]) < 4:
            continue
        pool.append((stem, choices, int(row["answer"]), row["subject"]))

    rng.shuffle(pool)
    out = []
    for stem, choices, gold, subj in pool[:n]:
        out.append({
            "qid": f"mmlu_{len(out):04d}",
            "task": "mmlu",
            "stem": stem,
            "options": choices,
            "gold_idx": gold,
            "meta": {"subject": subj},
        })
    print(f"  mmlu:     built {len(out)} from pool of {len(pool)}")
    return out


# ---------------------------------------------------------------------------

BUILDERS = {"gsm8k_mc": build_gsm8k_mc, "logiqa": build_logiqa, "mmlu": build_mmlu}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=600,
                    help="questions per task (plan wants 500; 600 gives headroom)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--tasks", nargs="*", default=list(BUILDERS))
    args = ap.parse_args()

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(args.seed)

    print(f"Building {args.n} questions per task -> {OUT_DIR}")
    for task in args.tasks:
        rows = BUILDERS[task](args.n, rng)
        path = OUT_DIR / f"{task}.jsonl"
        with path.open("w", encoding="utf-8") as f:
            for r in rows:
                f.write(json.dumps(r, ensure_ascii=False) + "\n")
        # gold position balance -- a skew here would confound hint targeting
        counts = [0] * 4
        for r in rows:
            counts[r["gold_idx"]] += 1
        print(f"    -> {path.name}  n={len(rows)}  gold A/B/C/D = {counts}")


if __name__ == "__main__":
    main()
