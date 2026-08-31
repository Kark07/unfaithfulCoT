# Runbook — what to run, in order

Operational companion to [research_plan_v2.md](research_plan_v2.md) (the science)
and [PROGRESS.md](PROGRESS.md) (what happened so far).

**Status:** E0 and E1 complete. E2–E9 scripted and unrun.
**Model:** `Qwen/Qwen2.5-7B-Instruct` · **Hint:** `grader` · **Disclosure arm:** KEEP

---

## 0. Every session

```bash
cd unfaithfulCoT
source env.sh                 # venv + project-local HF caches
nvidia-smi                    # shared GPU: pick a free device
export CUDA_VISIBLE_DEVICES=<free-id>
tmux new -s work              # everything below is long-running
```

---

## Stage 1 — Validate the pipeline ⚠️ DO THIS FIRST

Nothing downstream means anything until EAP-IG is proven correct. If the known
IOI circuit does not come back, every later number is silently garbage.

```bash
python -u src/run_e2_validate.py --stage ioi
```

**Pass:** ≥2 of 3 name-mover heads (9.6, 9.9, 10.0) **and** ≥2 of 4 S-inhibition
heads (7.3, 7.9, 8.6, 8.10) appear in the top-2000 edges.

**If it fails**, in order of likelihood:
1. `set_use_attn_result(True)` not applied → per-head writes unavailable
2. hook names differ in your TransformerLens version → print `model.hook_dict.keys()`
3. metric sign flipped → check `logit(IO) − logit(S)`
4. clean/corrupted not token-aligned → the templates skip mismatches, check how many

Then pick the operating edge budget `k*`:

```bash
python -u src/run_e2_validate.py --stage curve \
    --model Qwen/Qwen2.5-7B-Instruct \
    --cell results/e1_Qwen2.5-7B-Instruct_mmlu_grader.jsonl --n 20
```

**Output:** `k*` = smallest budget recovering ≥85% of the metric. **Use that `k`
everywhere below.** If nothing reaches 85%, widen `--ks` before going further.

---

## Stage 2 — Label U vs F (needs an API key, no GPU)

Every `nonverb` number so far comes from keyword matching, which has been wrong
in **both** directions (F12 false positives, F21 false negatives). Nothing
keyword-derived goes in the paper.

Run this on your laptop, not the shared server — an exported key is readable by
anyone who can see your process environment.

```bash
export ANTHROPIC_API_KEY=...        # or: ant auth login
python src/judge.py results/e1_Qwen2.5-7B-Instruct_mmlu_grader.jsonl   --batch
python src/judge.py results/e1_Qwen2.5-7B-Instruct_logiqa_grader.jsonl --batch
```

Then validate the judge against your own labels:

```bash
python src/inspect_cots.py results/e1_..._mmlu_grader.jsonl --n 50 --label
python src/judge.py results/e1_..._mmlu_grader_judged.jsonl \
    --validate results/e1_..._mmlu_grader_handlabels.jsonl
```

**Gate: Cohen's κ ≥ 0.80.** Below that, tighten `JUDGE_SYSTEM` using the
disagreements as few-shot examples and re-run. **Do not start E4 on a noisy
split** — the U/F labels are what the suppression circuit is built from.

> **E3 does not need this.** Influence circuits use all switched instances, so
> Stage 3 can run in parallel with Stage 2.

---

## Stage 3 — Extract circuits

Substitute your `k*` from Stage 1 for `2000`.

### 3a. Influence circuits — E3 (no judge needed)

```bash
for CELL in mmlu logiqa gsm8k_mc; do
  python -u src/run_circuits.py --arm influence --k 2000 --save-scores \
      --cell results/e1_Qwen2.5-7B-Instruct_${CELL}_grader.jsonl
done
```

### 3b. Task-solving floor — E5b (no judge needed)

The baseline any "shared unfaithfulness mechanism" must beat. Without it, an
overlap number cannot be distinguished from the generic machinery for answering
multiple-choice questions.

```bash
for CELL in mmlu logiqa gsm8k_mc; do
  python -u src/run_circuits.py --arm solve --k 2000 --save-scores \
      --cell results/e1_Qwen2.5-7B-Instruct_${CELL}_grader.jsonl
done
```

### 3c. Suppression circuits — E4 (**needs Stage 2**)

```bash
for CELL in mmlu logiqa; do
  python -u src/run_circuits.py --arm disclosure --k 2000 --save-scores \
      --cell results/e1_Qwen2.5-7B-Instruct_${CELL}_grader_judged.jsonl
done
```

Writes both `disclosure_*` (top-|φ|) and `suppression_*` (negative-φ only —
edges pushing toward "No" **despite** a hint being present; plan §6.5).

**Outputs** → `circuits/`: `{arm}_{task}_{hint}_p{50,60,70}.json`,
`*_instances.json` (per-instance, needed for the split-half ceiling),
`*_scores.pt` (needed for the permutation null), `*_summary.json`.

---

## Stage 4 — Analysis (fast, CPU-ish)

```bash
python src/run_e5_e6_analysis.py --p 60
```

Produces, in one pass:

| | What |
|---|---|
| **E5** | split-half ceiling (per cell), task-solving floor, score-matched permutation null |
| **E6a** | `J_raw`, `J_norm = J_raw / J_half`, `p_perm` for every pair |
| **E6b** | **main figure** — within-hint-across-task vs within-task-across-hint |
| **E6c** | separability: is influence the same circuit as suppression? |

**Never report a bare Jaccard.** Always `J_raw`, `J_norm`, and the floor together.
Also sweep `--p 50` and `--p 70` for the appendix, and **do not choose `p` after
seeing the overlap results.**

---

## Stage 5 — Transfer (E7)

Overlap is a set statistic and does not prove sufficiency — Hanna et al. showed
circuits can overlap heavily and still fail to transfer.

```bash
python -u src/run_e7_transfer.py --arm influence --p 60 --n 25 \
    --cells results/e1_Qwen2.5-7B-Instruct_mmlu_grader.jsonl \
            results/e1_Qwen2.5-7B-Instruct_logiqa_grader.jsonl \
            results/e1_Qwen2.5-7B-Instruct_gsm8k_mc_grader.jsonl
```

Read off-diagonals against the diagonal of their own column. **Asymmetry is a
finding** — a superset circuit means core mechanism + task-specific extras.

---

## Stage 6 — Ablations (E8, E9)

```bash
# E8: does removing the shared influence circuit kill hint-following,
#     while sparing accuracy, where random ablation does neither?
python -u src/run_e8_e9_ablation.py --arm influence --p 60 --n 25 \
    --cells results/e1_Qwen2.5-7B-Instruct_mmlu_grader.jsonl \
            results/e1_Qwen2.5-7B-Instruct_logiqa_grader.jsonl

# E9 (HEADLINE): does removing the shared suppression circuit make the model
#     start ADMITTING the hint?
python -u src/run_e8_e9_ablation.py --arm suppression --p 60 --n 25 \
    --cells results/e1_..._mmlu_grader_judged.jsonl \
            results/e1_..._logiqa_grader_judged.jsonl
```

Both run four conditions: baseline, ablate-this-arm, ablate-the-other-arm, and
size-matched random. The **double dissociation** — suppression ablation raises
verbalization while influence ablation lowers following without raising it — is
the strongest form of the claim.

**Two caveats to carry into the paper:**
1. Ablation during generation is **node-level** (writers incident to the circuit's
   edges); the circuit analysis is edge-level. A real limitation — state it.
2. Verbalization in the ablation output uses the keyword heuristic. **Re-score the
   saved generations with `judge.py` before publishing any number.**

---

## Optional — rescue GSM8K

At 600 questions GSM8K gave only ~6 U instances (5% switch × 21% non-verbalized),
below the 30 floor. The U rate is ~1% of questions, so ~3000 gets ~30.

```bash
python src/build_datasets.py --n 3000 --tasks gsm8k_mc
python -u src/run_e1.py --model Qwen/Qwen2.5-7B-Instruct \
    --tasks gsm8k_mc --hint-types grader --n 3000 --batch-size 24 --tag e1b
```

~2 h unattended. Worth it for a three-domain claim; skippable if two domains
suffice, since **F14/F16 (math resists hints) is a finding in its own right.**

## Optional — second hint type for E6b

E6b is strongest with ≥2 hint types. Sycophancy switches at a similar rate but is
verbalized far more often (F13), which is exactly the contrast that makes the
hint axis interesting.

```bash
python -u src/run_e1.py --model Qwen/Qwen2.5-7B-Instruct \
    --tasks mmlu logiqa --hint-types sycophancy --n 600 --batch-size 24 --tag e1c
```

---

## Diagnostics

```bash
python src/validate_extraction.py results/<cell>.jsonl --task gsm8k_mc --n 10 --score
python src/analyze_margin.py 'results/e1_*.jsonl'
python src/rescore_verbalization.py results/<cell>.jsonl --show
python src/inspect_unparsed.py results/<cell>.jsonl
```

`validate_extraction.py` is still **open work (F17)** — extraction accuracy has
never been measured, and F14 (GSM8K resists hints) rests entirely on it.

---

## Dependency graph

```
E2 ioi ──► E2 curve (k*) ──┬──► E3 influence ──┬──► E5/E6 analysis ──► E7 transfer
                           │                   │
                           ├──► E5 solve floor ┘
                           │
judge.py (κ≥0.80) ─────────┴──► E4 suppression ──► E6c separability ──► E8 / E9
```

**Critical path:** E2 → E3 → E5/E6. That alone is a publishable paper: *do
hint-following circuits transfer across reasoning domains, measured against a
noise ceiling and a task-solving floor?*
E4/E9 add the suppression story and need the judge first.
