# Progress Log

Living tracker for the unfaithfulness-circuits project. Updated whenever
something is built, found, or decided.

- **Plan of record:** [research_plan_v2.md](research_plan_v2.md) (v2.1)
- **Superseded:** `research_plan.md` (v1, kept only for the §1 diff table)
- **Deadline status:** original targets (28/29 Aug 2026) **MISSED**. Both were
  non-archival, so nothing is burned — re-targeting a later venue (see §3b).
- **Last updated:** 31 Aug 2026 — E0+E1 complete; E2–E9 scripted, unrun

---

## 1. Status at a Glance

| Phase | Date | Work | Status |
|---|---|---|---|
| Setup | Aug 22 | datasets, prompt contract (3600/3600 aligned) | ✅ |
| **E0** | Aug 22–23 | pilot + model gate | ✅ model=Qwen2.5-7B, hint=grader, disclosure arm KEPT |
| **E1** | Aug 26 | full-scale generation, 600 q × 3 tasks | ✅ 117 / 123 / 29 switched |
| Deadline | Aug 28–29 | submit | ❌ **missed** — re-targeting (§3b) |
| Tooling | Aug 31 | E2–E9 all scripted | ✅ written, syntax-checked, **unrun** |
| **E2** | — | IOI sanity + `k*` | ⬜ **next, blocks everything** |
| **E3/E5b** | — | influence circuits + task-solving floor | ⬜ |
| Judge | — | κ ≥ 0.80 on 50 hand labels | ⬜ (parallel with E3) |
| **E4** | — | disclosure + suppression circuits | ⬜ needs judge |
| **E5/E6** | — | ceiling, floor, null, overlap, two-way, separability | ⬜ |
| **E7** | — | transfer matrix | ⬜ |
| **E8/E9** | — | ablations + double dissociation | ⬜ |

**Critical path right now:** `run_e2_validate.py --stage ioi`. EAP-IG is untested
code performing the most important computation in the project; if it cannot recover
the known IOI circuit, every downstream number is silently garbage.

**Operational steps:** [RUNBOOK.md](RUNBOOK.md).

---

## 2. What Each Script Does

Run order is top to bottom. Only `build_datasets.py` and `run_e0_pilot.py` are
entry points; the other two `src/` modules are libraries.

### `src/prompts.py` — the prompt contract *(library)*

Builds the three token-aligned variants that the whole method rests on
(plan §6.0–6.2). If this is wrong, every circuit downstream is meaningless —
which is exactly the mistake v1 made.

| Function | Does |
|---|---|
| `Question` | Container: stem, 4 options, gold index. Rejects malformed items. |
| `calibrate_neutral(tok, hint_type)` | Picks a neutral filler `P_0` whose **token length matches** the hinted slot, for the specific tokenizer in use. Raises if no candidate matches. |
| `build_triplet(...)` | Returns `{P_X, P_Y, P_0}` for one question. Refuses if the hint target is the gold answer. |
| `assert_token_aligned(tok, *prompts)` | All variants must tokenize to identical length; returns the differing positions. |
| `check_triplet(...)` | Enforces all three contracts: (1) equal length, (2) `P_X`/`P_Y` differ at **exactly one** token — the letter, (3) `P_X`/`P_0` differ **only inside the hint slot**. |
| `select_xy(q, option_logits)` | Picks hint targets: X = highest-logit *incorrect* option (most plausible distractor → maximises switch rate), Y = second-highest. |

### `src/build_datasets.py` — dataset construction *(entry point)*

`python src/build_datasets.py --n 600` → `data/processed/{gsm8k_mc,logiqa,mmlu}.jsonl`

- **GSM8K → 4-way MC.** Distractors come from GSM8K's own `<<a+b=c>>` calculator
  annotations, i.e. real intermediate values. These are genuine "stopped one step
  early" errors, far more plausible than synthetic perturbations. Falls back to
  magnitude slips (×2, ÷2, ×10, ±10, ±1) when annotations run out. Rejects anything
  within 1% of gold, negative-when-gold-positive, or absurd in magnitude.
- **LogiQA 2.0** from `datatune/LogiQA2.0`, parsed out of a raw-JSON `text` column.
- **MMLU** restricted to 27 knowledge-recall subjects. **Excludes mathematics and
  `formal_logic`** (they would overlap the other two tasks and destroy the paper's
  "three distinct cognitive modalities" claim) and `moral_scenarios` (degenerate).

### `src/model_utils.py` — model + scoring passes *(library)*

Plain HF `transformers`, **not** TransformerLens — E0 needs generation only, so the
go/no-go decision does not block on circuit tooling.

| Function | Plan ref | Does |
|---|---|---|
| `load_model` | — | Loads model/tokenizer, forces fp32 on CPU. |
| `chat_wrap` | §6.5 | Applies the instruct chat template. |
| `assert_chat_aligned` | §6.2 | Re-checks alignment **after** templating — it is not safe to assume it survives. |
| `letter_logits` | Pass α | A/B/C/D logits at the answer position. `M_infl = logit(X) − logit(Y)`. Also ranks distractor plausibility for `select_xy`. |
| `disclosure_logits` | Pass γ | Yes/No logits at the disclosure probe. `M_disc = logit(Yes) − logit(No)`. |
| `generate` | — | Free generation. `return_truncated=True` reports whether the token budget ran out. |
| `extract_answer` | — | Pulls the answer letter out of free text; handles bold markdown, "Final Answer", bare `(B)`, "option B". Last match wins. |
| `heuristic_mentions_hint` | §7.3 | **Placeholder only.** Keyword check for E0. E1 replaces it with an LLM judge validated against 50 hand labels. |

### `src/run_e0_pilot.py` — the E0 pilot *(entry point)*

`python -u src/run_e0_pilot.py --model Qwen/Qwen2.5-1.5B-Instruct --n 30`

Runs the full §7 filter cascade and **prints the gate decision itself**:

1. **Stability** — unhinted answer identical across k samples at T=0.7 (on `P_0`)
2. **Competence** — that answer is correct
3. **Target selection** — X/Y from real option logits
4. **Alignment** — raw and post-chat-template
5. **Switch** — does greedy generation under `P_X` yield X?
6. **Verbalization** — heuristic (E0 only)
7. **Probe sanity** — does `M_disc` move between `P_X` and `P_0`?

Output columns: `trunc | stab | comp | elig | SWITCH | nonverb | U est | probeD`.

**Read the report in this order:**
1. **`trunc`** — if >10%, the token budget is too small; re-run at double before
   believing anything else. Truncation looks exactly like instability.
2. **`SWITCH`** — the model gate. ≥20% on ≥2 of 3 → proceed; 10–20% → escalate to
   3B; <10% → escalate to 7B.
3. **`probeD`** — the disclosure-arm gate. <0.5 → drop the arm and retitle.

### `src/inspect_cots.py` / `src/inspect_unparsed.py` — human review *(entry points)*

- `inspect_cots.py <results.jsonl> --n 10 --label` — read CoTs, judge coherence and
  hint-mention by hand. Reports heuristic agreement. This seeds E1's 50 hand labels.
- `inspect_unparsed.py <results.jsonl>` — show complete generations that
  `extract_answer` failed on. Every one is a lost instance, and yield is the
  tightest constraint (§7.2).

### Circuit experiment scripts *(written 31 Aug, none run yet)*

| File | Role |
|---|---|
| `src/eap_ig.py` | **EAP-IG core.** Residual-stream identity: every reader sees the sum of upstream writes, so per-edge attribution collapses to an outer product of writer deltas against reader-input gradients. IG interpolates at the embedding. `faithfulness()` is a sufficiency test against the corrupted baseline (mean-ablation, never zero). |
| `src/circuit_utils.py` | Shared: instance loading, the three scoring passes, circuit save/load, aggregation, Jaccard, split-half ceiling, score-matched permutation null. |
| `src/run_e2_validate.py` | **E2.** `--stage ioi` recovers the known IOI circuit in GPT-2 small (the correctness gate); `--stage curve` sweeps edge budget to pick `k*`. |
| `src/run_circuits.py` | **E3/E4/E5b.** One script, three arms — `influence` (P_X vs P_Y), `disclosure` (P_X vs P_0, sign preserved → suppression), `solve` (task-solving floor). |
| `src/run_e5_e6_analysis.py` | **E5/E6.** Ceiling, floor, null; `J_raw`/`J_norm`/`p_perm`; the two-way decomposition (main figure); influence-vs-suppression separability. |
| `src/run_e7_transfer.py` | **E7.** Asymmetric transfer matrix — circuit from task A tested on task B. |
| `src/run_e8_e9_ablation.py` | **E8/E9.** Ablate shared circuit, measure hint-following / verbalization / accuracy, against ablate-other-arm and size-matched-random controls. |
| `src/judge.py` | LLM judge (Claude Opus 5), Chen et al.'s two-part criterion, Batches API, Cohen's κ validation. |

### `setup_remote.sh` + `env.sh` — shared-server environment

Creates `./.venv`, installs torch cu128 (Blackwell sm_120 needs it), and
**redirects every cache into the project directory** so multi-GB model downloads
don't fill the shared home quota or collide with another user's cache. Generates
`env.sh`; source that in each new shell. Checks free VRAM and warns if someone
else is on the GPU. **Makes no global changes and uses no sudo.**

---

## 3. Findings Log

### F1 — v1's clean/corrupted pair breaks EAP-IG *(22 Aug, from plan review)*
v1 used the *unhinted* prompt as corrupted. EAP-IG patches position-by-position, so
clean/corrupted must be token-aligned; removing the hint changes length.
**→ Fix:** corrupted = same hint pointing at a different letter (`P_Y`), one token apart.

### F2 — v1's metric measures hint-following, not unfaithfulness *(22 Aug)*
`logit(X) − logit(a_u)` takes an **identical value** on a verbalized and a
non-verbalized instance, so it cannot see the property that defines unfaithfulness.
v1 *filtered* on unfaithfulness but *scored* on hint-following.
**→ Fix:** decompose into **influence** (answer position) and **disclosure** (probe
position); unfaithfulness is their conjunction. Unlocked E9, the headline experiment.

### F3 — Jaccard thresholds were uninterpretable *(22 Aug)*
v1 preregistered J>0.5 = universal, borrowed from Merullo's **78% node** overlap.
Edge-level Jaccard is structurally far lower, so the threshold would have forced the
"task-specific" conclusion for arithmetic reasons.
**→ Fix:** E5 measures a split-half noise ceiling and a task-solving floor; report
normalized overlap. Also replaced the hypergeometric test (assumes exchangeable
edges) with a score-matched permutation null.

### F4 — Alignment contract verified at scale *(22 Aug)*
**3600/3600 prompts aligned, 0% drop** (3 tasks × 600 × 2 hint types), against §6.2's
<5% gate. `P_X`/`P_Y` differ at exactly one position in every case.

### F5 — Metadata hints sit at a *fixed* token position *(22 Aug)*
Because the metadata block is prepended, the hint is at token 24 in every question.
Sycophancy hints drift with stem length (34, 68, 66…).
**→ Implication:** if the EAP-IG implementation is position-resolved rather than
position-summed, metadata instances aggregate far more cleanly across questions.
An independent argument for metadata as the primary hint type.

### F6 — LogiQA's standard HF loaders are dead *(22 Aug)*
`datasets` 5.0 dropped script-based datasets, breaking `EleutherAI/logiqa`,
`baber/logiqa2`, `lighteval/logiqa_harness`.
**→ Fix:** `datatune/LogiQA2.0`, JSON-parsed from a `text` column. 10,348 rows.

### F7 — Truncation was indistinguishable from instability *(22 Aug, from smoke test)*
A generation that runs out of budget produces no answer — identical to an unstable
one. On long LogiQA items this would have made the model look unstable and sent us
escalating to 7B to fix a `max_new_tokens` setting, corrupting the single most
important decision of Day 1.
**→ Fix:** `return_truncated`, separate tracking, and the report **refuses to let you
trust the gate** if truncation >10%. Default budget 320 → 400.

### F8 — Answer parser missed bold markdown *(22 Aug, from smoke test)*
`**Final Answer:** (D)` returned `None` — the `**` broke the pattern. Every CoT
ending that way was a silently discarded instance.
**→ Fix:** patterns widened to skip emphasis/punctuation. **12/12** on format
variants. `inspect_unparsed.py` added so remaining gaps surface instead of hiding.

### F9 — MMLU `moral_scenarios` is degenerate *(22 Aug)*
Its options are literally `["Wrong, Wrong", "Not wrong, Not wrong", …]`; models sit
near chance and it isn't knowledge recall.
**→ Fix:** excluded. MMLU rebuilt from 27 subjects, still 600/600 aligned.

### F10 — v1's CFP details were wrong, and the venue may be a scope mismatch *(22 Aug)*
Verified against live CFPs. **5 pages** (not "4–6"), **double-blind**, **non-archival**
at both candidates. More importantly, Interp4Discovery's scope is interpretability for
**natural-science discovery** (protein, climate, astronomy) — this paper is model
internals for safety. [Interpretability as a Science](https://interpscience.github.io/)
names *"experimental designs that distinguish mechanisms from artifacts"* as a core
topic, which is precisely what E5 does.
**→ Status:** venue decision deferred to after E0 (see §4, D3).

### F11 — This machine is not the A6000 *(22 Aug)*
Local box is an RTX 3050 Laptop (4 GB) with CPU-only torch 2.4. Real compute is a
**shared** company server.
**→ Fix:** everything written to be portable; `setup_remote.sh` rewritten for
shared-server hygiene (project-local caches, no global installs, VRAM check).

---

## 4. Decisions Log

| # | Decision | Rationale | Date |
|---|---|---|---|
| D1 | GSM8K-MC replaces AQuA as the math task | AQuA is 5-way algebraic MC where sub-3B models sit near chance; random unhinted answers make "switched" indistinguishable from noise | 22 Aug |
| D2 | Qwen2.5-1.5B-Instruct primary, escalation ladder to 3B/7B | Clean architecture (no soft-capping / sliding-window alternation like Gemma-2), well-supported in TransformerLens, strong for its size. Escalation gated on E0. | 22 Aug |
| D3 | **Venue decided after E0** | User's call. Decision rule in plan §0, keyed to whether the disclosure arm survives. | 22 Aug |
| D4 | **Work to Aug 28, not Aug 29** | Hitting the earlier deadline keeps both venues open; slipping forecloses InterpScience | 22 Aug |
| D5 | Cut from critical path: E10 (2nd model), Pass β, `X_rand` control | 6 days, not 9. Cut deliberately now rather than in a panic on Day 5. | 22 Aug |
| D6 | `data/processed/` is committed to git | ~1 MB; pins the exact question set for reproducibility and lets the server run without HF dataset access | 22 Aug |
| D7 | ACDC validation replaced by faithfulness-vs-size curves | ACDC's ~5h/query was GPT-2 Large; at 1.5–7B it's a time sink, not a validation | 22 Aug |

---

## 5. Open Items

| Priority | Item | Blocks |
|---|---|---|
| 🔴 | **Run `setup_remote.sh` on the shared A6000** | everything |
| 🔴 | **Run E0 pilot** → model gate + disclosure-arm decision | E1 onward |
| 🟠 | Read 10 CoTs by hand (`inspect_cots.py --label`) — coherence is a human call | E0 gate |
| 🟠 | Venue decision (plan §0) | paper framing |
| 🟡 | Verify 4 flagged 2026 arXiv IDs in plan §12 | submission |
| 🟡 | Get `transformer_lens` working + IOI sanity check (E2) | E3 onward |
| 🟡 | LLM judge for verbalization + 50 hand labels (κ ≥ 0.8) | E4 |

---

## 6. How To Run

### On the shared server (first time)
```bash
git clone <repo-url> unfaithfulness_cot && cd unfaithfulness_cot
bash setup_remote.sh          # venv + cu128 torch + project-local caches
```

### Every new shell
```bash
source env.sh                 # activates venv, sets HF_HOME, pins CUDA_VISIBLE_DEVICES
nvidia-smi                    # GPU is shared -- pick a free device first
export CUDA_VISIBLE_DEVICES=<free-device>
```

### E0
```bash
python -u src/run_e0_pilot.py --model Qwen/Qwen2.5-1.5B-Instruct --n 30
python src/inspect_cots.py results/e0_Qwen2.5-1.5B-Instruct.jsonl --n 10 --label
python src/inspect_unparsed.py results/e0_Qwen2.5-1.5B-Instruct.jsonl
```

Datasets are committed — no need to rebuild. To regenerate anyway:
`python src/build_datasets.py --n 600`

---

## 7. Results

Filled in as experiments run. Empty tables live in
[research_plan_v2.md §8](research_plan_v2.md).

### E0 run 1 — 22 Aug, Qwen2.5-1.5B-Instruct, sycophancy, n=30, 400 tok
**VERDICT: INVALID — truncation-confounded. Gate not actionable.**

| Task | trunc | stab | comp | elig | SWITCH | nonverb | U est | probeD |
|---|---|---|---|---|---|---|---|---|
| gsm8k_mc | 33% | 53% | 94% | 15 | 0% | 0% | 0 | n/a |
| logiqa | **47%** | **7%** | 100% | **2** | 0% | 0% | 0 | n/a |
| mmlu | 13% | 50% | 93% | 14 | **21%** | 0% | 0 | 0.12 |

Throughput: ~9 s/question on the A6000.

**Reading:**
- The script's truncation guard fired correctly (F7 earning its keep). Stability
  requires all 3 samples to yield an answer, so per-sample truncation compounds:
  47% any-truncation on LogiQA ≈ 19% per sample, collapsing its eligible pool to 2.
  The "ESCALATE to 7B" verdict is reading that artefact, **not** the model.
- **MMLU is the only clean cell (13% trunc) and it clears the gate at 21% switch.**
  Competence is 93–100% throughout. Working hypothesis: 1.5B is adequate; the other
  two tasks are budget-limited, not capability-limited.
- `probeD = 0.12` is computed on 3 instances. Far too few to drop the disclosure arm.
  **Decision held.**

### ⚠️ F12 — `nonverb = 0%` across all cells *(22 Aug)*
Every hint-following instance was labelled as *mentioning* the hint — the inverse of
Chen et al.'s <20% verbalization. If real, there are **no unfaithful instances** and
the project premise fails at this model scale. Two candidates:
1. **Heuristic over-firing.** `heuristic_mentions_hint` keys on "professor"/"stanford".
   A CoT that merely *restates the prompt* then argues independently is scored as
   verbalizing, when it is in fact unfaithful. False positive.
2. **Small models parrot prompt content**, unlike the frontier models Chen et al. tested.
n=3 cannot separate these. **Resolution: hand-read the CoTs** (`inspect_cots.py`), and
compare against metadata hints, which Chen et al. found suppress verbalization more.

### E0 runs 2–3 — 22 Aug, Qwen2.5-1.5B-Instruct, 800 tok, n=30
Truncation fixed (0–7%). Gate said PROCEED at 1.5B, but yield was fatal.

| Task | Hint | trunc | stab | comp | elig | SWITCH | nonverb | U |
|---|---|---|---|---|---|---|---|---|
| gsm8k_mc | syco | 3% | 63% | 95% | 18 | **0%** | – | 0 |
| gsm8k_mc | meta | 0% | 53% | 94% | 15 | **0%** | – | 0 |
| gsm8k_mc | grader | 0% | 63% | 95% | 18 | **0%** | – | 0 |
| logiqa | syco | 7% | 13% | 50% | 2 | 0% | – | 0 |
| logiqa | meta | 0% | 23% | 57% | 4 | 25% | – | 0 |
| logiqa | grader | 0% | 23% | 57% | 4 | 0% | – | 0 |
| mmlu | syco | 0% | 50% | 93% | 14 | 21% | **0%** | 0 |
| mmlu | meta | 0% | 40% | 83% | 10 | 0% | – | 0 |
| **mmlu** | **grader** | 0% | 60% | 89% | 16 | **19%** | **100%** | **3** |

### F13 — Hint modality controls VERBALIZATION, not switch rate *(22 Aug)*
MMLU: sycophancy 21% switch / **0%** non-verbalized; grader 19% switch / **100%**
non-verbalized. Near-identical switching, opposite disclosure. With a professor the
model narrates its deference; with a verified answer key it absorbs the hint silently
and rationalizes.
**→ `grader` is the primary hint type.** Also a standalone result: this is the hint
axis of E6b, which v1 collected and never analysed. Resolves F12 — unfaithful
instances exist; the old keyword heuristic was hiding them.

### F14 — GSM8K resists hints *(22 Aug, refined 23 Aug)*
0 switches / 51 eligible at 1.5B across all three hint types; only 4% at 7B while
LogiQA went 0% → 33%. Math is **resistant**, not starved.
**Structural cause (a flaw in the filter design):** the competence filter keeps only
questions the model answers *correctly*. On math "correct" means "computed", so
certainty and correctness coincide and no hint can tip it. On knowledge/logic a model
can be right while genuinely uncertain — which is where hints bite. `margin` and
`letter_logits` are now recorded per instance; `analyze_margin.py` tests whether
switching concentrates at low confidence.

### F15 — LogiQA was starved upstream, not hint-resistant *(22 Aug — RESOLVED 23 Aug)*
At 1.5B: stability 13–23%, competence 50–57% → 2–4 eligible from 30, so its 0% switch
carried no information. **Resolved by scale:** at 7B, stability 53%, competence 75%,
12 eligible, **33% switch**. The diagnosis was right and the fix was model size.

### F16 — GSM8K immunity matches the literature *(22 Aug, verified)*
[Faithfulness as Information Flow](https://arxiv.org/abs/2605.24286) injects hints into
arithmetic (DeepMind Mathematics `arithmetic_mul_div_multiple`) using **"A Stanford
professor guessed {hint}"** — our sycophancy hint almost verbatim. Decisive detail:
**their hints are correct 75% of the time**, and wrong-hint following only "climbs
sharply across training" under RL. They had to make hints *statistically reliable* and
train on them before models leaned on them for arithmetic. They also report "the
shortcut is learned but not verbalized" — independent confirmation of F13.

Dataset choices point the same way: Turpin et al. used BIG-Bench Hard (symbolic /
commonsense, not arithmetic); Chen et al. used MMLU + GPQA, **no math**; FaithCoT-Bench
uses AQuA (harder 5-way algebra) rather than GSM8K. **Nobody demonstrates wrong-hint
following on easy verifiable arithmetic.** GSM8K's near-zero is expected, not a bug.

### F17 — Extraction is un-auditable on the unhinted side *(22 Aug — OPEN)*
`switched := (a_u != X) and (a_h == X)` depends on `extract_answer` reading BOTH runs,
but only the hinted CoT was stored — `a_u` could never be checked after the fact. A
silent failure on the unhinted side fabricates or destroys switches invisibly, and
F14 rests entirely on it. Math CoTs end in numbers and equations and can fail
differently from prose.
**→ Fix:** `unhinted_gen` / `hinted_gen` now stored verbatim;
`src/validate_extraction.py` prints them beside the extracted letter for hand scoring.
**Extraction accuracy has NOT yet been measured. Runs before commit `284eebb` lack the
verbatim fields, so this needs a fresh short run.**

### F18 — How the field measures this *(22 Aug, verified)*
Fetched Chen et al. 2025 (arXiv 2505.05410) methodology directly:

| Question | Their method |
|---|---|
| Did the hint flip the answer? | **Automated, deterministic.** "We filter the prompt pairs and retain only those where the model's answer changes from non-hint to hint after inserting the hint". **No judge, no humans.** Identical to ours. |
| Does the CoT verbalize the hint? | **LLM judge** — Claude 3.5 Sonnet (New) — "validate the checker's quality through manual inspection on a subset". |

1. **Our switch test is the field standard.** A judge there would be strictly worse —
   it injects error into a counterfactual we measure exactly. The only defensible model
   role on that side is a *backstop extractor* when the regex fails (F17).
2. **They report NO quantitative agreement** — no kappa, no precision/recall. Computing
   and reporting Cohen's kappa is a real improvement and lands on the
   Interpretability-as-a-Science topic "measurement validity and evaluation design".

Their criterion is **two-part** — the CoT must (a) mention the hint and (b) depend on
it. Adopted in `src/judge.py`. **Gate: kappa >= 0.80 before E4.**

### E0 run 4 — 23 Aug, **Qwen2.5-7B-Instruct**, grader, n=30, 800 tok — **GATE PASSED**

| Task | Hint | trunc | stab | comp | elig | SWITCH | nonverb | U | probeD |
|---|---|---|---|---|---|---|---|---|---|
| gsm8k_mc | grader | 0% | 93% | 96% | 27 | **4%** | 100% | 1 | 0.12 |
| logiqa | grader | 3% | 53% | 75% | 12 | **33%** | 100% | 4 | **1.06** |
| mmlu | grader | 0% | 90% | 85% | 23 | **22%** | 100% | 5 | **0.95** |

**Three decisions locked:**
1. **Model = Qwen2.5-7B-Instruct.** Stability 93/53/90%, competence 96/75/85%, all far
   above 1.5B. LogiQA's eligible pool tripled (4 → 12).
2. **Disclosure arm = KEEP.** `probeD` 1.06 / 0.95, well clear of the 0.5 gate (1.5B
   gave 0.12). **E9 — ablate suppression to force verbalization — is back on.** It was
   one number away from being cut.
3. **`nonverb = 100%` in every cell.** Every switch is unfaithful.

**Projected yield at 600 questions/task:** MMLU ≈ **101 U**, LogiQA ≈ **79 U**,
GSM8K ≈ **22 U** (below the 30 floor — needs ~1000+; GSM8K train has 7473 available).

### F19 — 7B makes E1 generation the long pole *(23 Aug)*
Sequential generation is ~30 s/question at 7B. E1 needs 600 q × 3 tasks × ~4
generations ≈ **15 h per hint type**, which does not fit the remaining days.
**→ Fix:** `generate_batch()` / `letter_logits_batch()` in `model_utils.py`,
**left-padded** (decoder-only models produce garbage under right-padding).

### F20 — EAP-IG at 7B: **RESOLVED, feasible** *(26 Aug)*
E0 chose 7B for *behaviour*, but no circuit code has ever run on it. EAP-IG memory
scales with activations × edges and 7B is ~4.5× the model. **If EAP-IG cannot run at
7B we are caught between a model that yields behaviour (7B) and one that yields
circuits (1.5B), and the design must change.** Sharpest remaining fork; cheap to settle.
**→ `src/check_eapig_feasibility.py`. RESULT (26 Aug), RTX PRO 6000 Blackwell Max-Q,
101.1 GB free:**

| Stage | Cost |
|---|---|
| TransformerLens load | 15.3 GB, OK |
| forward + backward with grads | 33.2 GB peak, gradients flow |
| full activation cache | 1.6 GB per cache; EAP-IG est. ~4.9 GB |
| after teardown | **67.3 GB free** |

Graph: 28 layers × 28 heads, d_model 3584, **935,747 edges**; edge-score tensor 0.004 GB.

**Consequence for scheduling:** at m=5 integration steps, EAP-IG is a handful of
forward+backward passes per instance — **minutes** for a few hundred instances, not
hours. **Generation is the bottleneck, not circuit extraction.** Prioritise E1
throughput; circuits are cheap. Headroom also allows batched EAP-IG if needed.

### F21 — The keyword heuristic had NO grader-hint terms *(23 Aug)*
`_HINT_WORDS` covered sycophancy ("professor", "stanford") and metadata
("metadata", "success-rate", "<answer>") but contained **nothing for the grader
hint** — no "answer key", no "verified answer". So a CoT reading *"the answer key
says B, so B"* — an explicitly verbalized CoT — matched no keyword and was scored
**NOT verbalized**.

**This makes `nonverb = 100%` on every grader cell potentially trivial**, and F13
(hint modality controls verbalization) rests on exactly that number. Fixing F12's
false positives introduced false negatives in the same function.

**→ Fixed:** grader/answer-key terms added; 8/8 on a mixed test.
**→ Real lesson:** keyword matching has now failed in BOTH directions on the same
metric. The `nonverb` figures stay **provisional** until `judge.py` has run and
been validated at kappa >= 0.80. No number derived from the keyword heuristic goes
in the paper.

### E1 — *(pending)* full-scale generation, 7B + grader

| Task | Hint | n | elig | SWITCH | U | judge kappa |
|---|---|---|---|---|---|---|
| gsm8k_mc | grader | | | | | |
| logiqa | grader | | | | | |
| mmlu | grader | | | | | |

Model locked: **Qwen2.5-7B-Instruct**  |  Disclosure arm: **KEEP**  |  Venue: ______

### ✅ E1 — 26 Aug, Qwen2.5-7B-Instruct, grader, n=600/task, 800 tok, batch 24

| cell | n | trunc | stab | comp | elig | SWITCH | \|U∪F\| | \|R\| | U~ | probeD |
|---|---|---|---|---|---|---|---|---|---|---|
| **mmlu/grader** | 600 | 0% | 82% | 85% | 421 | **28%** | **117** | 304 | 68 | 0.53 |
| **logiqa/grader** | 600 | 3% | 65% | 74% | 291 | **42%** | **123** | 168 | 115 | 0.71 |
| gsm8k_mc/grader | 600 | 0% | 92% | 99% | 545 | **5%** | 29 | 516 | 6 | 0.33 |

Runtime ≈ 21 min/cell with batched generation (F19 fix); ~1 h total vs the ~15 h
sequential generation would have cost.

**Two cells clear the 30-instance floor with large margins** — 117 and 123
switched instances, plus resistance sets of 304 and 168 for the E8/E9 controls.
GSM8K yields 29 switched but only 6 U.

### ✅ F21 CONFIRMED — the 100% non-verbalization was a keyword artefact
With grader terms added to `_HINT_WORDS`, non-verbalization is **58% / 93% / 21%**,
not 100% everywhere.

**F13 must be restated.** Grader hints suppress verbalization *substantially but
not universally*, and **the rate varies sharply by task**: MMLU 42% verbalized vs
LogiQA 7%. That task difference is itself interesting — but it is still a keyword
number, and keyword matching has now been wrong in both directions. **The judge
decides it. No keyword-derived figure goes in the paper.**

### 🟠 GSM8K yield — below floor, recoverable
5% switch × 21% non-verbalized ≈ 1% of questions become U instances. ~3000
questions would give ~30 U (GSM8K train has 7473 available), ~2 h unattended.
Optional: **F14/F16 — math resists hints — stands as a finding either way**, and
the paper survives as a two-domain result.

---

## 3b. Deadline

**Missed.** The working deadline was 28 Aug (InterpScience) with 29 Aug as the
fallback (Interp4Discovery). Both passed on 31 Aug with E2–E9 unrun.

**This does not damage the work.** Both venues were non-archival, so nothing is
burned. The findings, the datasets, the E1 instance sets, and the full experiment
suite are intact and re-targetable. Candidate venues: ICLR 2027 workshops, the
next Mechanistic Interpretability workshop, or a standalone arXiv preprint.

**What the extra time buys:** the things that were going to be cut for time — the
second hint type for E6b's two-way decomposition, the GSM8K expansion, the κ
validation done properly on 50 hand labels rather than 10, and E9 with the judge
rather than the keyword heuristic. The rushed version would have been visibly
rushed.

---

## 3c. Code written but NOT YET RUN (31 Aug)

Every remaining experiment is now scripted. All syntax-checked; the non-GPU logic
(graph construction, causal masking, aggregation, Jaccard, permutation null) is
unit-tested. **None has run on the GPU.**

| File | Experiment | Needs |
|---|---|---|
| `src/eap_ig.py` | EAP-IG core: attribution + IG + faithfulness | — |
| `src/circuit_utils.py` | shared: instance loading, scoring passes, circuit I/O, overlap stats | — |
| `src/run_e2_validate.py` | **E2** IOI sanity + faithfulness-vs-size curve → `k*` | GPU |
| `src/run_circuits.py` | **E3/E4/E5b** influence, disclosure+suppression, task-solving floor | GPU; E4 needs judge |
| `src/run_e5_e6_analysis.py` | **E5/E6** ceiling, floor, null, overlap, two-way, separability | circuits |
| `src/run_e7_transfer.py` | **E7** asymmetric transfer matrix | GPU + circuits |
| `src/run_e8_e9_ablation.py` | **E8/E9** causal ablation + double dissociation | GPU + circuits |

See [RUNBOOK.md](RUNBOOK.md) for the exact order and the gates at each stage.

### Design notes carried into these scripts

- **E3 does not need the judge.** Influence circuits use all switched instances;
  only E4 needs the U/F split. So circuit extraction and judge validation can
  proceed in parallel.
- **Ablation during generation is node-level**, while circuit analysis is
  edge-level. Per-edge ablation at every decoding step was not worth the
  complexity. **A real limitation — state it in the paper.**
- **`permutation_null` samples edges proportional to |score|**, preserving
  EAP-IG's heavy skew. A hypergeometric test would assume edges are exchangeable
  and report significance for the handful of generic edges (BOS attention,
  late-layer MLPs, the answer-letter head) present in every circuit of every task.
- **Edge count convention differs** between `check_eapig_feasibility.py` (935,747
  — counts q/k/v as separate reader inputs) and `eap_ig.py` (330,891 — one reader
  per head). Both are valid; **state which one the paper reports.**

---

## 3d. Open work, ranked

| Priority | Item | Blocks |
|---|---|---|
| 🔴 | **E2 IOI sanity** — EAP-IG is untested code doing the most important computation | everything |
| 🔴 | **`k*` from the faithfulness curve** | E3–E9 |
| 🟠 | **Judge + κ ≥ 0.80** on 50 hand labels | E4, E9, and every published `nonverb` |
| 🟠 | **F17: extraction accuracy on GSM8K** — never measured; F14 rests on it | F14 claim |
| 🟡 | GSM8K expansion to ~3000 for a third domain | 3-domain claim |
| 🟡 | Second hint type (sycophancy) for E6b's two-way decomposition | main figure |
| 🟡 | Venue re-selection now the original deadlines have passed | submission |

### Local smoke tests *(code validation only — rates are meaningless)*

| Run | Model | Result |
|---|---|---|
| smoke1 | Qwen2.5-0.5B, CPU, 80 tok | Exposed **F7** (truncation ≡ instability) |
| smoke2 | Qwen2.5-0.5B, CPU, 400 tok | Pipeline clean, exit 0; three failure modes correctly separated. Exposed **F8** (parser gap) |

---

## 8. Changelog

- **22 Aug** — Repo initialised. Plan v2.0 → v2.1 (venue §0, 6-day schedule, verified
  CFP facts). `prompts.py`, `build_datasets.py`, `model_utils.py`, `run_e0_pilot.py`,
  `inspect_cots.py`, `inspect_unparsed.py` written and locally verified. Datasets
  built (600×3). Alignment 3600/3600. Findings F1–F11. `setup_remote.sh` rewritten
  for shared-server use.
