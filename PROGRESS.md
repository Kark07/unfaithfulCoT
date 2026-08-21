# Progress Log

Living tracker for the unfaithfulness-circuits project. Updated whenever
something is built, found, or decided.

- **Plan of record:** [research_plan_v2.md](research_plan_v2.md) (v2.1)
- **Superseded:** `research_plan.md` (v1, kept only for the §1 diff table)
- **Working deadline:** **28 Aug 2026 AoE** (earlier of the two candidate venues)
- **Last updated:** 22 Aug 2026 (repo initialised)

---

## 1. Status at a Glance

| Day | Date | Planned | Status |
|---|---|---|---|
| 1 | Aug 22 | Datasets, prompt contract, remote env, **E0 pilot + model gate** | 🟡 code done & verified locally; **E0 not yet run — needs the server** |
| 2 | Aug 23 | E1 full generation, E2 pipeline validation | ⬜ |
| 3 | Aug 24 | E1 judging + κ, E3 influence circuits, E4 suppression circuits | ⬜ |
| 4 | Aug 25 | E5 baselines, E6 overlap analysis | ⬜ |
| 5 | Aug 26 | E7 transfer, E8+E9 ablations, **venue locked** | ⬜ |
| 6 | Aug 27 | Figures, full draft, anonymize | ⬜ |
| 7 | Aug 28 | Revise + **submit** | ⬜ |

**Critical path right now:** get `setup_remote.sh` running on the shared A6000, then E0.
Nothing downstream is meaningful until E0 clears its gate.

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

### E0 — Pilot *(not yet run)*

| Task | trunc | stab | comp | elig | SWITCH | nonverb | U est | probeD |
|---|---|---|---|---|---|---|---|---|
| gsm8k_mc | | | | | | | | |
| logiqa | | | | | | | | |
| mmlu | | | | | | | | |

Model locked: ______  |  Disclosure arm: keep / drop  |  Venue: ______

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
