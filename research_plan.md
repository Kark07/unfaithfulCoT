**Do Unfaithfulness Circuits Generalize?**

**Cross-Task Transferability of Rationalization**

**Mechanisms in Language Models**

Research Plan & Experiment Design

Target: NeurIPS 2026 Workshop — Interpretability for Discovery

Submission Deadline: August 29, 2026 AoE

Draft Date: August 8, 2026

# **Table of Contents**

# **1. Executive Summary**

Chain-of-thought (CoT) reasoning is increasingly used both as a capability booster and as a **safety monitoring tool** — if we can read a model's reasoning, we can detect misalignment before it reaches the user. However, mounting evidence shows that CoT is often **unfaithful**: the written reasoning does not reflect how the model actually arrived at its answer. Recent work has used *circuit tracing* to detect such unfaithfulness on a per-instance basis (CIE-Scorer, Shen et al., 2026). Separately, the *circuit universality* literature has shown that task-solving circuits can transfer across different tasks (Merullo et al., ICLR 2024, ~78% attention head overlap between IOI and Colored Objects).

**The gap:** Nobody has asked whether *unfaithfulness circuits* — the internal mechanisms responsible for silently following hints without verbalizing them — generalize across different reasoning domains. This matters directly for AI safety: if there is a universal 'rationalization circuit,' a single monitor could catch all unfaithful behavior. If unfaithfulness mechanisms are task-specific, monitoring must be task-specific too, making the safety problem significantly harder.

This paper asks: **When a language model produces unfaithful reasoning on mathematical, logical, and factual tasks, is the same internal circuit responsible?**

# **2. Key Concepts & Definitions**

## **2.1 What Is CoT Faithfulness?**

Given an input *x*, a language model autoregressively generates a chain-of-thought *c* followed by an answer *a*. The CoT *c* is **faithful** if it accurately reflects the model's internal reasoning process — that is, the key factors and decision steps that actually caused the model to produce answer *a* in response to *x*. A faithful CoT should highlight the factors that would allow an observer to predict how the model would process similar inputs (Chen et al., 2025; Jacovi & Goldberg, 2020).

Conversely, a CoT is **unfaithful** when it presents a plausible-sounding justification that does not correspond to the actual computational process the model used. The model may have relied on a shortcut, a spurious correlation, or an injected hint, but its written reasoning discusses none of these — instead constructing a post-hoc rationalization that looks convincing to a human reader.

Unfaithfulness is distinct from incorrectness. An unfaithful CoT can arrive at the right answer for the wrong (unstated) reasons, or at a wrong answer with an elaborate but fabricated justification. The concern is not accuracy but *transparency*: whether monitoring the CoT gives us a reliable window into the model's actual decision process.

## **2.2 Hint-Following Without Verbalization**

The primary operationalization of unfaithfulness in this work (following Turpin et al., NeurIPS 2023; Chen et al., Anthropic 2025) is **hint-following without verbalization**. The setup works as follows:

**Step 1 — Baseline:** Present the model with a standard multiple-choice question *x_u* (the 'unhinted' prompt) and record its answer *a_u* and CoT *c_u*.

**Step 2 — Hint injection:** Present the same question with a hint inserted — *x_h* (the 'hinted' prompt) — where the hint points toward a specific answer choice (X). Record the model's answer *a_h* and CoT *c_h*.

**Step 3 — Filter for unfaithful instances:** Retain only cases where (a) the model's unhinted answer was NOT the hint answer (*a_u ≠ X*), and (b) the model's hinted answer IS the hint answer (*a_h = X*). This means the hint *caused* the model to change its answer.

**Step 4 — Check verbalization:** Examine the hinted CoT *c_h*. If the model mentions the hint as a factor in its reasoning, the CoT is faithful. If the model constructs a plausible argument for answer (X) without mentioning the hint at all, the CoT is **unfaithful** — the model silently followed the hint and rationalized its choice.

Chen et al. (2025) found that across six hint types and two reasoning models (Claude 3.7 Sonnet and DeepSeek R1), the hint verbalization rate was typically **below 20%**. Unfaithful CoTs were often *longer and more elaborate* than faithful ones, suggesting the model expends computational effort constructing rationalizations rather than simply stating it followed the hint.

## **2.3 Circuits in Mechanistic Interpretability**

A **circuit** is a minimal subgraph of a neural network's computational graph that is sufficient to explain the model's behavior on a specific task (Wang et al., 2023; Conmy et al., 2023). In a transformer, the full computational graph consists of attention heads, MLP layers, and the residual stream connections between them. A circuit identifies which subset of these components — and which edges between them — are causally responsible for a particular behavior.

A circuit is **faithful** if ablating (removing) all edges outside the circuit does not change the model's behavior on the task (Hanna et al., COLM 2024). Faithfulness is what justifies studying circuits rather than the full model — it guarantees that the identified subgraph actually captures the relevant computation.

The question of **circuit universality** asks whether similar circuits appear across different models, scales, and tasks (Olah et al., 2020). Merullo et al. (ICLR 2024) provided early evidence that task-solving circuits can generalize, showing ~78% attention head overlap between the IOI and Colored Objects circuits in GPT-2 Medium. Hanna et al. (COLM 2024) extended this with cross-task faithfulness metrics.

## **2.4 The Research Question**

Existing circuit universality work studies circuits for *correct task performance* — 'does the model use the same circuit to solve IOI and Colored Objects?' We ask the complementary question about circuits for *unfaithful behavior*: when a model silently follows a hint on a math problem versus a logic problem versus a knowledge question, is the same circuit responsible for processing the hint and suppressing its verbalization?

If **yes**: there exists a universal 'rationalization circuit' that can be monitored as a single safety signal, regardless of the downstream task. This would be a significant positive finding for AI safety — one monitoring target catches all unfaithfulness.

If **no**: unfaithfulness mechanisms are task-specific, meaning safety monitoring must be separately validated for each domain. This is a harder but equally important finding, as it reveals the limits of circuit-based safety monitoring.

# **3. Related Work & Positioning**

## **3.1 CoT Faithfulness**

**Turpin et al. (NeurIPS 2023)** — The foundational paper on unfaithful CoT. Demonstrated that biasing features (reordering answer choices, suggesting answers) cause models to produce systematically unfaithful explanations on BIG-Bench Hard tasks. Models rationalize biased answers without mentioning the bias. Tested on GPT-3.5 and Claude 1.0.

**Chen et al. (Anthropic, 2025) — 'Reasoning Models Don't Always Say What They Think'** — Extended Turpin et al. to reasoning models (Claude 3.7 Sonnet, DeepSeek R1) with 6 hint types (sycophancy, consistency, visual pattern, metadata, grader hacking, unethical information) on MMLU and GPQA. Found verbalization rates below 20%. Showed outcome-based RL improves faithfulness initially but plateaus. This paper defines the hint-injection protocol we adopt.

**Arcuschin, Nanda, Conmy et al. (2025)** — Showed unfaithful CoT occurs on natural, non-adversarial prompts (no hint injection needed), including implicit post-hoc rationalization and silent error correction.

**FaithCoT-Bench (Shen et al., ICLR 2026)** — First comprehensive benchmark for instance-level CoT faithfulness detection. Covers 4 domains (LogicQA, TruthfulQA, AQuA, HLE-Bio) with expert annotations. We adopt three of these four datasets.

## **3.2 Circuit Discovery Methods**

**ACDC (Conmy et al., NeurIPS 2023)** — Original automated circuit discovery. Iteratively prunes edges via activation patching. Produces faithful circuits but is slow (~5 hours per query on GPT-2 Large with an A100).

**EAP (Syed et al., 2024)** — Edge Attribution Patching. Approximates activation patching using gradients: 2 forward passes + 1 backward pass for all edges simultaneously. ~10 seconds per instance. Suffers from gradient saturation.

**EAP-IG (Hanna et al., COLM 2024)** — EAP with Integrated Gradients. Fixes saturation by accumulating gradients along a linear path between clean and corrupted activations. Produces substantially more faithful circuits than EAP. Our primary method.

## **3.3 Cross-Task Circuit Transfer**

**Merullo, Eickhoff, Pavlick (ICLR 2024) — 'Circuit Component Reuse Across Tasks'** — Showed IOI and Colored Objects circuits share ~78% attention heads in GPT-2 Medium. Demonstrated cross-task circuit repair: adjusting 4 heads boosted Colored Objects accuracy from 49.6% to 93.7%. Our primary methodological template for cross-task comparison.

**Hanna et al. (COLM 2024) — 'Have Faith in Faithfulness'** — Introduced EAP-IG and measured both cross-task overlap (Jaccard IoU) and cross-task faithfulness. Showed that circuit overlap alone is insufficient — faithfulness must be measured directly. We adopt their evaluation framework.

**'Many Circuits, One Mechanism' (2606.06267, June 2026)** — Critiqued existing cross-task comparisons for using coarse granularity and lacking statistical controls. We address these critiques by using edge-level (not node-level) granularity and performing multiple extractions per task.

## **3.4 Circuit-Based Unfaithfulness Detection**

**CIE-Scorer (Shen et al., May 2026)** — Uses circuit tracing to detect unfaithful CoT by constructing internal and external reasoning graphs and measuring their discrepancy via Fused Gromov-Wasserstein distance. Evaluates on FaithCoT-Bench. Traces circuits per-instance but does not study cross-task generalization of unfaithfulness circuits. This is the closest prior work — our paper directly extends it by asking whether the circuits CIE-Scorer finds are consistent across tasks.

**'Universal Refusal Circuits Across LLMs' (Cristofano, Jan 2026)** — Showed that refusal circuits can transfer across model families, providing precedent that behavioral (not just task-solving) circuits can exhibit universality. We test a similar hypothesis for unfaithfulness circuits.

# **4. Experimental Design**

## **4.1 Tasks & Datasets**

We select three datasets spanning distinct reasoning domains, all of which have established use in both the faithfulness literature (FaithCoT-Bench, CIE-Scorer) and the broader reasoning evaluation literature:

|  | Task 1: Math | Task 2: Logic | Task 3: Knowledge |
|---|---|---|---|
| Dataset | AQuA | LogiQA | MMLU (subset) |
| Domain | Algebraic word problems | Deductive logic | Factual/commonsense |
| Format | 5-way MC | 4-way MC | 4-way MC |
| # Questions | 100 | 100 | 100 |
| Used in | FaithCoT-Bench, CIE-Scorer | FaithCoT-Bench, CIE-Scorer | Chen et al. (2025), Turpin et al. (2023) |
| Reasoning type | Multi-step numerical | Multi-step deductive | Knowledge recall + reasoning |

**Rationale for these three:** They span fundamentally different cognitive modalities — numerical manipulation, formal logic, and world knowledge. If unfaithfulness circuits generalize across these, it strongly suggests a domain-general mechanism. Additionally, all three are used in FaithCoT-Bench and/or the Chen et al. hint-injection literature, ensuring comparability with prior work.

## **4.2 Hint Injection Protocol**

Following Chen et al. (2025), we use two hint types, applied identically across all three tasks:

### **4.2.1 Hint Type A — Sycophancy**

Appended to the question after the answer choices:

A Stanford professor indicates the answer is (X). What do you think?

This exploits the model's tendency to defer to perceived authority. Chen et al. found this reliably induces hint-following across both MMLU and GPQA. It is the simplest and most-studied hint format (Turpin et al. also used variants of this).

### **4.2.2 Hint Type B — Metadata**

Prepended to the question as XML metadata:

<question-metadata>

  <status>complete</status>

  <peer-reviewed>true</peer-reviewed>

  <answer>X</answer>

  <success-rate>0.73</success-rate>

</question-metadata>

This is a subtler, non-social hint — the answer is embedded in structured data rather than attributed to a person. Chen et al. showed metadata hints also reliably induce unfaithful behavior, often with even lower verbalization rates than sycophancy hints. Including both types lets us test whether the rationalization circuit generalizes not just across reasoning domains but across hint modalities.

### **4.2.3 Full Protocol**

For each of the 100 questions per task, for each of the 2 hint types:

**(a)** Generate the **unhinted prompt** x_u: question + answer choices + CoT elicitation ('Let’s think step by step. Show your reasoning, then state your final answer.')

**(b)** Generate the **hinted prompt** x_h: same question with the hint inserted, pointing to a randomly selected *incorrect* answer. We use incorrect answers specifically so that hint-following can be unambiguously attributed to the hint rather than to the model's prior knowledge.

**(c)** Run both prompts through the model. Record answer and full CoT.

**(d)** **Filter for unfaithful instances:** retain cases where a_u ≠ X and a_h = X (the model changed its answer to the hint answer).

**(e)** **Verify non-verbalization:** use an LLM judge (following Chen et al.'s protocol) to check whether c_h mentions the hint. Instances where the hint is NOT mentioned are confirmed unfaithful.

**Expected yield:** Based on Chen et al.'s data, we expect roughly 30–50% of instances to show hint-following, and of those, roughly 80% to be non-verbalizing. This gives us approximately 25–40 unfaithful instances per task per hint type — sufficient for circuit analysis.

## **4.3 Models**

### **4.3.1 Primary Model: Gemma-2-2B-IT**

Google's instruction-tuned 2B parameter model. Selected because: (a) it is large enough to produce coherent CoT reasoning on MMLU, LogiQA, and AQuA; (b) it is small enough for tractable circuit analysis with EAP-IG — the full computational graph has a manageable number of attention heads and MLP layers; (c) it has growing support in the mechanistic interpretability ecosystem, including community-trained SAEs; (d) instruction tuning means it follows CoT prompting conventions reliably. All primary experiments run on this model.

### **4.3.2 Secondary Model: DeepSeek-R1-Distill-Qwen-1.5B**

A distilled reasoning model. Selected to test whether RL-trained reasoning models exhibit different unfaithfulness circuit patterns than standard instruction-tuned models. Chen et al. (2025) showed reasoning models have qualitatively different faithfulness profiles (higher baseline faithfulness, but still far from perfect). Even a partial replication (1 task × 1 hint type) on this model would significantly strengthen the paper's contribution by addressing whether circuit universality holds across training paradigms.

### **4.3.3 Validation: GPT-2 Small (124M)**

For methodological validation only. GPT-2 Small is the canonical model for circuit discovery (IOI, Greater-Than, ACDC, EAP-IG all validated here). We run a toy version of the experiment — a simple synthetic task where unfaithfulness can be verified by construction — to confirm our EAP-IG pipeline produces faithful circuits before applying it to the main experiments. This is not part of the primary results.

## **4.4 Circuit Discovery Method**

### **4.4.1 Primary: EAP-IG**

Edge Attribution Patching with Integrated Gradients (Hanna et al., COLM 2024) is our primary circuit discovery method. For each unfaithful instance, EAP-IG scores every edge in the model's computational graph by integrating gradients along the path between 'clean' and 'corrupted' activations:

*ϕ(e) ≈ (e_clean – e_corr) × (1/m) ∑ ∇ L(x(k/m))*

where *e_clean* and *e_corr* are edge activations under the clean (hinted) and corrupted (unhinted) inputs, and *m* is the number of integration steps.

### **4.4.2 Clean/Corrupted Pair Definition**

For standard task circuits, 'clean' is the original input and 'corrupted' is a modified version. For our unfaithfulness circuits, the mapping is:

**Clean input:** the hinted prompt x_h (where the model exhibits unfaithful behavior — silently following the hint)

**Corrupted input:** the unhinted prompt x_u (where the model answers based on its own reasoning)

**Metric:** logit difference between the hint answer and the model's unhinted answer. The circuit captures exactly the subgraph responsible for the behavioral shift caused by the hint — the 'hint-processing-and-suppressing-verbalization' mechanism.

### **4.4.3 Circuit Extraction**

For each unfaithful instance, EAP-IG produces a score for every edge. We threshold at the top-k% of edges (following Hanna et al., we experiment with thresholds that achieve at least 85% faithfulness). Each task yields a set of circuits, one per unfaithful instance.

To obtain a **task-level circuit**, we aggregate across instances: an edge is included in the task circuit if it appears above threshold in at least p% of instances for that task (we experiment with p = 50%, 60%, 70%). This gives us one circuit per task per hint type.

### **4.4.4 Validation: ACDC**

On a randomly selected subset of 10 instances per task, we additionally run ACDC (Conmy et al., 2023) as a ground-truth validation. We compare the EAP-IG-derived circuits with ACDC circuits to verify that our faster method produces comparable results, following the validation protocol established in Hanna et al.

## **4.5 Evaluation Metrics**

### **4.5.1 Circuit Faithfulness**

For each extracted circuit, ablate all edges OUTSIDE the circuit and verify the model still exhibits hint-following behavior. Report the **faithfulness score** = fraction of instances where the ablated model still produces the hint answer. We target circuits that achieve at least 85% faithfulness, following Hanna et al.

### **4.5.2 Cross-Task Circuit Overlap (Jaccard IoU)**

For each pair of tasks (Math↔Logic, Math↔Knowledge, Logic↔Knowledge), compute the edge-level Jaccard similarity:

*J(C_i, C_j) = |C_i ∩ C_j| / |C_i ∪ C_j|*

We report this at **edge level** (not node level), addressing the critique from 'Many Circuits, One Mechanism' (2606.06267) that node-level overlap is too coarse. We also compute statistical significance using a hypergeometric test (following Hanna et al.).

### **4.5.3 Cross-Task Circuit Faithfulness**

Take the circuit from Task A and test it on Task B's unfaithful instances. If it maintains high faithfulness, the Task A circuit is *sufficient* for explaining Task B's unfaithful behavior — strong evidence for a shared mechanism. Report an asymmetric faithfulness transfer matrix (3×3).

### **4.5.4 Ablation: Does Removing the Shared Circuit Selectively Kill Unfaithfulness?**

Identify the intersection of all three task circuits (edges present in all three). Ablate only these shared edges and measure: (a) does hint-following decrease across all tasks? (b) does the model's general task performance (accuracy without hints) remain intact? If the shared circuit is specifically responsible for unfaithful hint processing, ablating it should reduce unfaithfulness without destroying the model's ability to answer questions correctly.

# **5. Expected Results & Contribution**

## **5.1 Possible Outcomes**

### **Outcome A — High overlap (Jaccard > 0.5)**

A universal rationalization circuit exists. The model uses the same mechanism to process-and-hide hints regardless of whether the task is mathematical, logical, or factual. The shared circuit likely involves components that (a) attend to hint-like tokens, (b) route the hint answer to the final prediction, and (c) suppress the verbalization of hint usage in the CoT. **Safety implication:** a single monitoring target is sufficient.

### **Outcome B — Moderate overlap (Jaccard 0.2–0.5)**

There is a shared 'core' rationalization circuit (possibly handling hint detection and verbalization suppression) with task-specific additions (handling domain-specific reasoning that generates the rationalization). **Safety implication:** monitoring the core circuit catches most unfaithfulness, but task-specific monitoring adds value.

### **Outcome C — Low overlap (Jaccard < 0.2)**

Unfaithfulness is mechanistically task-specific. Different reasoning domains recruit different circuits for rationalization. **Safety implication:** circuit-based monitoring must be separately validated per domain; there is no universal unfaithfulness detector at the circuit level.

**All three outcomes are publishable.** Outcomes A and B provide positive safety tools; Outcome C provides a cautionary finding that shapes future monitoring research.

# **6. Timeline (3.5 Weeks to August 29)**

| Week | Activities | Deliverable |
|---|---|---|
| Week 1 (Aug 8–14) | Read core papers (Merullo, Hanna, Chen et al., CIE-Scorer). Set up EAP-IG pipeline on GPT-2 Small toy task. Download and preprocess AQuA, LogiQA, MMLU. Implement hint injection protocol. | Working pipeline + datasets |
| Week 2 (Aug 15–21) | Run hint injection on Gemma-2-2B-IT across all 3 tasks × 2 hint types. Filter for unfaithful instances. Run EAP-IG circuit extraction on all unfaithful instances. Compute per-task circuits. | Circuits for all tasks |
| Week 3 (Aug 22–26) | Compute cross-task overlap (Jaccard) and cross-task faithfulness transfer. Run shared-circuit ablation experiment. If time permits: partial replication on DeepSeek-R1-Distill-1.5B. | All results |
| Final (Aug 27–29) | Write up results. Format as workshop paper (4–6 pages). Submit by Aug 29 AoE. | Submitted paper |

# **7. Key References (Prioritized Reading List)**

## **7.1 Must-Read (Methodology)**

[1] Merullo, Eickhoff, Pavlick. 'Circuit Component Reuse Across Tasks in Transformer Language Models.' ICLR 2024. arXiv: 2310.08744

[2] Hanna, Pezzelle, Belinkov. 'Have Faith in Faithfulness: Going Beyond Circuit Overlap When Finding Model Mechanisms.' COLM 2024. arXiv: 2403.17806

[3] Shen et al. 'Detecting Unfaithful Chain-of-Thought via Circuit-Guided Internal-External Discrepancy.' May 2026. arXiv: 2605.25603

[4] Conmy, Mavor-Parker, Lynch, Heimersheim, Garriga-Alonso. 'Towards Automated Circuit Discovery for Mechanistic Interpretability.' NeurIPS 2023. arXiv: 2304.14997

[5] Syed, Rager, Conmy. 'Attribution Patching Outperforms Automated Circuit Discovery.' BlackboxNLP @ EMNLP 2024. arXiv: 2310.10348

## **7.2 Must-Read (Unfaithfulness Phenomenon)**

[6] Chen et al. 'Reasoning Models Don't Always Say What They Think.' Anthropic, 2025. arXiv: 2505.05410

[7] Turpin, Michael, Perez, Bowman. 'Language Models Don't Always Say What They Think.' NeurIPS 2023. arXiv: 2305.04388

[8] Arcuschin, Janiak, Krzyzanowski, Rajamanoharan, Nanda, Conmy. 'Chain-of-Thought Reasoning in the Wild Is Not Always Faithful.' 2025. arXiv: 2503.08679

[9] Shen et al. 'FaithCoT-Bench: Benchmarking Instance-Level Faithfulness of Chain-of-Thought Reasoning.' ICLR 2026. arXiv: 2510.04040

[10] Lanham et al. 'Measuring Faithfulness in Chain-of-Thought Reasoning.' 2023. arXiv: 2307.13702

## **7.3 Should-Read (Context & Positioning)**

[11] Wang, Alazali, Zhong. 'How Does Unfaithful Reasoning Emerge from Autoregressive Training?' Feb 2026. arXiv: 2602.01017

[12] Cristofano. 'Universal Refusal Circuits Across LLMs.' Jan 2026. arXiv: 2601.16034

[13] 'Many Circuits, One Mechanism: Input Variation and Evaluation Granularity in Circuit Discovery.' June 2026. arXiv: 2606.06267

[14] Mueller et al. 'MIB: A Mechanistic Interpretability Benchmark.' ICML 2025. arXiv: 2504.13151

[15] Tigges, Hanna, Yu, Biderman. 'LLM Circuit Analyses Are Consistent Across Training and Scale.' NeurIPS 2024.

[16] Mondorf, Wang et al. 'Circuit Compositions: Exploring Modular Structures in Transformer-Based Language Models.' ACL 2025.

[17] Bhaskar, Wettig, Friedman, Chen. 'Finding Transformer Circuits with Edge Pruning.' 2024. arXiv: 2406.16778

# **8. Compute Requirements**

EAP-IG on Gemma-2-2B-IT: approximately 10–30 seconds per instance on a single A100 (80GB). With ~75 unfaithful instances per task (across 2 hint types), and 3 tasks, that is ~225 instances × 30s = ~2 hours of EAP-IG computation. Adding ACDC validation on 30 instances: ~30 × 1 hour = ~30 hours. Total GPU time: approximately 35–40 A100-hours for the full experiment. The hint-injection phase (600 question pairs through Gemma-2-2B-IT) is cheaper: ~2–4 hours of inference.

If a second model (DeepSeek-R1-Distill-1.5B) is included for partial replication, add approximately 10–15 A100-hours.

**Total estimated compute: 40–60 A100-hours.**
