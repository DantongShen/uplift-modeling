# Methodology and Experiment Design

Transferable lessons about measurement, experiment design, and data discipline from `07_privacy_constraints.ipynb`. These are not notebook-specific results; they generalize across uplift modeling and causal inference work.

---

## 1. Ablation: the ground-truth judge of feature value

**Ablation = remove a component, retrain, and measure how much overall performance drops. The drop is the component's true contribution.**

```
Want to know f8's contribution?
→ Train with f8, measure Qini
→ Train without f8, measure Qini
→ The difference is f8's real value
```

Why ablation is the arbiter: its score comes from **real labels** (Qini on the held-out test set), an external judge. Importance scores and KS come from the model's own behavior (the model grading itself).

Two flavors used in the notebook:
- **Solo test**: train on the feature(s) alone; measures standalone signal (f8+f2 alone: Qini 0.0112, near random)
- **Removal test**: train on everything else; measures marginal contribution (removing f8/f2 at full scale: 0.0760 → 0.0707, only ~7%)

---

## 2. Reliance ≠ Value (two rulers measuring the same task)

Two ways to ask "is feature X important for uplift ranking?":

```
Ruler 1: ask the model what it uses (KS, feature importance)
  f8/f2 score: KS ≈ 0.999, top importance → "most important!"

Ruler 2: ablation with real labels
  f8/f2 standalone Qini: 0.0112 → nearly worthless
```

Same task, two rulers, opposite readings; one ruler is broken. Ruler 1 measures *reliance* (how much the model leans on a feature); only Ruler 2 measures *value* (what the feature actually contributes). A model can lean heavily on features that contribute almost nothing.

Hiring analogy: "who gets called on most in meetings" (reliance) vs "does revenue drop when they take a month off" (value). The loudest presenter may contribute nothing to sales.

---

## 3. Response model ≠ uplift model (two different tasks)

This explains *why* the model relied on worthless features. It is the mechanism behind Lesson 2's disconnect, but it is a separate lesson about **tasks**, not rulers:

```
Task A: who will visit?          f8/f2: ROC AUC = 0.937 (excellent)
Task B: whom does the ad change? f8/f2: Qini    = 0.0112 (useless)
```

The same feature can ace one task and fail the other. Outcome signal is not uplift signal: high-visit-probability users include Sure Things, whom the ad doesn't change.

**The causal chain connecting Lessons 2 and 3:**

```
f8/f2 carry strong OUTCOME signal (Task A strength)
→ X-Learner's first-stage μ models are outcome predictors,
  so they spend their splits on f8/f2 (reliance emerges)
→ but Task-A strength doesn't transfer to Task B (value ≈ 0)
→ Ruler 1 reports the reliance as "importance" → misdiagnosis
→ only ablation (Ruler 2) catches it
```

This is the field's founding lesson — response models are not uplift models — replayed at the individual-feature level.

---

## 4. Circular measurement

Any analysis of the form **"cut groups by the model's own scores → analyze the inputs"** measures the model's behavior, not the world.

- Case: KS between top/bottom uplift deciles gave f8 a score of 0.999 because the deciles were cut on a score that f8 drives. The result is true by construction, not a discovery.
- Extra wrinkle found in the data: importance rank and KS rank disagree (f0: importance #2, KS 0.82; f8: importance #3, KS 0.999), so KS doesn't even measure reliance cleanly; it mixes reliance with distributional artifacts of the features.
- General rule: score-derived groups can describe *what the model does*; claims about *what is true* need an external judge (real labels).

---

## 5. Regime-dependent conclusions

The same intervention can reverse direction under different conditions:

```
Remove f8/f2 under light config (10% subsample, 50 trees):  Qini +60%
Remove f8/f2 under full config (full data, 100 trees):      Qini  -7%
```

With ample capacity the model exploits dominant features safely; under constrained capacity they monopolize splits and crowd out genuine uplift signal. Lessons:

- An "improvement" discovered on a subsample/light config must be re-validated at full scale before acting on it.
- Privacy degradation (aggregation, censoring, feature loss) forces models into constrained regimes where dominance harm activates. Feature selection matters *more* as data degrades, not less.

---

## 6. Noise vs Bias: two different diseases

When an estimate degrades, always ask: **wider or crooked?**

```
NOISE (variance): estimate fluctuates around the truth; CIs widen as
  samples shrink, but the expectation stays correct.
  → Random subsampling, lossless aggregation. Cure: more data.

BIAS: estimate is systematically wrong in one direction; repeating the
  measurement never fixes it.
  → Threshold censoring (MNAR): low-visit cohorts are dropped
    systematically; the control arm (lower base rate) loses more cohorts,
    inflating its estimated rate faster → ATE is pushed DOWN, even to
    sign reversal (k=50, t=5: ATE = -0.0010 with 91% data loss).
  Cure: more data does NOT help; the missingness mechanism must be modeled.
```

The privacy-relevant punchline: aggregation adds noise; threshold censoring adds bias. They are different in kind, and the second is worse: it can make an effective ad campaign measure as harmful.

---

## 7. Controls: the universal adjudication tool

The question a control answers: **"is the pattern caused by my specific intervention, or would anything produce it?"**

- The KS-ordered ablation rose then collapsed; this was meaningless until random removal orders showed what "any ordering" produces.
- Pair with a variance baseline (Experiment A: refit the same config with different seeds) to establish the error bar first; only differences exceeding it deserve interpretation.
- Deep symmetry with the project itself: no control group → no incrementality; no control ordering → no evidence the KS ordering means anything. Same logic, different scale.

---

## 8. Cross-file data alignment: position is fragile, keys are identity

**The incident:** scores.parquet was saved from notebook 3's split with a reset RangeIndex. Notebook 7 re-split with different stratification and reset again. Same positions now pointed to different users. Applying masks across the two frames silently mixed unrelated people, and every feature KS collapsed to ≈ 0 (the exact signature of random grouping).

- pandas raises no error: when index label sets match, alignment "succeeds"; the row correspondence is just silently wrong.
- Rules:
  1. Pass row-level data across notebooks with an explicit stable key (row_id); join by key, never by position.
  2. Keep a permanent smoke alarm: e.g., KS of the uplift score itself between top/bottom groups must be ≈ 1.0 by construction. If it isn't, alignment is broken.
  3. Learn the noise signature: "everything ≈ 0 / pure noise" in a derived analysis often means broken alignment upstream, not a real null result.

---

## 9. Measurement honesty annotations (cheap to write, expensive to skip)

Every reported number should carry its trust boundary. Examples used:

- "Single random partition per (k, t); high-censoring estimates carry partition-to-partition variability."
- "Random subsampling ignores opt-in selection bias; results are a lower bound on measurement damage."
- "Large-sample p-values are uninformative; compare statistics directly."
- "Survivor-bias artifact; do not read as recovery" (k=100, t=10).
- "Single fit per configuration; read the 7% gap as directional."

These one-liners are what separate a research memo from a demo notebook.