# Uplift Modeling Learning Notes

A free-form log of concepts, questions, and takeaways picked up while reading docs, papers, and building the project.

---

## Uplift Modeling vs Individual Treatment Effects

These two terms are often used interchangeably in practice, but they sit at different levels of abstraction.

**ITE (Individual Treatment Effect)** is the theoretical quantity we care about:

```
ITE_i = Y_i(1) - Y_i(0)
```

where `Y_i(1)` is user i's outcome if treated, and `Y_i(0)` is the outcome if not treated. The fundamental problem of causal inference is that we can only ever observe one of these two potential outcomes for any given user. ITE is therefore never directly observable.

**CATE (Conditional Average Treatment Effect)** is the best estimable proxy for ITE:

```
CATE(x) = E[Y(1) - Y(0) | X = x]
```

Instead of asking "what is the effect for this exact user", we ask "what is the average effect for users with features similar to x". This is estimable from data.

**ATE (Average Treatment Effect)** is the population-level average:

```
ATE = E[Y(1) - Y(0)]
```

This is a single number, easy to compute from a randomized experiment as the difference in group means.

**Uplift Modeling** is the set of ML methods used to estimate CATE from data. The output of any uplift model (T-Learner, X-Learner, etc.) is an estimate of CATE, often called the "uplift score."

### Four user segments

Uplift modeling segments users by how they respond to treatment:

| Segment | Behavior | Action |
|---|---|---|
| Persuadables | Convert only if treated | Target these |
| Sure Things | Convert regardless of treatment | Wasted spend to target |
| Lost Causes | Never convert regardless of treatment | Wasted spend to target |
| Sleeping Dogs | Convert less when treated (negative effect) | Actively avoid targeting |

Only persuadables generate incremental value from an ad. Targeting sure things wastes budget on users who would have converted anyway. Targeting sleeping dogs actively harms conversion. The goal of uplift modeling is to identify persuadables and rank them to the top of the targeting list.

### A critical distinction: data source

The Diemert et al. paper formalizes **uplift modeling (UM)** and **ITE prediction** as two distinct problem setups, and the key difference is where the data comes from.

**Uplift Modeling assumes RCT data.** Treatment was randomly assigned (an A/B test), so by design there is no confounding: the treatment group and control group are statistically identical in expectation. The Criteo dataset was collected from several randomized control trials for exactly this reason. With RCT data, you can directly estimate treatment effects by comparing treated vs control outcomes, and the only challenge is estimating heterogeneity across users.

**ITE Prediction is the more general problem, often applied to observational data.** Here, treatment was not randomly assigned. Users self-selected into treatment (e.g., they chose to click an ad, or a doctor chose to prescribe a drug based on patient condition). This introduces confounding: the treated group differs systematically from the control group in ways correlated with the outcome. You need additional methods to correct for this, such as propensity score weighting, doubly robust estimators, or instrumental variables.

In short:

| | Data source | Confounding concern | Typical methods |
|---|---|---|---|
| Uplift Modeling | RCT / A/B test | None by design | T-Learner, X-Learner, S-Learner |
| ITE Prediction | Observational | Must be corrected | DR-Learner, IPW, DML |

This project uses the Criteo RCT dataset, so we are in the uplift modeling setting. No confounding correction is needed, but the treatment/control imbalance (85% treated, 15% control) still matters for model training, which is why X-Learner is preferred over T-Learner here.

### The T ⊥ X assumption

In a properly run RCT, treatment assignment is random and unrelated to any user characteristics:

```
T ⊥ X
```

This means `P(T=1 | X=x) = P(T=1)` for all x. Knowing a user's features tells you nothing about whether they were treated. The propensity score (probability of being treated) is a constant across all users.

This assumption is what makes RCT data so powerful for causal inference. You can directly compare treated and control groups without worrying that the two groups are systematically different in ways that affect the outcome.

In observational data, `T ⊥ X` does not hold. Users with certain features are more likely to receive treatment (e.g. sicker patients are more likely to get a drug, high-intent users are more likely to see a retargeting ad). Propensity score methods try to recover a version of this assumption by conditioning: `T ⊥ Y(0), Y(1) | X`, meaning treatment is independent of potential outcomes given observed covariates. This is the unconfoundedness (or ignorability) assumption, and it requires that all confounders are observed, which is a strong and untestable claim.

For this project, `T ⊥ X` holds by design. The Criteo dataset was built from RCTs so the assumption is satisfied without any correction.

### Incrementality tests

An incrementality test is a specific type of RCT used in advertising. A random portion of the target population is held out from seeing ads (the control group), while the rest are exposed as usual (the treatment group). The difference in conversion or visit rates between the two groups measures the true incremental effect of the advertising, i.e. how many conversions were caused by the ad rather than would have happened anyway.

This is exactly how the Criteo dataset was collected. The held-out control group is what makes causal estimation possible: without it, you only observe users who were exposed, and you cannot separate the ad's effect from the baseline behavior of users who were going to convert regardless.

---

## Meta-Learner Models

Meta-learners are a family of uplift modeling methods that decompose the CATE estimation problem into standard supervised learning sub-problems. Instead of building a custom causal model from scratch, they reuse off-the-shelf ML models (like LightGBM) as building blocks.

### T-Learner

Train two separate models: one on treated users predicting `visit`, one on control users predicting `visit`. The uplift score is the difference in predicted probabilities:

```
uplift(x) = µ_t(x) - µ_c(x)
```

Simple and transparent, but it treats the two groups completely independently. When the groups are imbalanced (85% treated, 15% control here), the control model is trained on much less data and tends to be noisier.

### X-Learner

An extension of T-Learner designed for imbalanced groups (Künzel et al., 2019). It runs in four stages:

**Stage 1:** Train µ_t and µ_c exactly like T-Learner.

**Stage 2:** Compute pseudo-treatment effects per user by using the other group's model to impute the counterfactual outcome.
- For treated users: `D_t = Y_t - µ_c(X_t)`
- For control users: `D_c = µ_t(X_c) - Y_c`

**Stage 3:** Train two effect models (regressors) to predict the pseudo-effects.
- τ_t predicts D_t on treated users
- τ_c predicts D_c on control users

**Stage 4:** Combine with a propensity weight:
```
CATE(x) = g(x) · τ_c(x) + (1 - g(x)) · τ_t(x)
```
where `g(x) = P(T=1 | X=x)`. In RCT data this is a constant (~0.85 for the Criteo dataset).

The key advantage over T-Learner is that Stage 2 allows each group's model to borrow information from the other group when computing pseudo-effects, making better use of all available data. The uplift scores operate on a different scale from T-Learner (not constrained as calibrated probabilities) so the two methods cannot be compared by score values alone. Ranking quality via the Qini curve is the proper comparison.

---

## Evaluation: Why Standard AUC Fails and Why Qini AUC Works

### The problem with standard AUC

Standard ROC AUC answers: "Can the model rank users who visit above users who do not?" It requires a binary ground truth label per user.

This fails for uplift modeling for two reasons.

First, the label problem. The thing we want to rank users by — their individual treatment effect — is never observable. Each user is either treated or control, never both. So there is no per-user uplift ground truth to evaluate against.

Second, the wrong objective. A standard classifier trained to predict visit probability will rank "sure things" highest — users who would visit regardless of the ad. These users have zero incremental value. Targeting them wastes budget. The model optimizing standard AUC is solving the wrong problem.

### How the Qini curve works

The Qini curve sidesteps the unobservability problem by evaluating at the group level rather than the individual level.

The steps:

1. Rank all test users by predicted uplift score, highest first
2. Walk through the ranked list and at each step compute cumulative incremental visits: visits in treatment users so far minus visits in control users so far, adjusted for the treatment/control size ratio
3. Plot this against the fraction of users targeted
4. Compare to the diagonal, which represents random targeting (no model)

The key insight: even though individual uplift is unobservable, we can compare groups. If the model correctly puts persuadables at the top of the ranking, those top-ranked users will show a higher treatment vs control visit rate than lower-ranked users. A model that ranks randomly will show no such pattern.

The **Qini AUC** is the area between the model curve and the random baseline. A score of 0 means the model is no better than random. A higher score means the model more effectively concentrates persuadables at the top of the ranking.

### Business interpretation

The Qini curve answers a practical question: "If I can only afford to target X% of my user base, how much of the total possible incremental uplift can I capture?" A perfect model captures all uplift by targeting only the persuadables. A random model captures uplift proportionally to the fraction targeted. The gap between the two is the value the model adds.

This directly maps to budget efficiency — the goal of uplift modeling in advertising.

---

## Two datasets released by Diemert et al.

The paper releases two distinct datasets and it is important to know which one this project uses and why.

**CRITEO-UPLIFTv2** is what this project uses. It contains 13.9M rows of real user data collected from several RCT incrementality tests run by Criteo. Treatment was randomly assigned, outcomes (visit, conversion) are genuine observed behavior, and T ⊥ X holds by design. Because the true ITE per user is unobservable (fundamental problem of causal inference), the evaluation metric is AUUC rather than PEHE. The v2 label matters: v1 had a flaw where different incrementality tests had different treatment ratios, allowing a model to implicitly learn which test an instance came from. v2 fixed this by sub-sampling all tests to a uniform global treatment ratio of 85%.

**CRITEO-ITE** is a semi-synthetic companion dataset built on the same real features X. Outcomes are replaced with mathematically generated response surfaces (linear, exponential, multi-peaked), and treatment assignment is deliberately confounded to simulate an observational setting. Because outcomes are synthetic, the true ITE is known and PEHE can be computed directly. This dataset exists purely to enable ITE benchmarking, which is otherwise impossible on real data.

The core difference is real vs synthetic outcomes, and RCT vs confounded treatment assignment. Choosing CRITEO-UPLIFTv2 means we are in the uplift modeling setting, not ITE prediction, so our methods (T-Learner, X-Learner) and evaluation metric (AUUC / Qini AUC) are appropriate.

---

## AUUC vs Qini AUC

There are three related but distinct curves in uplift evaluation. All sort users by predicted uplift score descending, but they compute different y-axis values at each step k.

**Notation:**

At each step k (top-k users by predicted uplift score):
- `Y_T(k)`, `Y_C(k)` = cumulative visits among treated and control users in top-k
- `n_T(k)`, `n_C(k)` = number of treated and control users in top-k (within-k counts, vary with k)
- `N_T`, `N_C` = total treated and control users in the full dataset (global, fixed)
- `N` = `N_T + N_C` = total dataset size

**Rate version**

```
y(k) = Y_T(k) / n_T(k) - Y_C(k) / n_C(k)
```

The observed visit rate difference between treated and control within the top-k group. Decreases as k grows since higher-uplift users are ranked first. This is the building block for AUUC but is not itself what `uplift_auc_score` computes.

**AUUC**

```
y(k) = (Y_T(k) / n_T(k) - Y_C(k) / n_C(k)) × N
```

Takes the within-k uplift rate and scales it to all N users (treated + control) in the dataset. At k = N, y = ATE × N. `sklift.metrics.uplift_auc_score` computes the area under this curve minus the random baseline. Diemert et al. (Criteo paper) use AUUC as the standard evaluation metric for uplift models on RCT data.

**Qini curve**

```
y(k) = Y_T(k) - Y_C(k) × (N_T / N_C)
```

Adjusts cumulative control visits by the global treatment/control ratio, then subtracts from cumulative treatment visits. Approximately equivalent to `n_T(k) × uplift_rate`, scoping the result to the treated users in top-k. At k = N, y = ATE × N_T. `sklift.metrics.qini_auc_score` computes the area under this curve minus the random baseline.

**Conceptual difference between AUUC and Qini**

Both formulas are built on the same within-k uplift rate `Y_T(k)/n_T(k) - Y_C(k)/n_C(k)`. The difference is the population they scale to:

- AUUC multiplies by N (all users), so the ceiling represents incremental effect across the full population.
- Qini multiplies by approximately n_T(k), scaling to treated users only, so the ceiling represents incremental effect among the treated group.

This is why AUUC and Qini ceilings differ by a factor of approximately N / N_T = 1 / trmnt_ratio (e.g. 1/0.85 ≈ 1.18 under 85/15 imbalance). Both measure the same ranking quality; they differ only in whose incremental effect they are counting.

**Why absolute scalar values differ across the two metrics**

The two scalar scores are not directly comparable because they use different normalizations. At each step k, AUUC y-values are larger than Qini y-values by a factor of approximately `N / n_T(k)`. The two sklift functions also normalize the final area differently. As a result, AUUC and Qini AUC scores for the same model will differ in absolute value but rank models identically. Do not compare the two numbers to each other; only compare within the same metric across models.

**Why Qini is more stable under treatment/control imbalance**

With a small control group, `n_C(k)` is very small for early k, making `Y_C(k) / n_C(k)` noisy. The AUUC formula multiplies that noisy rate by N, amplifying the noise. The Qini formula uses the global ratio `N_T / N_C` as a fixed weight, which is less sensitive to small within-k control counts and produces a more stable curve under severe imbalance.

---

## Bootstrap and Sampling Distributions

Bootstrap resampling draws repeated samples with replacement from the observed data to approximate the sampling distribution of a statistic. Running 500 resamples produces 500 estimates of a metric (e.g. Qini AUC), and the spread of those estimates tells us how much the metric would vary across different test sets.

### When the bootstrap distribution is normal

By the Central Limit Theorem (CLT), the sampling distribution of statistics that aggregate many observations (sums, means, areas under curves) converges to normal as sample size grows. Qini AUC is essentially an area under a rank-ordered cumulative sum, close enough to a mean-like aggregation that the normal approximation holds well with large samples (279K rows here). The bootstrap distribution of Qini AUC should look like a roughly symmetric bell curve.

### When normality does not apply

The CLT applies to means and mean-like statistics. It does not guarantee normality for:

- **Median**: depends on the density at the median point, converges more slowly and less cleanly
- **Min / Max**: converge to extreme value distributions (Gumbel, Fréchet), not normal
- **Quantiles**: similar issue to median, especially in the tails
- **Ratios**: can be skewed when the denominator has high variance

For these statistics, bootstrap confidence intervals are still valid (the bootstrap does not require normality), but the distribution shape will not be a bell curve.

### Why bootstrap CIs are still valid without normality

The bootstrap does not rely on the CLT. It directly approximates the sampling distribution empirically, whatever shape that distribution takes. The 2.5th and 97.5th percentiles of the bootstrap distribution give a valid 95% CI regardless of whether the distribution is normal, skewed, or multimodal. This is one of the main advantages of bootstrap over parametric methods that assume normality.

---

## KS Statistic (Kolmogorov-Smirnov)

The KS statistic measures the maximum distance between two cumulative distribution functions (CDFs):

```
KS = max|CDF_A(x) - CDF_B(x)|
```

It ranges from 0 (identical distributions) to 1 (perfectly separated, no overlap). It is non-parametric: it makes no assumption about the shape of either distribution.

To compute it, sort all values across both groups and walk through them. At each value x, compute what fraction of group A and group B fall at or below x. The KS statistic is the largest gap you find anywhere along that walk.

Used in this project to rank features by how well they separate persuadables from low-uplift users. f8 and f2 scored ~0.999 (near-perfect separation). f11 scored 0.05 (barely separable). A large sample size makes p-values effectively zero for all features, so the statistic magnitude is what matters for ranking, not the p-value.

---

## References

- scikit-uplift official docs and tutorial: https://www.uplift-modeling.com/en/latest/
- Two models approaches: https://www.uplift-modeling.com/en/latest/user_guide/models/two_models.html
- A Large Scale Benchmark for Uplift Modeling (Diemert et al., AdKDD @ KDD 2018): https://arxiv.org/pdf/2111.10106
- Uber CausalML GitHub: https://github.com/uber/causalml
- Why Uplift Cannot Be Directly Validated (Matteocourthoud): https://matteocourthoud.github.io/post/evaluate_uplift/
- Booking.com Causal Inference Paper: https://arxiv.org/abs/2308.09066
- Dataset on HF: https://huggingface.co/datasets/criteo/criteo-uplift

