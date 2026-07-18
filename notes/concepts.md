# Domain and Statistical Concepts

A reference log of concepts picked up while reading docs, papers, and building the uplift modeling project.

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

First, the label problem. The thing we want to rank users by (their individual treatment effect) is never observable. Each user is either treated or control, never both. So there is no per-user uplift ground truth to evaluate against.

Second, the wrong objective. A standard classifier trained to predict visit probability will rank "sure things" highest: users who would visit regardless of the ad. These users have zero incremental value. Targeting them wastes budget. The model optimizing standard AUC is solving the wrong problem.

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

This directly maps to budget efficiency: the goal of uplift modeling in advertising.

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

## Statistical Power and Minimum Detectable Effect

When measuring the ATE from a holdout experiment, the question is whether the observed difference between treatment and control is real or sampling noise. This requires a hypothesis test.

### Two-proportion z-test for ATE

The ATE estimate is the difference in observed visit rates:

```
ATE_hat = visit_rate_treatment - visit_rate_control
```

Under the null (true ATE = 0), this estimate has a standard error of:

```
SE = sqrt(p * (1-p) * (1/n_T + 1/n_C))
```

where p is the pooled visit rate and n_T, n_C are the treatment and control group sizes (pooled approximation, valid when the effect is small relative to p). The test statistic z = ATE_hat / SE follows a standard normal distribution under the null.

### Minimum Detectable Effect (MDE)

MDE is defined as a fixed multiple of the standard error:

```
MDE = (z_alpha + z_beta) * SE
    = (z_alpha + z_beta) * sqrt(p * (1-p) * (1/n_T + 1/n_C))
```

where z_alpha = 1.96 (two-sided alpha=0.05) and z_beta = 0.84 (80% power). MDE has no meaning on its own; it is SE scaled by a constant that encodes the desired power level. As N grows, SE shrinks and MDE shrinks with it.

### Relationship between MDE and ATE

The test statistic for the ATE estimate is `ATE_hat / SE`. Detection requires this ratio to exceed z_alpha. For 80% power to detect a true effect, the true ATE must satisfy:

```
true_ATE / SE  >  z_alpha + z_beta
```

Rearranging: `true_ATE > (z_alpha + z_beta) * SE = MDE`. So "MDE < ATE" is shorthand for "the signal-to-noise ratio ATE/SE is large enough to detect the effect at 80% power."

```
MDE < ATE  →  ATE/SE large enough; detects the effect 80% of the time
MDE > ATE  →  ATE/SE too small; effect is real but lost in sampling noise
MDE = ATE  →  exactly at the 80% power boundary
```

N_min is the sample size where MDE equals ATE exactly. Solving for N:

```
N_min = (z_alpha + z_beta)^2 * p*(1-p) * (1/k + 1/(1-k)) / ATE^2
```

where k is the treatment allocation fraction. Increasing N shrinks SE, which shrinks MDE, until MDE drops below ATE.

### Effect of treatment/control imbalance

The MDE formula contains (1/n_T + 1/n_C). For a fixed total N:

```
50/50 split:  1/(0.5N) + 1/(0.5N) = 4/N
85/15 split:  1/(0.85N) + 1/(0.15N) ≈ 7.84/N
```

The 85/15 imbalance increases the variance term by a factor of 7.84/4 = 1.96, so MDE is sqrt(1.96) ≈ 1.4x larger. Since MDE ∝ 1/sqrt(N), a 1.4x MDE penalty translates to a 1.4² ≈ 2x sample-size penalty: the same detection threshold requires roughly 2x more total users than a balanced experiment. Statistical power is bottlenecked by the smaller group: in an 85/15 split, adding more treated users does almost nothing for power; only expanding the control group helps.

In the privacy constraints notebook: N_min = 25,991 under 85/15. The equivalent under 50/50 would be roughly 13,300 — the imbalance costs ~2x in required sample size (the 1.4x MDE penalty squared).

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

## Privacy-Era Ad Measurement: ATT, SKAN, and AAK

### The Transition: IDFA to ATT to SKAN to AAK

```
IDFA era: user-level tracking, precise attribution and targeting
    ↓
2021, ATT launches (iOS 14.5): users can refuse tracking; 70-85% do
    ↓
SKAN becomes the default: aggregated data, coarsened conversion values,
privacy thresholds, delayed postbacks
    ↓
2024+, AAK takes over: more flexible, but the
"aggregation + privacy threshold" paradigm is unchanged
```

**IDFA (Identifier for Advertisers)**: a unique per-device advertising ID from Apple that enabled cross-app tracking of the same user, the basis of user-level attribution.

**ATT (App Tracking Transparency)**: the permission prompt introduced in iOS 14.5. An app must obtain explicit user consent before reading the IDFA. Current opt-in rates are only ~15-30%, meaning 70-85% of iOS users are invisible to advertisers.

**SKAN** is the aggregated measurement scheme built for that 70-85% who opted out.

### What Is SKAN

SKAdNetwork (StoreKit Ad Network) is Apple's privacy-safe mobile attribution framework. It reports deterministic, aggregated attribution data. Advertisers learn "this campaign drove N installs" but receive no device-level or user-level data.

**Attribution flow:**

```
impression ≥3s → click → StoreKit renders → install → first launch within window
   (view)      (engage)     (render)       (install)   (attribution + postback)
```

The 3-second timer is an anti-fraud mechanism: an ad must be genuinely displayed for at least 3 seconds to count as a view. StoreKit renders an App Store card without leaving the current app, cryptographically signed by the ad network.

**Postback** is the core concept: an anonymous attribution receipt sent by the iOS device to the ad network. Three key properties:

1. The sender is the device (iOS itself), not the app or a server. Attribution completes on-device; no party can tamper with it.
2. Contains no user or device identifiers. A postback can never be linked to a specific user.
3. Delayed delivery: minimum 24 hours plus random jitter, preventing user identity inference from timing.

### The Three Privacy Mechanisms

**Crowd anonymity tiers (Tier 0-3)**: SKAN 4.0 assigns a privacy tier to every install based on campaign install volume. Large volume receives Tier 3 (fullest postback fields); small volume receives Tier 0 (almost nothing). The core philosophy: how large a crowd you hide in determines your data granularity.

**Conversion value degradation:**

```
information:  6 bits         ~1.6 bits        0 bits
              Fine CV    →   Coarse CV    →   null
              (0-63)         (low/med/high)   (nothing)
              large volume   medium volume    small volume
```

CV encodes "what the user did after installing" using a scheme designed by the advertiser. As crowd size falls, the value coarsens and eventually disappears.

**Hierarchical source identifier**: SKAN 4.0 allows up to a 4-digit source ID (10,000 combinations), but how many digits are actually received is determined dynamically by crowd size. Small volume receives only 2 digits (100 combinations, equivalent to SKAN 2.0 levels).

Unifying rule: every data dimension in SKAN follows the same law. Crowd size determines data granularity. CV and source ID are isomorphic degradation mechanisms.

### SKAN Version History

| Version | Year | Key changes |
|---|---|---|
| 1.0 | 2018 | Basic install attribution only; barely used (IDFA was still freely available) |
| 2.0 | 2020 | First production-grade version: CV introduced (6-bit), 100 campaign IDs, 24h timer |
| 3.0 | 2021 | Official view-through attribution; did-win field (losing networks also receive a postback) |
| 4.0 | 2022 | Largest upgrade: hierarchical source ID, 3 postbacks across time windows, coarse CV fallback, web-to-app attribution |

SKAN 4.0's three postbacks cover conversion windows of 0-2 days, 3-7 days, and 8-35 days. Postbacks 2 and 3 only ever carry coarse CVs. Every release gives as much data as possible within the privacy-threshold framework; the framework itself has never loosened.

### AdAttributionKit (AAK)

AAK is SKAN's successor (first introduced 2024; major updates shipping with iOS 18.4). Five key additions:

1. **Configurable attribution windows**: customizable per ad network and campaign type
2. **Re-engagement windows**: multiple independent, overlapping re-engagement conversion windows tracked via Conversion Tags (SKAN had zero re-engagement support)
3. **Country codes in postbacks**: optional geo signal that does not consume CV bits, but still subject to privacy thresholds
4. **Attribution cooldown**: prevents a fresh impression from instantly overriding a recent genuine attribution
5. **Improved testing**: Developer Mode supports faster, non-randomized attribution testing

AAK's improvements are all about flexibility and coverage (windows, re-engagement, geo, cooldown), but the fundamental paradigm is unchanged: aggregated, threshold-gated, device-mediated reporting with no user-level identifiers.

### Attribution vs Incrementality

These two terms are often conflated but measure different things.

**Attribution** asks: "which ad gets credit for this conversion?" This is what SKAN primarily constrains. It assigns credit to a campaign but cannot say whether the conversion would have happened without the ad.

**Incrementality** asks: "would the conversion have happened without the ad?" This requires a holdout control group (an RCT) and measures the causal effect of the ad, not just correlation with exposure.

Attribution degradation propagates into incrementality estimation: as label data becomes aggregated or censored, the signal available for training uplift models and evaluating their ranking quality degrades. But the two concepts are distinct: incrementality is measurable even when attribution is unavailable, as long as a holdout experiment is run.

### Apple Ads Attribution Is Independent of SKAN

Apple Ads uses its own deterministic attribution via a token taken from the device. ATT has no authority over it. The iOS ad ecosystem for third-party channels (Meta, TikTok, etc.) lives under SKAN/AAK constraints, but Apple Ads advertisers are not bound by SKAN. That said, Apple Ads advertisers still need incrementality measurement: attribution tells you which ad was clicked before a conversion, not whether the ad caused the conversion.

---

## Missing Data Mechanisms: MCAR, MAR, MNAR

When data is missing, the critical question is *why* it is missing. The answer determines whether estimates are biased, and whether collecting more data fixes the problem.

| | Why missing | Example | Effect | Cure |
|---|---|---|---|---|
| **MCAR** (Missing Completely At Random) | Pure chance, unrelated to any variable | A server randomly drops 5% of records | Smaller sample, no bias | More data |
| **MAR** (Missing At Random) | Depends on observed variables, not on the missing value itself | Older users less likely to report income, but age is observed | No inherent bias if you condition on the observed variable | Adjust for the observed variable |
| **MNAR** (Missing Not At Random) | Depends on the missing value itself | Users with very low income skip the income field because income is low | Survivors differ from dropped cases in the exact dimension being measured; bias is irreducible | More data does not help; must model the censoring mechanism |

The key diagnostic: ask whether the reason something is missing is correlated with its true value. If yes, it is MNAR.

### SKAN threshold censoring as MNAR

SKAN drops any cohort whose visit count falls below a privacy threshold t. The reason the cohort is missing is its visit count: the quantity being estimated.

```
low visit count → dropped (missing)
high visit count → survives

Survivors have inflated visit rates by construction.
```

This creates systematic downward bias in the ATE estimate, not just wider CIs. The control arm (lower base visit rate) loses more cohorts than the treatment arm at any threshold, inflating the estimated control rate faster than the treatment rate, and compressing the measured gap.

More data at the same cohort structure does not fix it: small-volume cohorts are always dropped at the same threshold. The bias is baked into the measurement design.

### Why MNAR is worse than noise

With MCAR (pure noise), enough data converges to the truth. With MNAR, the estimate converges to the wrong number, and converges confidently with narrow CIs around a biased value. A narrow CI around a biased estimate is more dangerous than a wide CI around an unbiased one.

### Broader applicability

While this notebook simulates Apple's ATT/SKAN specifically, the same structural shift (user-level tracking replaced by aggregated, threshold-gated, on-device measurement) is occurring platform-wide: Google's Privacy Sandbox and GDPR-driven consent regimes follow the same paradigm. The degradation patterns quantified here are properties of the aggregation-plus-threshold design itself, not of Apple's implementation.

---

## References

- scikit-uplift official docs and tutorial: https://www.uplift-modeling.com/en/latest/
- Two models approaches: https://www.uplift-modeling.com/en/latest/user_guide/models/two_models.html
- A Large Scale Benchmark for Uplift Modeling (Diemert et al., AdKDD @ KDD 2018): https://arxiv.org/pdf/2111.10106
- Uber CausalML GitHub: https://github.com/uber/causalml
- Why Uplift Cannot Be Directly Validated (Matteocourthoud): https://matteocourthoud.github.io/post/evaluate_uplift/
- Booking.com Causal Inference Paper: https://arxiv.org/abs/2308.09066
- Dataset on HF: https://huggingface.co/datasets/criteo/criteo-uplift
