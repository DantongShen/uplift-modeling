# Uplift Modeling with Criteo Dataset

Estimating individual-level incremental treatment effects for ad targeting using causal ML meta-learners.

## Motivation

In privacy-first ad environments (e.g. Apple's ATT framework), measuring the true **incremental effect** of an ad, not just who converts but who converts *because* of the ad, is critical for budget efficiency. This project applies uplift modeling to tackle exactly that problem.

## Dataset

[Criteo Uplift Modeling Dataset](https://ailab.criteo.com/criteo-uplift-prediction-dataset/) — ~14M rows, 12 anonymized features, binary treatment/visit/conversion labels.

- Target: `visit` (visit rate ~4.7%, more signal than conversion at 0.29%)
- Treatment ratio: 85% treated / 15% control

The paper releases two datasets: **CRITEO-UPLIFTv2** (real RCT data, used here) and **CRITEO-ITE** (semi-synthetic with generated response surfaces, used for PEHE benchmarking). This project uses CRITEO-UPLIFTv2.

## Methods

| Method | Description |
|--------|-------------|
| T-Learner | Train separate models on treatment and control groups, subtract predicted probabilities |
| X-Learner | Extension of T-Learner that handles treatment/control imbalance better via pseudo-treatment effects |

## Project Structure

```
uplift-modeling/
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_tlearner.ipynb
│   ├── 03_xlearner.ipynb
│   ├── 04_evaluation.ipynb
│   ├── 05_robustness_insight.ipynb
│   ├── 06_business_impact.ipynb
│   └── 07_privacy_constraints.ipynb
├── dashboard/
│   └── app.py
├── images/
├── models/
└── LEARNING_NOTES.md
```

## Results

**EDA (notebook 1)**
- Observed ATE by treatment assignment: 0.0103 (intention-to-treat)
- Observed ATE by exposure (exposed treatment vs control): 0.3763

**T-Learner (notebook 2)**
- Trained on 10% stratified sample (~1.4M rows), 80/20 train/test split
- Mean uplift score on test set: 0.0155, consistent with observed ATE of 0.0103

**X-Learner (notebook 3)**
- Implemented via `causalml` `BaseXClassifier` with LGBMClassifier (outcome) and LGBMRegressor (effect)
- Mean uplift score: 0.1170, std 0.1494; broader distribution shifted right vs T-Learner
- Score scale differs from T-Learner as Stage 3 predicts continuous pseudo-effects, not calibrated probabilities

**Evaluation (notebook 4)**

| Metric | T-Learner | X-Learner |
|---|---|---|
| Qini AUC | 0.0380 | 0.0760 |
| Incremental visits at top 20% | 984 | 1,684 (+71%) |
| Decile 1 observed lift | 0.0252 | 0.0369 |

X-Learner's Qini AUC is 2x T-Learner's. The advantage is most pronounced at small targeting fractions: at the top 20%, X-Learner captures 71% more incremental visits. Both models converge at 100% targeting (2,458 total incremental visits). The decile analysis confirms X-Learner concentrates persuadables more tightly in the top buckets.

![Qini Curve](images/qini_curve.png)

![Uplift by Decile](images/uplift_by_decile.png)

## Tech Stack

Python · LightGBM · scikit-uplift · causalml · Streamlit

## Project Phases

- [x] Phase 1: EDA (`01_eda.ipynb`)
- [x] Phase 2: T-Learner (`02_tlearner.ipynb`)
- [x] Phase 3: X-Learner (`03_xlearner.ipynb`)
- [x] Phase 4: Evaluation (`04_evaluation.ipynb`)
- [ ] Phase 5: Robustness & Insight (`05_robustness_insight.ipynb`)
- [ ] Phase 6: Business Impact (`06_business_impact.ipynb`)
- [ ] Phase 7: Privacy Constraints (`07_privacy_constraints.ipynb`)
- [ ] Phase 8: Dashboard (`dashboard/app.py`)
- [ ] Phase 9: README & Documentation

## References

- Diemert et al. (2018). [A Large Scale Benchmark for Uplift Modeling](https://arxiv.org/pdf/2111.10106)
- [scikit-uplift docs](https://www.uplift-modeling.com/en/latest/)
- [Uber CausalML](https://github.com/uber/causalml)
- [Criteo Uplift Dataset on Hugging Face](https://huggingface.co/datasets/criteo/criteo-uplift)

## Citation

```bibtex
@inproceedings{Diemert2018,
  author = {{Diemert Eustache, Betlei Artem} and Renaudin, Christophe and Massih-Reza, Amini},
  title={A Large Scale Benchmark for Uplift Modeling},
  publisher = {ACM},
  booktitle = {Proceedings of the AdKDD and TargetAd Workshop, KDD, London, United Kingdom, August, 20, 2018},
  year = {2018}
}
```
