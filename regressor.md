## Plan: threshold-split classifier + log-transformed regressors

TL;DR: Use the existing two-stage workflow as a base, but split the dataset at M = 1e-3. Train one classifier for small vs large, then train two separate regressors: one on the small-value region and one on the large-value region, both using log transform on positive targets before training and inverse-transforming predictions for parity benchmarking.

**Steps**
1. Reuse existing tooling in `tools/train_magnetization_two_stage.py` and `tools/train_magnetization_positive_only.py`.
2. Implement threshold-based dataset preparation:
   - classifier dataset: all records, label = 0 if M <= 1e-3 else 1.
   - small regressor dataset: records with 0 < M <= 1e-3.
   - large regressor dataset: records with M > 1e-3.
   - exclude exact zeros from both log-transformed regressors, since log(0) is invalid.
3. Add or extend target-transform support for `log` (and optionally `log10`) in dataset writing and inference.
4. Train the classifier on the full dataset using `task="classification"`.
5. Train one regressor on the small-value subset and one on the large-value subset using `task="regression"` with targets transformed by log before training.
6. Run inference separately for the classifier and both regressors.
7. Merge predictions with a hard gate:
   - if classifier predicts small => use small-regressor output
   - otherwise => use large-regressor output
8. Invert the log transform on both regressor outputs before computing metrics.
9. Produce benchmarks and parity plots on raw-scale combined predictions, plus separate subset metrics for small vs large.

**Relevant files**
- `tools/train_magnetization_two_stage.py` — existing two-stage pipeline and merge logic
- `tools/train_magnetization_positive_only.py` — existing target-transform and log-inversion workflow
- `src/cgcnn/training.py` — train_model entry point and explicit split support
- `src/cgcnn/inference.py` — predict_model and checkpoint inference
- `src/cgcnn/data.py` — dataset loading and explicit `include_ids` support

**Verification**
1. Run the new threshold-split pipeline and verify it creates classifier, small regressor, and large regressor checkpoints.
2. Confirm classifier metrics and merge behavior on test set.
3. Confirm parity plot and raw-scale metrics after inverse-transforming log predictions.
4. Compare small/large subset metrics against the original single-model baseline.

**Decisions**
- Use classifier threshold at M = 1e-3.
- Exclude exact zeros from log-transformed regression.
- Use hard gating on classifier output to choose between small and large regressors.

**Further considerations**
1. If exact zeros are abundant, it may be better to treat them as a separate class or remove them from the log-transform workflow entirely.
2. A log-transform on both subsets is valid only if every target in the subset is strictly positive.
3. The classifier itself must be validated carefully, because misclassifying a sample into the wrong regressor can harm final parity.
