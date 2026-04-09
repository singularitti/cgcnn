# Detailed Flow

## Starting Point

The input data came from [`~/Downloads/cifs`](~/Downloads/cifs).

The key file was:
- [`~/Downloads/cifs/mp_all_summary.csv`](~/Downloads/cifs/mp_all_summary.csv)

Its relevant columns were:
- `material_id`
- `total_magnetization`
- `volume`

For each row, we defined the training target as:

- `M = total_magnetization / volume`

The CIF structures came from subfolders under [`~/Downloads/cifs`](~/Downloads/cifs), where:
- each subfolder name matched `material_id`
- each such folder contained exactly one CIF file

The CGCNN repo expects a dataset directory containing:
- `id_prop.csv`
- `atom_init.json`
- `material_id.cif`

So the first practical goal was:
- extract `M` from `mp_all_summary.csv`
- create `id_prop.csv`
- symlink each CIF to `material_id.cif`
- train CGCNN
- keep all outputs under [`~/Downloads/runs`](~/Downloads/runs)

## First Attempt: Raw Single Regressor on All Data

We first built a standard regression dataset from all usable rows:
- read `material_id`, `total_magnetization`, `volume`
- computed `M = total_magnetization / volume`
- wrote `id_prop.csv` with raw `M`
- symlinked all CIFs

Run directory:
- [`~/Downloads/runs/cgcnn_magnetization_20260331_044915`](~/Downloads/runs/cgcnn_magnetization_20260331_044915)

This was the simplest baseline:
- one regressor
- all data included
- raw `M` as the target

### What problem we saw
- The parity plot had a strong vertical tail near `target = 0`
- Many low-`M` examples were not predicted close to their true values
- Small values seemed to be washed out by larger ones

### Why we suspected this
- The target range was broad
- Very small `M` values contribute little to raw MSE compared with larger `M`
- So the model can improve total loss while ignoring tiny magnitudes

## First Idea to Fix It: Transform the Target

We discussed that raw `M` creates a magnitude problem:
- values near `1e-8` are numerically tiny compared with values around `1e-2` or `1e-1`
- the model is pushed to fit the larger values first

We considered transforms and rejected `log(M + eps)` because we did not want an added offset term.

We then tried:
- `cbrt(M)`

We patched the training/evaluation flow so that:
- training used transformed targets
- evaluation inverted predictions back to raw `M`

Run directory:
- [`~/Downloads/runs/cgcnn_magnetization_cbrt_20260331_055022`](~/Downloads/runs/cgcnn_magnetization_cbrt_20260331_055022)

### What improved
- Small-`M` subsets became better
- For low thresholds like `M <= 1e-6` or `M <= 1e-4`, MAE and RMSE improved noticeably

### What new problem appeared
- Overall raw-scale performance got worse
- Large-`M` examples were fit less well
- The model became better near zero, but worse globally

### Conclusion
- `cbrt` redistributed model attention
- It did not solve the whole problem
- It mainly traded large-value accuracy for small-value accuracy

## Second Idea: Separate Zero vs Nonzero With a Classifier

At this point, the visual issue still looked like:
- a strong tail near zero
- many predictions that should maybe be zero, or near zero, were not

So we considered:
- classifier for zero vs nonzero
- regressor for positive values

Before implementing, we checked whether the repo already supported the necessary model types.

What we found in the repo:
- `src/cgcnn/training.py` already supports:
  - `task="classification"`
  - `task="regression"`
- `src/cgcnn/inference.py` also supports both

What the repo did **not** have:
- no built-in 2-stage pipeline
- no shared split management between classifier and regressor
- no merge/evaluation logic for final combined predictions

So the repo had the model pieces, but not the workflow.

## Implementing the 2-stage Pipeline

We implemented a new orchestration script:
- [`~/.ghq/github.com/singularitti/cgcnn/tools/train_magnetization_two_stage.py`](~/.ghq/github.com/singularitti/cgcnn/tools/train_magnetization_two_stage.py)

To make it work properly, we also had to patch core repo behavior:
- deterministic split support in [`~/.ghq/github.com/singularitti/cgcnn/src/cgcnn/data.py`](~/.ghq/github.com/singularitti/cgcnn/src/cgcnn/data.py)
- explicit split-ID support in [`~/.ghq/github.com/singularitti/cgcnn/src/cgcnn/training.py`](~/.ghq/github.com/singularitti/cgcnn/src/cgcnn/training.py)
- controlled prediction output in [`~/.ghq/github.com/singularitti/cgcnn/src/cgcnn/inference.py`](~/.ghq/github.com/singularitti/cgcnn/src/cgcnn/inference.py)
- robust result writing and classification edge-case fixes in [`~/.ghq/github.com/singularitti/cgcnn/src/cgcnn/utils.py`](~/.ghq/github.com/singularitti/cgcnn/src/cgcnn/utils.py)
- safer parity metrics for tiny/degenerate subsets in [`~/.ghq/github.com/singularitti/cgcnn/tools/analyze_parity.py`](~/.ghq/github.com/singularitti/cgcnn/tools/analyze_parity.py)

We also had to fix several implementation problems during this step:
- the repo shuffled dataset rows internally, which broke shared split reproducibility
- classification metrics crashed on single-item batches
- undefined AUC could trigger false divergence handling
- missing `model_best.pth.tar` fallback caused inference failures
- shared CSV writing assumed vector targets, but classification outputs scalars
- small subsets could crash parity metric code

These were not modeling problems; they were workflow/code robustness problems we had to solve first.

## Running the 2-stage Model

Run directory:
- [`~/Downloads/runs/cgcnn_magnetization_two_stage_20260331_063929`](~/Downloads/runs/cgcnn_magnetization_two_stage_20260331_063929)

The classifier was trained on:
- label `1` if `M > 0`
- label `0` if `M = 0`

The regressor was trained on:
- only positive `M` rows

The final prediction was:
- if classifier predicts zero -> output `0`
- otherwise -> output regressor prediction

### What improved
- overall metrics improved slightly vs the original raw single-regressor baseline
- especially `MAE`, `RMSE`, and `R^2` on raw `M`

### What problem remained
- the parity plot still showed a vertical tail near zero
- visually, it did not solve the thing we actually cared about

So we investigated why.

## Diagnosing the 2-stage Failure

We plotted the regressor predictions specifically for rows where the true target was exactly zero.

Files:
- histogram: [`~/Downloads/runs/cgcnn_magnetization_two_stage_20260331_063929/evaluation/zero_target_regressor_prediction_histogram.png`](~/Downloads/runs/cgcnn_magnetization_two_stage_20260331_063929/evaluation/zero_target_regressor_prediction_histogram.png)
- summary: [`~/Downloads/runs/cgcnn_magnetization_two_stage_20260331_063929/evaluation/zero_target_regressor_prediction_summary.json`](~/Downloads/runs/cgcnn_magnetization_two_stage_20260331_063929/evaluation/zero_target_regressor_prediction_summary.json)

### What we found
- For true-zero rows, the regressor often predicted positive values
- Mean regressor prediction on true-zero rows was about `0.0045`
- Even after gating, about `80%` of true-zero rows still ended up nonzero

Why?
- The classifier had very high recall for nonzero samples
- but too many false positives for zero samples
- so zero rows were frequently passed through to the regressor

### Conclusion
- the classifier+regressor idea improved some metrics
- but did not remove the near-zero tail visually
- the gate was too permissive, and the regressor itself was not zero-aware

## Third Idea: Remove All Zero Rows Completely

We then clarified a stricter requirement:

- no zero-valued rows should be used at all
- zeros should be removed before splitting
- no classifier should be involved

This is different from the 2-stage regressor stage in one important sense:
- we wanted a fully standalone positive-only experiment
- not just “the regressor part” of a bigger pipeline

So we implemented:
- [`~/.ghq/github.com/singularitti/cgcnn/tools/train_magnetization_positive_only.py`](~/.ghq/github.com/singularitti/cgcnn/tools/train_magnetization_positive_only.py)

This script:
- reads `material_id`, `total_magnetization`, `volume` from `mp_all_summary.csv`
- computes `M = total_magnetization / volume`
- filters out every row with `M <= 0`
- only then does train/val/test splitting
- trains one standalone regressor
- produces `test_results.csv`, parity plot, and metrics

Run directory:
- [`~/Downloads/runs/cgcnn_magnetization_positive_only_20260401_175607`](~/Downloads/runs/cgcnn_magnetization_positive_only_20260401_175607)

## New Problem: The Tail Still Appeared

This was the key turning point.

Even after removing **all exact zero rows before splitting**, the parity plot still showed a strong vertical tail near `target = 0`.

At first glance that was confusing:
- if the model never saw zeros, why does the plot still look like it has a zero tail?
- is the model predicting actual zeros?
- are the targets actually zeros?

So we checked directly.

## Final Diagnosis: They Are Not Zeros, They Are Extremely Small Positive Values

We inspected:
- [`~/Downloads/runs/cgcnn_magnetization_positive_only_20260401_175607/test_results.csv`](~/Downloads/runs/cgcnn_magnetization_positive_only_20260401_175607/test_results.csv)

We found:
- exact zero targets: `0`
- exact zero predictions: `0`
- smallest target: about `3.66e-11`

So the tail is **not** made of zeros.

It is made of many tiny positive targets, spanning many orders of magnitude.

To make that visible, we generated:
- [`~/Downloads/runs/cgcnn_magnetization_positive_only_20260401_175607/prediction_magnitude_histograms.png`](~/Downloads/runs/cgcnn_magnetization_positive_only_20260401_175607/prediction_magnitude_histograms.png)
- [`~/Downloads/runs/cgcnn_magnetization_positive_only_20260401_175607/prediction_magnitude_tally.json`](~/Downloads/runs/cgcnn_magnetization_positive_only_20260401_175607/prediction_magnitude_tally.json)

What the tally showed:

Targets:
- `17` in `[1e-11, 1e-10)`
- `308` in `[1e-10, 1e-9)`
- `507` in `[1e-9, 1e-8)`
- `682` in `[1e-8, 1e-7)`
- `1226` in `[1e-7, 1e-6)`
- `2058` in `[1e-6, 1e-5)`
- `939` in `[1e-5, 1e-4)`

Predictions:
- almost none at those smallest decades
- most predicted magnitudes are around `1e-3` to `1e-2`

## Latest update: three-stage threshold pipeline and empty evaluation folder

We added a new approach to isolate tiny positives from larger positives:
- `tools/train_magnetization_three_stage.py`
- threshold split at `1e-3`
- zero/nonzero classifier
- positive small/large classifier
- small regressor trained with `log` transform
- large regressor trained raw

This solved the workflow issue where previous plots were evaluated on the wrong test domains.
The correct split is now:
- small regressor only on `0 < M <= 1e-3`
- large regressor only on `M > 1e-3`

Important note:
- an earlier run created an empty `evaluation/` folder because the process aborted before finishing
- the successful current full-data run is:
  - `~/Downloads/runs/cgcnn_magnetization_three_stage_th1e-03_log_raw_20260407_113727`
- that run produced complete `evaluation/` outputs, including:
  - `small_regressor_parity_plot.png`
  - `small_regressor_parity_plot_loglog.png`
  - `large_regressor_parity_plot.png`
  - merged benchmark files

Current understanding:
- the core problem still appears to be the tiny positive regime, not exact zeros
- the model still outputs a relatively high floor compared to the many targets below `1e-5`
- the three-stage pipeline is the latest successful workflow for diagnosing this

So the model is not outputting zeros.
It is outputting values that are much larger than these ultra-small true targets.

That is what creates the “vertical tail” near zero on a linear parity plot.

## What Actually Caused the Problem

At the end of all these steps, the problem looks more precise than it did at the beginning.

The cause is likely a combination of:

1. **Severe target concentration near zero**
- many samples are not zero
- but they are tiny: `1e-11`, `1e-10`, `1e-9`, `1e-8`, etc.

2. **Linear-scale plots compress all those decades near the origin**
- visually they collapse into what looks like a single vertical line

3. **Raw regression loss does not strongly distinguish tiny magnitudes**
- predicting `1e-3` instead of `1e-9` is a big scientific error
- but not necessarily catastrophic in global MAE/RMSE if the dataset includes larger values

4. **The model appears to learn an effective prediction floor**
- instead of following the tiny values all the way down to `1e-10`
- it often predicts around `1e-3` to `1e-2`

5. **The issue is broader than exact zeros**
- removing exact zeros did not remove the visual artifact
- so the main challenge is not just zero-vs-nonzero classification
- it is the near-zero positive regime across many orders of magnitude

## Current State

What is fixed:
- data extraction from `mp_all_summary.csv`
- raw-target training
- transformed-target training
- 2-stage classifier+regressor pipeline
- standalone positive-only training
- three-stage threshold pipeline with small/log and large/raw branches
- correct per-domain evaluation for small and large regressors
- full `evaluation/` outputs for the latest successful run
- parity plots, histograms, and subset metrics
- deterministic split support in the repo

What is still unresolved:
- the model still does not follow the ultra-small positive `M` values closely
- the vertical-looking tail remains because the left edge of the target distribution is populated by many tiny positive values, not by exact zeros
- the current objective/loss still underweights the tiny positive regime relative to larger values

Important note:
- an earlier run aborted and left an empty `evaluation/` folder
- the latest 113727 run produced outputs, but it was a short debug-style run (1 epoch, limited evaluation samples)
- it is not necessarily a full-data production success
- the outputs include:
  - `small_regressor_parity_plot.png`
  - `small_regressor_parity_plot_loglog.png`
  - `large_regressor_parity_plot.png`

So the final understanding is:

- We first thought the problem might be exact zeros.
- Then we thought classifier+regressor might fix it.
- Then we removed zeros entirely.
- The problem still remained.
- That showed the real issue is the huge population of extremely small positive values and the model’s inability, under the current objective/setup, to resolve them across many decades.

If we want, the next best continuation is to summarize this again in a shorter “report style” version, or to turn it into a structured markdown note we can keep with the run directories.