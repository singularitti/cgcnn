# Training target distribution analysis

Created: `2026-06-25`

## Purpose

Plot and summarize the target distribution for the CGCNN `energy_above_hull`
training run in `~/run/training`. This checks whether the raw target
distribution is highly skewed and whether the train/validation/test split
changed that distribution.

## Inputs

- Prepared target table: `~/run/training/id_prop.csv`
- Split IDs:
  - `~/run/training/splits/train_ids.csv`
  - `~/run/training/splits/val_ids.csv`
  - `~/run/training/splits/test_ids.csv`
- Plotting code: `~/cgcnn/tools/plot_training_target_distribution.py`
- Command:

```bash
python tools/plot_training_target_distribution.py \
  --id-prop ~/run/training/id_prop.csv \
  --split-dir ~/run/training/splits \
  --output-dir ~/cgcnn/analysis/training_distribution
```

The source files under `~/run/training` were read in place and were
not modified by this analysis.

## Outputs

- `energy_above_hull_distribution_by_split.png`: histogram, log-tail,
  cumulative distribution, and split quantile plot.
- `energy_above_hull_distribution_summary.csv`: counts, quantiles, threshold
  fractions, and zero fractions for all samples and each split.
- `energy_above_hull_values_by_split.csv`: material IDs with target values and
  assigned split labels used to create the plot.
