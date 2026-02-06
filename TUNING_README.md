# CGCNN Hyperparameter Tuning Guide

This document explains how to run the CGCNN hyperparameter tuning pipeline, understand the folder structure, and interpret the results.

## Overview

The tuning pipeline (`scripts/tune_cif_nh2_v3.py`) performs a **randomized grid search** over the hyperparameter space. It trains multiple models with different hyperparameter configurations, validates each one, and reports comprehensive metrics.

**Key Facts:**
- **Dataset**: Crystal structure files (CIF format) + target property (regression)
- **Training approach**: 12 epochs per trial (early stopping ready)
- **Validation strategy**: 3-fold random split (train/validation/test)
- **Total trials**: 12 configurations × 3 seeds = 36 trials + 3 final candidates = 39 total
- **Metrics tracked**: Validation MAE, test MAE, prediction errors, outlier statistics

---

## Hyperparameter Search Space

The tuning searches over these hyperparameters:

| Parameter | Values | Count | Description |
|-----------|--------|-------|-------------|
| `optim_name` | Adam | 1 | Optimizer (Adam only) |
| `lr` | 0.03, 0.01, 0.005, 0.003, 0.001 | 5 | Learning rate (Adam) |
| `batch_size` | 32, 64, 128 | 3 | Training batch size |
| `n_conv` | 2, 3, 4, 5 | 4 | Number of convolution layers |
| `atom_fea_len` | 64, 96, 128 | 3 | Atomic feature length (dimension) |
| `h_fea_len` | 64, 128, 256, 384 | 4 | Hidden feature length (dimension) |
| `weight_decay` | 0.0, 1e-6, 1e-5, 1e-4 | 4 | L2 regularization strength |
| `n_h` | 2, 3 | 2 | Number of graph attention heads |

**Total possible combinations**: 5 × 3 × 4 × 3 × 4 × 4 × 2 = 4,320
**Actually tested**: 12 randomly sampled configurations × 3 seeds

### Parameter Meanings

- **`lr` (Learning Rate)**: Controls how much the model updates weights per step. Higher values (0.03) train faster but may diverge; lower values (0.001) train slower but more stably.

- **`batch_size`**: How many structures are processed before updating weights. Larger batches (128) are faster but less frequent updates; smaller batches (32) update more often but slower.

- **`n_conv`**: Depth of the graph neural network. More layers (5) capture longer-range interactions but risk overfitting; fewer layers (2) are simpler but may underfit.

- **`atom_fea_len`**: Dimension of atomic features. Larger (128) captures more information; smaller (64) is simpler and faster.

- **`h_fea_len`**: Dimension of hidden features in the network. Same trade-off as `atom_fea_len`.

- **`weight_decay`**: L2 regularization. Higher values (1e-4) penalize large weights more (prevents overfitting); 0.0 means no regularization.

- **`n_h`**: Number of attention heads in graph convolution. More heads (3) can learn different interaction types; fewer (2) is simpler.

---

## Folder Structure

```
runs/cif_nh2_tuning_full/
├── STATUS.txt                      # Overall tuning status (RUNNING/SUCCESS/FAILED)
├── ERROR.txt                       # Error message if tuning failed
├── sampled_hyperparameters.csv     # List of 12 sampled configurations
├── search_space.csv                # All possible hyperparameter values
├── final_candidates.csv            # Top 2 final candidates (ranked by test MAE)
│
├── cfg_01/                         # Configuration 1
│   ├── hyperparameters.csv         # This config's hyperparameters
│   ├── seed_2026/                  # Trial with seed 2026
│   │   ├── run_result.csv          # Summary metrics (1 row)
│   │   ├── test_results.csv        # Individual predictions (132 rows)
│   │   ├── model_best.pth.tar      # Best saved model
│   │   ├── checkpoint.pth.tar      # Latest checkpoint
│   │   ├── train_examples.csv      # IDs of train set samples
│   │   ├── validation_examples.csv # IDs of validation set samples
│   │   └── test_examples.csv       # IDs of test set samples
│   ├── seed_2027/                  # Trial with seed 2027 (different random split)
│   └── seed_2028/                  # Trial with seed 2028
│
├── cfg_02/ ... cfg_12/             # 11 more configurations (same structure)
│
├── final_best/                     # Best overall trial (lowest test MAE)
│   ├── seed_4041/
│   └── run_result.csv
│
├── final_cfg_1/                    # Top candidate configuration re-trained with more data
│   └── seed_4041/
│
└── final_cfg_2/                    # Second-best candidate configuration
    └── seed_4042/
```

---

## Interpreting Results

### 1. Quick Overview: STATUS.txt

```bash
cat runs/cif_nh2_tuning_full/STATUS.txt
```

Output:
```
state=FAILED
time_utc=2026-02-06T15:23:29.441803+00:00
note=dict contains fields not in fieldnames: 'config_rank_source', 'error'
```

- `state=RUNNING` → Tuning in progress
- `state=SUCCESS` → All trials completed and aggregation succeeded
- `state=FAILED` → Error during aggregation (but individual trials are saved)
- The ERROR.txt file contains the full traceback if it failed

### 2. Configuration Sampling: sampled_hyperparameters.csv

```bash
head -5 runs/cif_nh2_tuning_full/sampled_hyperparameters.csv
```

Example output:
```
config_id,optim_name,lr,batch_size,n_conv,atom_fea_len,h_fea_len,weight_decay,n_h
cfg_01,Adam,0.001,64,4,96,256,0.0001,2
cfg_02,Adam,0.03,128,3,96,256,0.0,2
cfg_03,Adam,0.03,32,4,64,128,1e-05,2
cfg_04,Adam,0.01,128,5,96,128,1e-05,3
```

Each row is one of the 12 randomly sampled configurations. These hyperparameters will be trained 3 times with different random seeds.

### 3. Per-Trial Results: run_result.csv

```bash
cat runs/cif_nh2_tuning_full/cfg_01/seed_2026/run_result.csv
```

Example output:
```
status,seed,epochs,best_val_mae,best_model,test_mae,outlier_abs_gt_10_count,max_abs_diff,p95_abs_diff,optim_name,lr,batch_size,n_conv,atom_fea_len,h_fea_len,weight_decay,n_h,error

ok,2026,12,7.245489120483398,/path/to/model_best.pth.tar,5.7979430908387,18,53.87413215637207,26.600625038146973,Adam,0.001,64,4,96,256,0.0001,2,
```

**Columns explained:**

| Column | Meaning | Example |
|--------|---------|---------|
| `status` | Success/failure of trial | `ok`, `failed` |
| `seed` | Random seed used | 2026 |
| `epochs` | Epochs trained | 12 |
| `best_val_mae` | Best validation MAE during training | 7.25 |
| `test_mae` | **Final test set MAE** (lower is better) | 5.80 |
| `outlier_abs_gt_10_count` | # of predictions with error > 10 | 18 |
| `max_abs_diff` | Largest prediction error | 53.87 |
| `p95_abs_diff` | 95th percentile of errors | 26.60 |
| `error` | Error message if failed | (empty if ok) |
| Remaining columns | Hyperparameters used | Adam, lr=0.001, ... |

**Key metrics to compare trials:**
1. **`test_mae`** — Primary metric. Lower is better. This is the MAE on unseen test structures.
2. **`best_val_mae`** — Validation MAE during training. Should be similar to test MAE (if similar → good generalization).
3. **`outlier_abs_gt_10_count`** — Number of "bad" predictions. Fewer is better.
4. **`max_abs_diff`** — Worst single prediction. Can indicate if model is unstable.

### 4. Detailed Predictions: test_results.csv

```bash
head -10 runs/cif_nh2_tuning_full/cfg_01/seed_2026/test_results.csv
```

Output:
```
id,target,prediction,diff,mae
pos_1157,3.1059999466,2.6373505592,0.4686493874,0.4686493874
pos_799,1.6890000105,2.5829281807,-0.8939281702,0.8939281702
pos_48,6.0799999237,4.1149458885,1.9650540352,1.9650540352
pos_1132,3.6989998817,2.9840912819,0.7149085999,0.7149085999
```

**Columns:**
- `id` — Structure ID (from your CIF dataset)
- `target` — True property value
- `prediction` — Model's predicted value
- `diff` — prediction - target (can be positive or negative)
- `mae` — |prediction - target| (always positive, used to compute test_mae)

**Use this to:**
- Find structures with large errors (high `mae` values)
- Analyze which materials the model struggles with
- Create error histograms or scatter plots (prediction vs target)

### 5. Data Split Information

```bash
cat runs/cif_nh2_tuning_full/cfg_01/seed_2026/train_examples.csv | head
cat runs/cif_nh2_tuning_full/cfg_01/seed_2026/validation_examples.csv | head
cat runs/cif_nh2_tuning_full/cfg_01/seed_2026/test_examples.csv | head
```

Each file contains a single column with structure IDs used in that set. Example:
```
pos_1
pos_100
pos_1000
```

This shows exactly which 1314 structures were in each set:
- **Train set**: Used to optimize weights
- **Validation set**: Used to check for overfitting during training
- **Test set**: Never seen during training, used to evaluate final performance

---

## How to Compare Results Across All Trials

### Find the best configuration

```bash
# Find trial with lowest test MAE
find runs/cif_nh2_tuning_full -name "run_result.csv" -exec sh -c '
  head -2 "$1" | tail -1 | awk -F, "{print \$6, \$1}"
' _ {} \; | sort -n | head -5
```

This shows the 5 trials with lowest test MAE.

### Get summary statistics for each config

```bash
for cfg in runs/cif_nh2_tuning_full/cfg_*; do
  echo "=== $(basename $cfg) ==="
  grep "^ok," $cfg/*/run_result.csv | awk -F, '{print $6}' | \
    awk '{if(NR==1){sum=$1; min=$1; max=$1} else {sum+=$1; if($1<min) min=$1; if($1>max) max=$1}}
         END {print "Test MAE: min="min", avg="sum/NR", max="max}'
done
```

This shows min/average/max test MAE for each configuration across its 3 seeds.

### Analyze variance across seeds

If test MAE differs greatly across seeds for the same config, the model is **sensitive to initialization**:
- Large variance → unstable training
- Small variance → robust hyperparameters

---

## How to Run Training

### Prerequisites

```bash
# Install dependencies
uv sync

# Ensure you have CIF structure files and an id_prop.csv
# Example: /Users/qz/Downloads/cif/
ls /Users/qz/Downloads/cif/id_prop.csv
```

### Run Full Tuning

```bash
# Navigate to repo root
cd /Users/qz/.ghq/github.com/singularitti/cgcnn

# Run tuning
uv run python scripts/tune_cif_nh2_v3.py tune \
  /Users/qz/Downloads/cif \
  runs/cif_nh2_tuning_full

# Monitor progress
tail -f runs/cif_nh2_tuning_full/*/seed_*/run_result.csv
```

This will:
1. **Sample 12 random hyperparameter configurations** from the search space
2. **Train 3 trials per configuration** with different random seeds (36 trials total)
3. **Evaluate each trial** on a held-out test set
4. **Save results** in the structured folder hierarchy
5. **Rank configurations** by test MAE and train final candidates

**Typical runtime:** ~1-2 hours for 36 trials (depends on hardware)

### Run Single Trial

To train a specific configuration manually:

```bash
uv run python scripts/tune_cif_nh2_v3.py run \
  /Users/qz/Downloads/cif \
  runs/my_single_trial \
  --optim_name Adam \
  --lr 0.001 \
  --batch_size 64 \
  --n_conv 4 \
  --atom_fea_len 96 \
  --h_fea_len 256 \
  --weight_decay 0.0001 \
  --n_h 2 \
  --epochs 30
```

This trains a single model with the specified hyperparameters and saves results to `runs/my_single_trial/`.

### Test a Trained Model

```bash
# Load model from any trial
python -c "
import torch
model = torch.load('runs/cif_nh2_tuning_full/cfg_01/seed_2026/model_best.pth.tar')
print(model)
"

# Use it in your own code
from cgcnn.model import CrystalGraphConvNet
model = CrystalGraphConvNet(
    orig_atom_fea_len=92,  # from atom_init.json
    nbr_fea_len=41,        # from atom_init.json
    atom_fea_len=96,
    n_h=2,
    n_conv=4,
    h_fea_len=256,
)
state = torch.load('runs/cif_nh2_tuning_full/cfg_01/seed_2026/model_best.pth.tar')
model.load_state_dict(state['model_state_dict'])
model.eval()
# Use model for predictions
```

---

## Tips for Interpreting Results

### 1. Check for Overfitting

Compare `best_val_mae` vs `test_mae` for each trial:

```bash
grep "^ok," runs/cif_nh2_tuning_full/cfg_01/*/run_result.csv | \
  awk -F, '{print $4 "\t" $6}' | \
  awk '{printf "Val MAE: %.2f, Test MAE: %.2f, Diff: %.2f\n", $1, $2, $2-$1}'
```

- If `test_mae >> val_mae` → Overfitting. Try higher `weight_decay` or lower `n_conv`.
- If `test_mae ≈ val_mae` → Good generalization.
- If `test_mae < val_mae` → Normal variance; test set is slightly easier.

### 2. Identify Failure Modes

Check which configurations struggled:

```bash
find runs/cif_nh2_tuning_full -name "run_result.csv" -exec sh -c '
  dir=$(dirname "$1")
  tail -1 "$1" | awk -F, "{if (\$6 > 20) print dir, \"high_mae:\" \$6}"
' _ {} \;
```

Find trials with test MAE > 20 to identify bad hyperparameter combinations.

### 3. Visualize Predictions

Create a scatter plot (prediction vs target) for the best trial:

```python
import pandas as pd
import matplotlib.pyplot as plt

results = pd.read_csv('runs/cif_nh2_tuning_full/cfg_01/seed_2026/test_results.csv')

plt.figure(figsize=(10, 6))
plt.scatter(results['target'], results['prediction'], alpha=0.6)
plt.plot([results['target'].min(), results['target'].max()],
         [results['target'].min(), results['target'].max()],
         'r--', label='Perfect prediction')
plt.xlabel('True Property Value')
plt.ylabel('Predicted Property Value')
plt.legend()
plt.title('Model Predictions vs Reality')
plt.grid(True, alpha=0.3)
plt.savefig('predictions.png')
plt.show()
```

### 4. Rank Configurations

To get a summary of all configurations:

```bash
echo "Config,Avg_Test_MAE,Min,Max,Seed_Variance"
for cfg in runs/cif_nh2_tuning_full/cfg_*; do
  grep "^ok," $cfg/*/run_result.csv | awk -F, '{vals[NR]=$6} END {
    avg=0; for(i=1;i<=NR;i++) avg+=vals[i]; avg/=NR;
    asort(vals);
    print FILENAME, avg, vals[1], vals[NR], vals[NR]-vals[1]
  }' | awk -v cfg=$(basename $cfg) '{printf "%s,%.2f,%.2f,%.2f,%.2f\n", cfg, $2, $3, $4, $5}'
done | sort -t, -k2 -n | head -10
```

This shows the best 10 configurations by average test MAE.

---

## Common Issues & Troubleshooting

### Issue: STATUS.txt says FAILED

The tuning likely completed most trials but failed during final aggregation (CSV fieldnames mismatch). Your individual trial results are still saved and valid. You can still analyze them using the commands above.

### Issue: Some trials say `status=failed`

Check the error field in run_result.csv. If a trial encountered NaN in validation loss, our fix (replacing `sys.exit(1)` with `RuntimeError`) ensures it writes the error message instead of crashing.

### Issue: All test MAE values are very high (>20)

- The dataset may be very noisy
- The hyperparameters may not be suitable
- Try increasing `epochs` or adjusting learning rate

### Issue: Validation MAE much worse than test MAE

This suggests the validation set is harder than the test set (random chance). Re-run with different seeds to verify.

---

## Next Steps

1. **Identify the best configuration** by looking at lowest test MAE
2. **Check generalization** by comparing val_mae vs test_mae
3. **Analyze failure cases** using test_results.csv to find problematic structures
4. **Retrain the best config** with more epochs for production use
5. **Use the saved model** (`model_best.pth.tar`) for predictions on new structures

---

## Questions?

- **Hyperparameters**: See "Hyperparameter Search Space" section above
- **File formats**: See "Folder Structure" section
- **Model architecture**: See `src/cgcnn/model.py`
- **Training details**: See `src/cgcnn/training.py`
