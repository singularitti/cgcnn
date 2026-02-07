# CLAUDE.md

## Quick Reference

### Environment & Setup

This project uses `uv` for dependency management:

```bash
uv sync --no-cache --upgrade  # Install all dependencies
source .venv/bin/activate  # Create virtual environment if needed
```

### Common Commands

```bash
# Training a model
uv run python -m cgcnn.training

# Making predictions on new structures
uv run python -m cgcnn.inference

# Hyperparameter tuning (randomized grid search with 12 configs × 3 seeds)
uv run python scripts/tune.py tune <data_dir> <output_dir>

# Running a single training trial with specific hyperparameters
uv run python scripts/tune.py run <data_dir> <output_dir> --lr 0.001 --batch_size 32 --n_conv 3
```

## Project Overview

CGCNN is a graph neural network that predicts material properties from crystal structures. The workflow involves:

1. Data Input: Crystal structures in CIF format
2. Graph Representation: Atoms as nodes, atomic bonds as edges
3. Graph Convolution: Message passing through ConvLayers to learn structure-property relationships
4. Prediction: Property prediction for materials

## Architecture Overview

### Core Components (in `src/cgcnn/`)

#### `model.py`

- `ConvLayer`: Single message-passing layer that aggregates neighbor atom information
  - Takes atom features and neighbor indices as input
  - Returns updated atom features
- `CrystalGraphConvNet`: Main model that stacks multiple ConvLayers
  - `n_conv` layers with intermediate processing
  - Pooling to aggregate atom features into material property prediction
  - Output: single scalar (property value) or normalized predictions

#### `data.py`

- `CIFData`: PyTorch Dataset class for crystal structures
  - Reads CIF files and converts them to graph format
  - Returns `(atom_fea, atom_idx, neighbors, distances, properties)` tuples
  - Requires: `id_prop.csv` (structure names → target values), `atom_init.json` (element features)

#### `training.py`

- `train_model()`: Main training function
  - Handles data loading, model training, validation, and checkpointing
  - Saves best model to `model_best.pth.tar` and latest to `checkpoint.pth.tar`
  - Uses MAE (Mean Absolute Error) as loss function

#### `inference.py`

- `predict_model()`: Inference function for new structures
  - Loads pre-trained model checkpoint
  - Predicts property values for new crystal structures

#### `utils.py`

- Helper functions for data preprocessing and metric calculations

### Training Workflow

```
1. Load crystal structure data (CIF files)
2. Convert to graph representation (atoms=nodes, bonds=edges)
3. Initialize atom features from atom_init.json
4. Pass through CrystalGraphConvNet
5. Compute loss and backpropagate
6. Validate on holdout set and save checkpoints
7. Return model_best.pth.tar
```

## Data Requirements

### Directory Structure

```
<data_dir>/
├── id_prop.csv     # Two columns: structure_id, property_value
├── atom_init.json  # Element → initial features mapping
└── *.cif           # Crystal structure files
```

### File Details

- `id_prop.csv`: Plain text with columns `id` and `property`
  - Example: `CIF_ID_1, 3.14`
  - Must match CIF filenames (CIF_ID_1.cif)
- `atom_init.json`: Element features for initialization
  - Example: `{"Al": [feature_1, feature_2, ...], "O": [...], ...}`
  - Available in `data/sample-regression/atom_init.json`
- `*.cif`: Standard Crystallographic Information Format

## Hyperparameter Tuning

### Overview

Located in `scripts/tune.py`, uses randomized grid search with multiple random seeds for robustness.

### Default Configuration

- 12 configurations (random sampling from search space)
- 3 seeds per configuration (2026, 2027, 2028)
- Total trials: 36 (12 × 3)

### Search Space (hyperparameters being varied)

- `lr` (learning rate): typically 0.001-0.1
- `batch_size`: typically 32-256
- `n_conv` (number of conv layers): 1-4
- `atom_fea_len` (atom feature length): 64-256
- `h_fea_len` (hidden feature length): 64-256
- `weight_decay`: 0-0.001
- `n_h` (number of hidden layers in dense network): 1-3

### Running Tuning

```bash
# Full tuning (12 configs × 3 seeds)
uv run python scripts/tune.py tune <data_dir> <output_dir>

# Single trial
uv run python scripts/tune.py run <data_dir> <output_dir> \
  --lr 0.001 --batch_size 32 --n_conv 3 --atom_fea_len 64 \
  --h_fea_len 64 --weight_decay 0 --n_h 1
```

### Output Structure

```
<output_dir>/
├── sampled_hyperparameters.csv     # All sampled configurations
├── search_space.csv                # Defined search space
├── ranked_search_results.csv       # Results ranked by performance
├── final_selection.csv             # Best configurations selected
├── cfg_01/
│   ├── hyperparameters.csv
│   ├── seed_2026/
│   │   ├── model_best.pth.tar      # Best model for this seed
│   │   ├── checkpoint.pth.tar      # Latest checkpoint
│   │   ├── run_result.csv          # Summary metrics (MAE, outliers, etc.)
│   │   ├── test_results.csv        # Per-structure predictions
│   │   ├── train_examples.csv      # Training data predictions
│   │   └── validation_examples.csv # Validation data predictions
│   ├── seed_2027/
│   └── seed_2028/
├── cfg_02/
├── ...
├── cfg_12/
├── final_cfg_1/                    # Best configuration re-run with more seeds
├── final_cfg_2/                    # Second-best configuration re-run
├── final_best/                     # Best overall model
├── STATUS.txt                      # Current tuning status
└── ERROR.txt                       # Any errors encountered
```

### Monitoring Tuning

- STATUS.txt: Shows current progress (e.g., "Running cfg_05/seed_2027")
- ERROR.txt: Lists any failed trials with error messages
- ranked_search_results.csv: View rankings as tuning progresses

### Key Metrics (in run_result.csv)

- MAE (Mean Absolute Error): Main performance metric
- Outliers: Count of predictions with error > 3σ
- Test MAE, Val MAE, Train MAE: Error on each dataset split
- Test outlier fraction: Fraction of outliers in test set

## Common Patterns & Non-Obvious Details

### Model Checkpoint Files

- `model_best.pth.tar`: Best validation performance (use for inference)
- `checkpoint.pth.tar`: Latest epoch (for resuming training)
- Format: PyTorch `.tar` file containing state_dict and metadata

### CIFData Graph Construction

- Atoms within cutoff distance become edges (default: 8Å)
- Edges are bidirectional
- Distances are encoded in edge features

### ConvLayer Message Passing

1. Neighbor atoms aggregate their features (weighted by distance)
2. Central atom combines aggregated info with its own features
3. Non-linear activation applied
4. Results in updated atom features for next layer

### Pooling Strategy

- After `n_conv` ConvLayers, all atom features are averaged
- Averaged vector passed through dense network for final prediction

### Validation & Checkpointing

- Model saves when validation MAE improves
- Training can be stopped early if no improvement for N epochs
- Best model always available at `model_best.pth.tar`

## Pre-trained Models

Pre-trained models are available in the `pre-trained/` directory. Use with `inference.py`:

```python
from cgcnn.inference import predict_model
predictions = predict_model(model_path, structure_dir, atom_init_path)
```

## Important Notes

1. CIF Format: All structures must be valid CIF files. Use pymatgen to validate if needed.
2. Element Features: `atom_init.json` must contain all elements in your structures.
3. Normalization: Model may apply property normalization during training.
4. GPU Support: Training uses CUDA if available, falls back to CPU.
5. Reproducibility: Set seed before training for reproducible results.
6. Feature Scaling: Consider normalizing properties before training for better convergence.

## Debugging Common Issues

Issue: Model doesn't converge

- Check property values aren't extreme (consider normalization)
- Increase `n_conv` for complex structures
- Adjust learning rate

### Issue: Out of memory

- Reduce `batch_size`
- Reduce `atom_fea_len` or `h_fea_len`
- Use gradient accumulation

### Issue: Poor generalization (high test error)

- Increase `weight_decay` for regularization
- Increase `n_h` to reduce model capacity
- Check data for outliers or mislabeling

### Issue: CIFData fails to load

- Verify CIF files are valid (use `pymatgen.core.IStructure.from_file`)
- Ensure `id_prop.csv` structure IDs match CIF filenames
- Check `atom_init.json` contains all elements in structures

## References

- README.md: Basic project overview
- TUNING_README.md: Detailed hyperparameter tuning documentation
- Paper: "Crystal Graph Convolutional Neural Networks for an Accurate and Interpretable Prediction of Material Properties" (Xie & Grossman, 2018)
