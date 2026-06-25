# MP Energy Above Hull Training Notes

This note records the factors that mattered for the Materials Project
`energy_above_hull` CGCNN training run. It is intentionally narrow: it documents
the observed failure mode and the operating pattern that avoided it.

## Dataset

Input data came from the Materials Project CIF folder and summary CSV:

- Source CIF tree: `~/run/mp-cif`
- Source summary CSV: `~/run/mp-cif/mp_all_summary.csv`
- Target column: `energy_above_hull`
- Prepared training directory: `~/run/training`
- Prepared target file: `~/run/training/id_prop.csv`
- Atom initializer: symlink to
  `~/cgcnn/data/sample-regression/atom_init.json`

The run used `energy_above_hull` from the source CSV, not formation energy from
the pretrained model.

## Failure Mode

The early training attempts failed in PyTorch `DataLoader` multiprocessing with:

```text
RuntimeError: received 0 items of ancdata
```

The important point is that this was not a normal CIF parsing failure. CIF
warnings were observed, but they were not the fatal error. The fatal error
happened while multiprocessing workers were sending tensor batches back to the
main training process.

In practical terms, many `DataLoader` workers were reading crystal data,
building tensors, and passing those tensors across process boundaries. Under
that load, PyTorch's inter-process communication path failed. The `ancdata`
message means the main process expected operating-system control data attached
to the message, such as handles for shared tensor storage, but received none.

This is a data transfer failure between processes, not evidence that two
training workers were racing to update the same model weights. The model update
still happens in the main training process. The fragile part was moving many
prepared graph tensors from worker processes to the parent process.

## Factors That Fixed The Run

The successful pattern avoided the fragile multiprocessing path and reduced file
system pressure:

1. Use the graph-cache dataset format.

   Training from graph cache avoids reparsing CIF files inside the training
   loop. The cache contains precomputed graph arrays, and the target values are
   read from `id_prop.csv`.

2. Put the graph cache on node-local scratch.

   Copying the graph cache to `/scratch/$SLURM_JOB_ID/...` made batch loading
   fast enough with a single loader process and avoided repeated small-file
   reads from the shared home filesystem.

3. Run the `DataLoader` with `workers=0`.

   With `workers=0`, the main training process loads batches itself. This gives
   up parallel loading, but it removes the inter-process tensor transfer that
   triggered the `ancdata` failure.

4. Keep more graph shards cached in memory.

   Increasing the graph-shard cache limit reduced repeated shard reloads. This
   mattered because `workers=0` makes the main process responsible for both
   loading and training.

## Working Command Shape

The successful training command used this structure:

```bash
uv run python tools/train_energy_above_hull.py \
  --reuse-prepared \
  --run-dir ~/run/training \
  --source-csv ~/run/mp-cif/mp_all_summary.csv \
  --cif-root ~/run/mp-cif \
  --atom-init ~/cgcnn/data/sample-regression/atom_init.json \
  --dataset-format graph_cache \
  --cache-dir /scratch/$SLURM_JOB_ID/cgcnn/mp_ehull_graph_cache \
  --id-prop-file ~/run/training/id_prop.csv \
  --epochs 30 \
  --batch-size 256 \
  --workers 0 \
  --seed 123 \
  --lr 0.02 \
  --optim SGD \
  --atom-fea-len 64 \
  --h-fea-len 32 \
  --n-conv 4 \
  --n-h 1 \
  --weight-decay 0.0 \
  --print-freq 10 \
  --skip-epoch-parity
```

## Run Folder Provenance

`~/run/training` was the persistent training run folder. It was used
to hold prepared inputs, run metadata, logs, checkpoints, and model outputs.

Inputs:

- `id_prop.csv`, generated from valid `energy_above_hull` rows in the Materials
  Project summary CSV.
- CIF symlinks pointing back to `~/run/mp-cif`.
- `atom_init.json`, symlinked from the CGCNN sample regression data.
- Command-line options recorded by the training command.
- Graph-cache inputs read from node-local scratch during the successful run.

Outputs:

- Training logs.
- `checkpoint.pth.tar`.
- `model_best.pth.tar`.
- Epoch checkpoints.
- `training_history.json`.
- Final prediction and evaluation files when the training script completes.

The node-local scratch graph cache was an execution-time copy used for speed.
The durable provenance remained the original graph cache, source CSV, CIF tree,
and the prepared training folder.
