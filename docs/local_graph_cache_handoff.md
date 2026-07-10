# Local CGCNN graph cache handoff

This note is for an agent that will build a CGCNN graph cache on a local
computer, transfer that cache to Stampede3, and train from the cached graphs
instead of reparsing CIF files during every remote run.

## Goal

Create a cache from the full Materials Project CIF dataset for the
`energy_above_hull` target. The remote training job should read graph tensors
directly from the cache and avoid opening 154,879 individual CIF files from the
shared filesystem during training.

The source run that motivated this cache used:

- Source CSV: `/work2/04996/tg842951/stampede3/run/mp-cif/mp_all_summary.csv`
- CIF directory: `/work2/04996/tg842951/stampede3/run/mp-cif`
- Target column: `energy_above_hull`
- ID column: `material_id`
- Valid rows: `154879`
- Split seed: `20260624`
- Split counts: train `123903`, validation `15487`, test `15489`
- Atom features: `/home1/04996/tg842951/cgcnn/data/sample-regression/atom_init.json`

## Portability Guidance

Do not cache `pymatgen` objects, Python class instances, object-dtype NumPy
arrays, or arbitrary pickled dataset objects. Those are fragile across Python,
package, and platform versions.

Use a primitive tensor/array schema. Recommended formats:

1. Sharded `.npz` files containing only numeric arrays and UTF-8 byte/string IDs.
2. Or sharded `.pt` files containing only CPU tensors and plain Python lists/dicts.

Prefer `.npz` for maximum macOS-to-Linux portability. `.pt` CPU tensors usually
load across macOS and Linux, including Apple Silicon to x86_64, but they still
depend more heavily on PyTorch serialization compatibility. Stampede3 Skylake is
x86_64 little-endian; modern macOS systems are also little-endian, so numeric
array byte order is not the concern. The main risk is using pickle-bound Python
objects or mismatched library versions.

If using `.npz`, write arrays with explicit little-endian dtypes or native
standard dtypes:

- `atom_fea`: `float32`, shape `(n_atoms, atom_feature_length)`
- `nbr_fea`: `float32`, shape `(n_atoms, max_num_nbr, gaussian_length)`
- `nbr_fea_idx`: `int64`, shape `(n_atoms, max_num_nbr)`
- `target`: `float32`, shape `(1,)`
- `crystal_atom_count`: `int64`, scalar or shape `(1,)`
- `id`: UTF-8 string, or store IDs separately in a shard manifest

## Graph Construction Must Match CGCNN

Match `src/cgcnn/data.py` exactly unless the training code is changed in lockstep:

- `max_num_nbr = 12`
- `radius = 8`
- `dmin = 0`
- `step = 0.2`
- Gaussian basis: `np.arange(dmin, radius + step, step)`, with variance `step`
- Atom features from `data/sample-regression/atom_init.json`
- Neighbor sorting: sort each atom's neighbors by distance and keep the first 12
- If fewer than 12 neighbors exist, pad neighbor indices with `0` and distances
  with `radius + 1.0` before Gaussian expansion

Each cached item must return the same structure tuple as `CIFData.__getitem__`:

```python
((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
```

Then the existing `collate_pool` function can be reused unchanged.

## Recommended Cache Layout

Use shard files to avoid 154,879 tiny cache files. For example:

```text
mp_ehull_graph_cache_v1/
├── manifest.json
├── id_prop.csv
├── splits/
│   ├── train_ids.csv
│   ├── val_ids.csv
│   └── test_ids.csv
└── shards/
    ├── shard_00000.npz
    ├── shard_00001.npz
    └── ...
```

Shard size should be chosen for practical local and remote IO. A good starting
point is 512 to 2048 structures per shard. The manifest should map each
`material_id` to `(shard_name, row_index_or_offsets)` and record all graph
parameters.

For variable-size structures in `.npz`, store concatenated arrays plus offsets:

- `ids`: string array of shape `(n_structures,)`
- `targets`: `float32`, shape `(n_structures, 1)`
- `atom_fea`: concatenated `float32`, shape `(sum_atoms, atom_feature_length)`
- `nbr_fea`: concatenated `float32`, shape `(sum_atoms, max_num_nbr, gaussian_length)`
- `nbr_fea_idx`: concatenated `int64`, shape `(sum_atoms, max_num_nbr)`
- `atom_offsets`: `int64`, shape `(n_structures + 1,)`

To read item `i` in a shard:

```python
start = atom_offsets[i]
end = atom_offsets[i + 1]
atom = atom_fea[start:end]
nbr = nbr_fea[start:end]
nbr_idx = nbr_fea_idx[start:end]
target = targets[i]
cif_id = ids[i]
```

## Required Metadata

Write `manifest.json` with at least:

- `schema_version`: `cgcnn_graph_cache_v1`
- `created_at`
- `source_csv`
- `source_csv_sha256`
- `cif_source`
- `atom_init_sha256`
- `target_column`: `energy_above_hull`
- `id_column`: `material_id`
- `valid_row_count`
- `graph_parameters`: `max_num_nbr`, `radius`, `dmin`, `step`
- `dtype_policy`: `float32` features/targets, `int64` indices/offsets
- `endianness`: little-endian or native little-endian
- `python_version`
- `numpy_version`
- `pymatgen_version`
- `split_seed`
- `split_counts`
- `shards`: file names and structure counts

## Remote Training Changes

Add a cached dataset class in the repo, for example `CachedGraphData`, that:

1. Reads `manifest.json`.
2. Loads only needed shards lazily.
3. Returns `((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)`.
4. Supports explicit `train_ids`, `val_ids`, and `test_ids`.
5. Reuses `collate_pool` and `get_train_val_test_loader`.

Then add a training driver that mirrors
`tools/run_mp_ehull_training_stampede.py`, but uses `CachedGraphData` instead
of `CIFData`. Keep output artifacts and run documentation the same:

- `RUN_LOG.md`
- `run_metadata.json`
- `id_prop_stats.json`
- `training_history.json`
- `checkpoint.pth.tar`
- `model_best.pth.tar`
- `test_results.csv`

## Transfer To Stampede3

After local cache creation, upload the cache directory to `$WORK` or `$SCRATCH`.
For training, prefer copying or unpacking the shard directory onto `$SCRATCH`
inside the run folder. Training should then read a moderate number of large shard
files rather than 154,879 CIF files.

Do not upload temporary local test caches unless explicitly requested. If test
caches are created for development, keep them separate from the full cache and
remove them after the full cache is validated.

## Validation Before Training

On Stampede3, validate the uploaded cache before starting a full training job:

- Manifest exists and parses as JSON.
- `valid_row_count == 154879`.
- Split counts match `123903`, `15487`, `15489`.
- Every split ID appears in the manifest index.
- Load a few real structures from different shards and verify shapes/dtypes.
- Run no reduced training sample unless explicitly requested.

Once validation passes, submit the full cached training job.
