# CGCNN Graph Caches

CGCNN graph caches avoid reparsing CIF files during training. A cache stores the
graph inputs returned by `CIFData.__getitem__`:

```python
((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id)
```

`CachedGraphData` reads `atom_fea`, `nbr_fea`, and `nbr_fea_idx` from sharded
`.npz` files. It reads `target` values from an `id_prop.csv` file.

## Compatibility

Two cache layouts are supported.

### v1: old target-coupled cache

This is the layout produced by the first MP `energy_above_hull` cache builder:

```text
mp_ehull_graph_cache_v1/
|-- manifest.json
|-- id_prop.csv
|-- atom_init.json
|-- splits/
|   |-- train_ids.csv
|   |-- val_ids.csv
|   `-- test_ids.csv
`-- shards/
    |-- shard_00000.npz
    |-- shard_00001.npz
    `-- ...
```

It is still compatible. The current reader accepts
`schema_version == "cgcnn_graph_cache_v1"` and falls back to the root
`id_prop.csv` when no newer `labels/` metadata exists.

The old shards may contain a `targets` array. Current code ignores that array and
uses `id_prop.csv` instead. This means the old graph cache can also be reused
with a different target property if you provide a separate compatible
`id_prop.csv`.

Upload the whole old cache directory intact, including:

- `manifest.json`
- `id_prop.csv`
- `splits/*.csv` if you want the same train/validation/test split
- `shards/*.npz`
- `atom_init.json`, `README.md`, and `RUN_LOG.md` for provenance

Do not upload the raw CIF files for cached training unless you also want a CIF
fallback path.

### v2: graph cache with separate label tasks

The newer builder writes target-independent graph shards:

```text
mp_graph_cache_v2/
|-- manifest.json
|-- graph_ids.csv
|-- atom_init.json
|-- labels/
|   `-- energy_above_hull/
|       |-- id_prop.csv
|       `-- splits/
|           |-- train_ids.csv
|           |-- val_ids.csv
|           `-- test_ids.csv
`-- shards/
    |-- shard_00000.npz
    |-- shard_00001.npz
    `-- ...
```

The graph shards are independent of target properties. Add new targets by
creating another label task under `labels/`, not by rebuilding the shards.

## Train From An Old v1 Cache

Use the cache root as `root_dir` and opt into `dataset_format="graph_cache"`.
If you do not pass `id_prop_file`, `CachedGraphData` uses the root
`id_prop.csv`.

```python
from pathlib import Path
import csv

from cgcnn.training import train_model


def read_ids(path):
    with Path(path).open(newline="") as handle:
        return [row["material_id"] for row in csv.DictReader(handle)]


cache_dir = Path("/work/path/mp_ehull_graph_cache_v1")

train_model(
    root_dir=str(cache_dir),
    task="regression",
    dataset_format="graph_cache",
    train_ids=read_ids(cache_dir / "splits" / "train_ids.csv"),
    val_ids=read_ids(cache_dir / "splits" / "val_ids.csv"),
    test_ids=read_ids(cache_dir / "splits" / "test_ids.csv"),
    epochs=30,
    batch_size=64,
    workers=8,
)
```

This path does not open any CIF files. It reads graph arrays from `shards/*.npz`
and targets from `mp_ehull_graph_cache_v1/id_prop.csv`.

## Reuse An Old v1 Cache With Another Target

Create a new CSV with the same `id_prop.csv` format:

```text
material_id_0,target_value
material_id_1,target_value
...
```

Every `material_id` in this file must appear in `manifest.json`'s cache index.
Then pass that file explicitly:

```python
cache_dir = Path("/work/path/mp_ehull_graph_cache_v1")

train_model(
    root_dir=str(cache_dir),
    task="regression",
    dataset_format="graph_cache",
    id_prop_file="/work/path/labels/formation_energy_per_atom_id_prop.csv",
    epochs=30,
    batch_size=64,
    workers=8,
)
```

If you need fixed splits for the new target, create split CSVs for that target
and pass their IDs through `train_ids`, `val_ids`, and `test_ids`.

## Build A New v2 Graph Cache

For a flat Materials Project CIF tree:

```bash
.venv/bin/python tools/build_mp_graph_cache.py \
  --data-root /path/to/mp-cif \
  --source-csv /path/to/mp-cif/mp_all_summary.csv \
  --target-column energy_above_hull
```

This writes graph shards plus one default label task.

## Add A New Target To An Existing v2 Cache

Use `--labels-only` to avoid reading CIFs or rebuilding graph shards:

```bash
.venv/bin/python tools/build_mp_graph_cache.py \
  --labels-only \
  --output-dir /path/to/mp_graph_cache_v2 \
  --source-csv /path/to/mp_all_summary.csv \
  --target-column formation_energy_per_atom
```

Then train with the new label file:

```python
train_model(
    root_dir="/path/to/mp_graph_cache_v2",
    task="regression",
    dataset_format="graph_cache",
    id_prop_file="/path/to/mp_graph_cache_v2/labels/formation_energy_per_atom/id_prop.csv",
)
```

## When To Rebuild Graph Shards

Rebuild graph shards only when graph inputs change:

- CIF files change
- `atom_init.json` changes
- graph parameters change: `max_num_nbr`, `radius`, `dmin`, or `step`

Do not rebuild graph shards just because the target property changes.
