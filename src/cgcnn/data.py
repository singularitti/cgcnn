import csv
import functools
import json
import os
import random
import warnings
from collections import OrderedDict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
import torch
from pymatgen.core.structure import Structure
from torch.utils.data import DataLoader, Dataset
from torch.utils.data.dataloader import default_collate
from torch.utils.data.sampler import SubsetRandomSampler

__all__ = [
    "AtomCustomJSONInitializer",
    "AtomInitializer",
    "CachedGraphData",
    "CIFData",
    "GaussianDistance",
    "build_crystal_graph",
    "collate_pool",
    "get_train_val_test_loader",
    "graph_arrays_to_tensors",
    "load_cif_structure",
]


def get_train_val_test_loader(
    dataset,
    collate_fn=default_collate,
    batch_size=64,
    train_ratio=None,
    val_ratio=0.1,
    test_ratio=0.1,
    return_test=False,
    num_workers=1,
    pin_memory=False,
    train_indices=None,
    val_indices=None,
    test_indices=None,
    **kwargs,
):
    """
    Utility function for dividing a dataset to train, val, test datasets.

    !!! The dataset needs to be shuffled before using the function !!!

    Parameters
    ----------
    dataset: torch.utils.data.Dataset
      The full dataset to be divided.
    collate_fn: torch.utils.data.DataLoader
    batch_size: int
    train_ratio: float
    val_ratio: float
    test_ratio: float
    return_test: bool
      Whether to return the test dataset loader. If False, the last test_size
      data will be hidden.
    num_workers: int
    pin_memory: bool

    Returns
    -------
    train_loader: torch.utils.data.DataLoader
      DataLoader that random samples the training data.
    val_loader: torch.utils.data.DataLoader
      DataLoader that random samples the validation data.
    (test_loader): torch.utils.data.DataLoader
      DataLoader that random samples the test data, returns if
        return_test=True.
    """
    explicit_indices = any(
        indices is not None for indices in [train_indices, val_indices, test_indices]
    )
    if explicit_indices:
        if train_indices is None or val_indices is None:
            raise ValueError(
                "train_indices and val_indices must be provided when using explicit splits."
            )
        if return_test and test_indices is None:
            raise ValueError("test_indices must be provided when return_test=True.")
        train_sampler = SubsetRandomSampler(train_indices)
        val_sampler = SubsetRandomSampler(val_indices)
        if return_test:
            test_sampler = SubsetRandomSampler(test_indices)
    else:
        total_size = len(dataset)
        if kwargs["train_size"] is None:
            if train_ratio is None:
                assert val_ratio + test_ratio < 1
                train_ratio = 1 - val_ratio - test_ratio
                print(
                    f"[Warning] train_ratio is None, using 1 - val_ratio - "
                    f"test_ratio = {train_ratio} as training data."
                )
            else:
                assert train_ratio + val_ratio + test_ratio <= 1
        indices = list(range(total_size))
        if kwargs["train_size"]:
            train_size = kwargs["train_size"]
        else:
            train_size = int(train_ratio * total_size)
        if kwargs["test_size"]:
            test_size = kwargs["test_size"]
        else:
            test_size = int(test_ratio * total_size)
        if kwargs["val_size"]:
            valid_size = kwargs["val_size"]
        else:
            valid_size = int(val_ratio * total_size)
        train_sampler = SubsetRandomSampler(indices[:train_size])
        val_sampler = SubsetRandomSampler(
            indices[-(valid_size + test_size) : -test_size]
        )
        if return_test:
            test_sampler = SubsetRandomSampler(indices[-test_size:])
    train_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=train_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    val_loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=val_sampler,
        num_workers=num_workers,
        collate_fn=collate_fn,
        pin_memory=pin_memory,
    )
    if return_test:
        test_loader = DataLoader(
            dataset,
            batch_size=batch_size,
            sampler=test_sampler,
            num_workers=num_workers,
            collate_fn=collate_fn,
            pin_memory=pin_memory,
        )
    if return_test:
        return train_loader, val_loader, test_loader
    else:
        return train_loader, val_loader


def collate_pool(dataset_list):
    """
    Collate a list of data and return a batch for predicting crystal
    properties.

    Parameters
    ----------

    dataset_list: list of tuples for each data point.
      (atom_fea, nbr_fea, nbr_fea_idx, target)

      atom_fea: torch.Tensor shape (n_i, atom_fea_len)
      nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
      nbr_fea_idx: torch.LongTensor shape (n_i, M)
      target: torch.Tensor shape (n_targets, )
      cif_id: str or int

    Returns
    -------
    N = sum(n_i); N0 = sum(i)

    batch_atom_fea: torch.Tensor shape (N, orig_atom_fea_len)
      Atom features from atom type
    batch_nbr_fea: torch.Tensor shape (N, M, nbr_fea_len)
      Bond features of each atom's M neighbors
    batch_nbr_fea_idx: torch.LongTensor shape (N, M)
      Indices of M neighbors of each atom
    crystal_atom_idx: list of torch.LongTensor of length N0
      Mapping from the crystal idx to atom idx
    target: torch.Tensor shape (N, n_targets)
      Target value for prediction
    batch_cif_ids: list
    """
    batch_atom_fea, batch_nbr_fea, batch_nbr_fea_idx = [], [], []
    crystal_atom_idx, batch_target = [], []
    batch_cif_ids = []
    base_idx = 0
    for i, ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id) in enumerate(
        dataset_list
    ):
        n_i = atom_fea.shape[0]  # number of atoms for this crystal
        batch_atom_fea.append(atom_fea)
        batch_nbr_fea.append(nbr_fea)
        batch_nbr_fea_idx.append(nbr_fea_idx + base_idx)
        new_idx = torch.LongTensor(np.arange(n_i) + base_idx)
        crystal_atom_idx.append(new_idx)
        batch_target.append(target)
        batch_cif_ids.append(cif_id)
        base_idx += n_i
    return (
        (
            torch.cat(batch_atom_fea, dim=0),
            torch.cat(batch_nbr_fea, dim=0),
            torch.cat(batch_nbr_fea_idx, dim=0),
            crystal_atom_idx,
        ),
        torch.stack(batch_target, dim=0),
        batch_cif_ids,
    )


class GaussianDistance:
    """
    Expands the distance by Gaussian basis.

    Unit: angstrom
    """

    def __init__(self, dmin, dmax, step, var=None):
        """
        Parameters
        ----------

        dmin: float
          Minimum interatomic distance
        dmax: float
          Maximum interatomic distance
        step: float
          Step size for the Gaussian filter
        """
        assert dmin < dmax
        assert dmax - dmin > step
        self.filter = np.arange(dmin, dmax + step, step)
        if var is None:
            var = step
        self.var = var

    def expand(self, distances):
        """
        Apply Gaussian disntance filter to a numpy distance array

        Parameters
        ----------

        distance: np.array shape n-d array
          A distance matrix of any shape

        Returns
        -------
        expanded_distance: shape (n+1)-d array
          Expanded distance matrix with the last dimension of length
          len(self.filter)
        """
        return np.exp(-((distances[..., np.newaxis] - self.filter) ** 2) / self.var**2)


class AtomInitializer:
    """
    Base class for intializing the vector representation for atoms.

    !!! Use one AtomInitializer per dataset !!!
    """

    def __init__(self, atom_types):
        self.atom_types = set(atom_types)
        self._embedding = {}

    def get_atom_fea(self, atom_type):
        assert atom_type in self.atom_types
        return self._embedding[atom_type]

    def load_state_dict(self, state_dict):
        self._embedding = state_dict
        self.atom_types = set(self._embedding.keys())
        self._decodedict = {
            idx: atom_type for atom_type, idx in self._embedding.items()
        }

    def state_dict(self):
        return self._embedding

    def decode(self, idx):
        if not hasattr(self, "_decodedict"):
            self._decodedict = {
                idx: atom_type for atom_type, idx in self._embedding.items()
            }
        return self._decodedict[idx]


class AtomCustomJSONInitializer(AtomInitializer):
    """
    Initialize atom feature vectors using a JSON file, which is a python
    dictionary mapping from element number to a list representing the
    feature vector of the element.

    Parameters
    ----------

    elem_embedding_file: str
        The path to the .json file
    """

    def __init__(self, elem_embedding_file):
        with open(elem_embedding_file) as f:
            elem_embedding = json.load(f)
        elem_embedding = {int(key): value for key, value in elem_embedding.items()}
        atom_types = set(elem_embedding.keys())
        super().__init__(atom_types)
        for key, value in elem_embedding.items():
            self._embedding[key] = np.array(value, dtype=float)


def load_cif_structure(cif_path: os.PathLike | str) -> Structure:
    """Load one CIF file as a pymatgen Structure."""
    return Structure.from_file(str(cif_path))


def build_crystal_graph(
    crystal: Structure,
    atom_initializer: AtomInitializer,
    gaussian_distance: GaussianDistance,
    max_num_nbr: int = 12,
    radius: float = 8,
    cif_id: str | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build CGCNN graph arrays from an already-loaded crystal structure."""
    atom_fea = np.vstack([
        atom_initializer.get_atom_fea(crystal[i].specie.number)
        for i in range(len(crystal))
    ]).astype(np.float32, copy=False)
    all_nbrs = crystal.get_all_neighbors(radius, include_index=True)
    all_nbrs = [sorted(nbrs, key=lambda x: x[1]) for nbrs in all_nbrs]
    nbr_fea_idx, nbr_fea = [], []
    for nbr in all_nbrs:
        if len(nbr) < max_num_nbr:
            prefix = f"{cif_id} " if cif_id is not None else ""
            warnings.warn(
                f"{prefix}not find enough neighbors to build graph. "
                "If it happens frequently, consider increase "
                "radius."
            )
            nbr_fea_idx.append(
                list(map(lambda x: x[2], nbr)) + [0] * (max_num_nbr - len(nbr))
            )
            nbr_fea.append(
                list(map(lambda x: x[1], nbr))
                + [radius + 1.0] * (max_num_nbr - len(nbr))
            )
        else:
            nbr_fea_idx.append(list(map(lambda x: x[2], nbr[:max_num_nbr])))
            nbr_fea.append(list(map(lambda x: x[1], nbr[:max_num_nbr])))
    nbr_fea_idx = np.array(nbr_fea_idx, dtype=np.int64)
    nbr_fea = gaussian_distance.expand(np.array(nbr_fea, dtype=np.float32))
    nbr_fea = nbr_fea.astype(np.float32, copy=False)
    return atom_fea, nbr_fea, nbr_fea_idx


def graph_arrays_to_tensors(
    atom_fea: np.ndarray,
    nbr_fea: np.ndarray,
    nbr_fea_idx: np.ndarray,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Convert cached or freshly-built graph arrays to CGCNN tensors."""
    return (
        torch.as_tensor(np.ascontiguousarray(atom_fea), dtype=torch.float32),
        torch.as_tensor(np.ascontiguousarray(nbr_fea), dtype=torch.float32),
        torch.as_tensor(np.ascontiguousarray(nbr_fea_idx), dtype=torch.long),
    )


class CachedGraphData(Dataset):
    """
    Dataset wrapper for a sharded CGCNN graph cache.

    The cache stores primitive NumPy arrays in shard `.npz` files and returns
    the same item structure as CIFData:
    ((atom_fea, nbr_fea, nbr_fea_idx), target, cif_id).
    """

    def __init__(
        self,
        cache_dir,
        random_seed=123,
        shuffle=True,
        include_ids: Iterable[str] | None = None,
        max_cached_shards: int = 4,
    ):
        self.cache_dir = Path(cache_dir)
        if not self.cache_dir.exists():
            raise FileNotFoundError(f"cache_dir does not exist: {self.cache_dir}")
        manifest_file = self.cache_dir / "manifest.json"
        if not manifest_file.is_file():
            raise FileNotFoundError(f"manifest.json does not exist: {manifest_file}")
        with manifest_file.open() as handle:
            self.manifest = json.load(handle)
        if self.manifest.get("schema_version") != "cgcnn_graph_cache_v1":
            raise ValueError(
                "Unsupported graph cache schema: "
                f"{self.manifest.get('schema_version')}"
            )
        self.shards_dir = self.cache_dir / "shards"
        if not self.shards_dir.is_dir():
            raise FileNotFoundError(f"Shard directory does not exist: {self.shards_dir}")

        self.index = self.manifest.get("index")
        if not isinstance(self.index, dict):
            raise ValueError("manifest.json must contain an object-valued index.")

        id_prop_file = self.cache_dir / "id_prop.csv"
        if not id_prop_file.is_file():
            raise FileNotFoundError(f"id_prop.csv does not exist: {id_prop_file}")
        with id_prop_file.open() as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader if row]
        if include_ids is not None:
            include_ids = set(include_ids)
            self.id_prop_data = [
                row for row in self.id_prop_data if row and row[0] in include_ids
            ]
        if not self.id_prop_data:
            raise ValueError("id_prop.csv is empty!")
        self.n_targets = len(self.id_prop_data[0]) - 1
        if self.n_targets < 1:
            raise ValueError("id_prop.csv must contain at least one target column.")
        for row in self.id_prop_data:
            if len(row) != self.n_targets + 1:
                raise ValueError(
                    "All rows in id_prop.csv must have the same number of columns. "
                    "Expected {} target columns but got {} for id {}.".format(
                        self.n_targets, len(row) - 1, row[0] if row else "unknown"
                    )
                )
            if row[0] not in self.index:
                raise ValueError(f"Cached graph index is missing id {row[0]}.")
        if shuffle:
            random.seed(random_seed)
            random.shuffle(self.id_prop_data)

        self.max_cached_shards = max_cached_shards
        self._shard_cache: OrderedDict[str, dict[str, np.ndarray]] = OrderedDict()

    def __len__(self):
        return len(self.id_prop_data)

    @staticmethod
    def _decode_id(value) -> str:
        if isinstance(value, bytes):
            return value.decode("utf-8")
        if isinstance(value, np.bytes_):
            return bytes(value).decode("utf-8")
        return str(value)

    @staticmethod
    def _entry_location(entry) -> tuple[str, int]:
        if isinstance(entry, dict):
            return str(entry["shard"]), int(entry["row"])
        if isinstance(entry, (list, tuple)) and len(entry) == 2:
            return str(entry[0]), int(entry[1])
        raise ValueError(f"Invalid graph cache index entry: {entry!r}")

    def _load_shard(self, shard_name: str) -> dict[str, np.ndarray]:
        if shard_name in self._shard_cache:
            shard = self._shard_cache.pop(shard_name)
            self._shard_cache[shard_name] = shard
            return shard

        shard_path = self.shards_dir / shard_name
        if not shard_path.is_file():
            raise FileNotFoundError(f"Shard not found: {shard_path}")
        with np.load(shard_path, allow_pickle=False) as data:
            shard = {key: data[key] for key in data.files}
        self._shard_cache[shard_name] = shard
        while len(self._shard_cache) > self.max_cached_shards:
            self._shard_cache.popitem(last=False)
        return shard

    def __getitem__(self, idx):
        cif_id = self.id_prop_data[idx][0]
        shard_name, row_index = self._entry_location(self.index[cif_id])
        shard = self._load_shard(shard_name)
        shard_cif_id = self._decode_id(shard["ids"][row_index])
        if shard_cif_id != cif_id:
            raise ValueError(
                f"Manifest index mismatch for {cif_id}: shard row contains {shard_cif_id}"
            )
        atom_offsets = shard["atom_offsets"]
        start = int(atom_offsets[row_index])
        end = int(atom_offsets[row_index + 1])
        atom_fea, nbr_fea, nbr_fea_idx = graph_arrays_to_tensors(
            shard["atom_fea"][start:end],
            shard["nbr_fea"][start:end],
            shard["nbr_fea_idx"][start:end],
        )
        target = torch.as_tensor(
            np.ascontiguousarray(shard["targets"][row_index]),
            dtype=torch.float32,
        )
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id


class CIFData(Dataset):
    """
    The CIFData dataset is a wrapper for a dataset where the crystal structures
    are stored in the form of CIF files. The dataset should have the following
    directory structure:

    root_dir
    ├── id_prop.csv
    ├── atom_init.json
    ├── id0.cif
    ├── id1.cif
    ├── ...

    id_prop.csv: a CSV file whose first column records a unique ID for each
    crystal, and every subsequent column stores one target property value.
    Multiple target properties can be provided per crystal by adding more
    columns.

    atom_init.json: a JSON file that stores the initialization vector for each
    element.

    ID.cif: a CIF file that recodes the crystal structure, where ID is the
    unique ID for the crystal.

    Parameters
    ----------

    root_dir: str
        The path to the root directory of the dataset
    max_num_nbr: int
        The maximum number of neighbors while constructing the crystal graph
    radius: float
        The cutoff radius for searching neighbors
    dmin: float
        The minimum distance for constructing GaussianDistance
    step: float
        The step size for constructing GaussianDistance
    random_seed: int
        Random seed for shuffling the dataset

    Returns
    -------

    atom_fea: torch.Tensor shape (n_i, atom_fea_len)
    nbr_fea: torch.Tensor shape (n_i, M, nbr_fea_len)
    nbr_fea_idx: torch.LongTensor shape (n_i, M)
    target: torch.Tensor shape (n_targets, )
    cif_id: str or int
    """

    def __init__(
        self,
        root_dir,
        max_num_nbr=12,
        radius=8,
        dmin=0,
        step=0.2,
        random_seed=123,
        shuffle=True,
        include_ids: Iterable[str] | None = None,
    ):
        self.root_dir = root_dir
        self.max_num_nbr, self.radius = max_num_nbr, radius
        assert os.path.exists(root_dir), "root_dir does not exist!"
        id_prop_file = os.path.join(self.root_dir, "id_prop.csv")
        assert os.path.exists(id_prop_file), "id_prop.csv does not exist!"
        with open(id_prop_file) as f:
            reader = csv.reader(f)
            self.id_prop_data = [row for row in reader if row]
        if include_ids is not None:
            include_ids = set(include_ids)
            self.id_prop_data = [
                row for row in self.id_prop_data if row and row[0] in include_ids
            ]
        if not self.id_prop_data:
            raise ValueError("id_prop.csv is empty!")
        self.n_targets = len(self.id_prop_data[0]) - 1
        if self.n_targets < 1:
            raise ValueError("id_prop.csv must contain at least one target column.")
        for row in self.id_prop_data:
            if len(row) != self.n_targets + 1:
                raise ValueError(
                    "All rows in id_prop.csv must have the same number of columns. "
                    "Expected {} target columns but got {} for id {}.".format(
                        self.n_targets, len(row) - 1, row[0] if row else "unknown"
                    )
                )
        if shuffle:
            random.seed(random_seed)
            random.shuffle(self.id_prop_data)
        atom_init_file = os.path.join(self.root_dir, "atom_init.json")
        assert os.path.exists(atom_init_file), "atom_init.json does not exist!"
        self.ari = AtomCustomJSONInitializer(atom_init_file)
        self.gdf = GaussianDistance(dmin=dmin, dmax=self.radius, step=step)

    def __len__(self):
        return len(self.id_prop_data)

    @functools.cache  # Cache loaded structures
    def __getitem__(self, idx):
        row = self.id_prop_data[idx]
        cif_id, target_values = row[0], row[1:]
        crystal = load_cif_structure(os.path.join(self.root_dir, cif_id + ".cif"))
        atom_fea, nbr_fea, nbr_fea_idx = graph_arrays_to_tensors(
            *build_crystal_graph(
                crystal,
                self.ari,
                self.gdf,
                max_num_nbr=self.max_num_nbr,
                radius=self.radius,
                cif_id=cif_id,
            )
        )
        target = torch.as_tensor(
            [float(value) for value in target_values],
            dtype=torch.float32,
        )
        return (atom_fea, nbr_fea, nbr_fea_idx), target, cif_id
