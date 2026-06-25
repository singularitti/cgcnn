from __future__ import annotations

import importlib

_MODULE_EXPORTS = {
    "benchmark": [
        "compute_metrics",
        "format_metrics_report",
        "run_benchmark",
        "write_output_csv",
    ],
    "data": [
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
    ],
    "device": [
        "get_env_device",
        "resolve_device",
        "use_pinned_memory",
    ],
    "inference": ["predict_model"],
    "model": ["ConvLayer", "CrystalGraphConvNet"],
    "process_cleanup": [
        "ProcessInfo",
        "TimedProcessInfo",
        "cleanup_orphaned_python_workers",
        "cleanup_stale_torch_shm_managers",
        "ensure_orphaned_worker_reaper",
        "find_orphaned_python_workers",
        "find_torch_shm_managers",
        "list_processes",
        "list_timed_processes",
    ],
    "tools": [
        "NONNEGATIVE_TRANSFORMS",
        "POSITIVE_ONLY_TRANSFORMS",
        "SUPPORTED_TRANSFORMS",
        "RegressorStage",
        "ThresholdClassifier",
        "ThresholdRegressionPlan",
        "TransformName",
        "apply_transform",
        "build_regressor_stages",
        "generate_threshold_labels",
        "invert_transform",
        "split_indices_by_threshold",
        "split_items_by_threshold",
        "validate_transform_input",
    ],
    "training": ["train_model"],
    "utils": [
        "AverageMeter",
        "Normalizer",
        "_validate",
        "classification_metric_value",
        "class_eval",
        "mae",
        "save_checkpoint",
    ],
}

_EXPORT_TO_MODULE = {
    export: module
    for module, exports in _MODULE_EXPORTS.items()
    for export in exports
}

_SUBMODULES = {
    "benchmark",
    "data",
    "device",
    "inference",
    "model",
    "process_cleanup",
    "tools",
    "training",
    "utils",
}

__all__ = sorted([*_EXPORT_TO_MODULE, *_SUBMODULES])


def __getattr__(name: str):
    if name in _SUBMODULES:
        module = importlib.import_module(f"{__name__}.{name}")
        globals()[name] = module
        return module
    if name in _EXPORT_TO_MODULE:
        module = importlib.import_module(f"{__name__}.{_EXPORT_TO_MODULE[name]}")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted([*globals(), *__all__])
