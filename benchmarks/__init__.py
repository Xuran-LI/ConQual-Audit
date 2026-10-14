"""The four benchmarks of the paper. get("prefeval" | "lamp1" | "lamp2" | "lamp4") -> the dataset module."""

import importlib

DATASETS = ("prefeval", "lamp1", "lamp2", "lamp4")


def get(name):
    if name not in DATASETS:
        raise ValueError(f"Unknown dataset {name!r}; choose from {DATASETS}.")
    return importlib.import_module(f"benchmarks.{name}")
