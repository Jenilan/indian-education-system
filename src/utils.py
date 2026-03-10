"""Utility helpers for the Indian Education System project."""

import os
from pathlib import Path


def project_root() -> Path:
    """Return the root directory of the project."""

    return Path(__file__).resolve().parents[1]


def data_dir() -> Path:
    """Return the default data directory."""

    return project_root() / "data"


def ensure_data_dir() -> Path:
    """Ensure that the data directory exists and return its path."""

    path = data_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def resource_path(*parts: str) -> Path:
    """Return a path to a resource in the project (data, notebooks, etc.)."""

    return project_root().joinpath(*parts)


def safe_read_csv(path: str, **kwargs):
    """Helper wrapper around pandas.read_csv that provides better error messages."""

    import pandas as pd

    try:
        return pd.read_csv(path, **kwargs)
    except FileNotFoundError as ex:
        raise FileNotFoundError(
            f"Could not find CSV file at {path}. Ensure the file exists or pass a valid path."
        ) from ex
