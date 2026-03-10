"""Data loading utilities for the Indian Education System project.

This module provides functions to load local datasets and optionally fetch
publicly available education indicators from the World Bank API.

The code is written to be robust in environments where internet access may not
be available, falling back to local CSV assets when necessary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from .utils import data_dir, safe_read_csv


LOGGER = logging.getLogger(__name__)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names python-friendly (snake_case)."""

    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]+", "", regex=True)
    )
    return df


def load_local_district_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load a local district-level dataset.

    Parameters
    ----------
    path:
        Path to a CSV file. If None, will use `data/2015_16_Districtwise.csv` in the
        project data folder.

    Returns
    -------
    pd.DataFrame
        The loaded and cleaned dataset.
    """

    default_path = data_dir() / "2015_16_Districtwise.csv"
    csv_path = Path(path) if path else default_path

    df = safe_read_csv(csv_path)
    df = _clean_columns(df)
    return df


def fetch_world_bank_indicator(
    indicator: str,
    country_code: str = "IND",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    per_page: int = 1000,
) -> pd.DataFrame:
    """Fetch a single indicator from the World Bank API.

    Parameters
    ----------
    indicator:
        The World Bank indicator code (e.g., "SE.PRM.ENRR" for primary school
        enrollment).
    country_code:
        Country ISO code (default: "IND").
    start_year, end_year:
        Optional year filters to limit the result set.

    Returns
    -------
    pandas.DataFrame
        A time series of the indicator.

    Notes
    -----
    The World Bank API returns results in pages; this function will fetch all pages.
    """

    url = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}".format(
        country=country_code, indicator=indicator
    )

    params: Dict[str, object] = {
        "format": "json",
        "per_page": per_page,
    }
    if start_year:
        params["date"] = f"{start_year}:{end_year or ''}".strip(":")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    # data[0] is metadata, data[1] is list of observations
    if not data or len(data) < 2:
        raise ValueError("Unexpected response from World Bank API.")

    rows = data[1]
    df = pd.json_normalize(rows)
    # Keep only the fields the rest of the pipeline expects.
    if "date" in df.columns:
        df["year"] = df["date"].astype(int)
    df = df.rename(columns={"value": indicator, "country.value": "country"})
    return df[["country", "year", indicator]]


def fetch_world_bank_indicators(
    indicators: List[str],
    country_code: str = "IND",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch multiple World Bank indicators and join them into a single dataframe."""

    dfs = []
    for indicator in indicators:
        LOGGER.debug("Fetching indicator %s", indicator)
        df_i = fetch_world_bank_indicator(
            indicator=indicator,
            country_code=country_code,
            start_year=start_year,
            end_year=end_year,
        )
        dfs.append(df_i)

    merged = dfs[0]
    for df_i in dfs[1:]:
        merged = merged.merge(df_i, on=["country", "year"], how="outer")

    return merged
