"""Geospatial helpers for the Indian Education System project.

This module contains helpers to map state names to approximate geographic
coordinates for visualizations.

The approach here uses state centroid coordinates for scatter/choropleth-style
maps without requiring a large GeoJSON file.
"""

from __future__ import annotations

from typing import Dict, Optional, Tuple

import pandas as pd


# Approximate state/union territory centroids (latitude, longitude).
# Data sourced from public coordinates of state capitals.
STATE_CENTROIDS: Dict[str, Tuple[float, float]] = {
    "andhra pradesh": (15.9129, 79.7400),
    "arunachal pradesh": (28.2180, 94.7278),
    "assam": (26.2006, 92.9376),
    "bihar": (25.0961, 85.3131),
    "chhattisgarh": (21.2951, 81.8282),
    "goa": (15.2993, 74.1240),
    "gujarat": (22.2587, 71.1924),
    "haryana": (29.0588, 76.0856),
    "himachal pradesh": (31.1048, 77.1734),
    "jharkhand": (23.6102, 85.2799),
    "karnataka": (15.3173, 75.7139),
    "kerala": (10.8505, 76.2711),
    "madhya pradesh": (22.9734, 78.6569),
    "maharashtra": (19.7515, 75.7139),
    "manipur": (24.6637, 93.9063),
    "meghalaya": (25.4670, 91.3662),
    "mizoram": (23.1645, 92.9376),
    "nagaland": (26.1584, 94.5624),
    "odisha": (20.9517, 85.0985),
    "punjab": (31.1471, 75.3412),
    "rajasthan": (27.0238, 74.2179),
    "sikkim": (27.5330, 88.5122),
    "tamil nadu": (11.1271, 78.6569),
    "telangana": (18.1124, 79.0193),
    "tripura": (23.9408, 91.9882),
    "uttar pradesh": (26.8467, 80.9462),
    "uttarakhand": (30.0668, 79.0193),
    "west bengal": (22.9868, 87.8550),
    "delhi": (28.7041, 77.1025),
    "chandigarh": (30.7333, 76.7794),
    "puducherry": (11.9416, 79.8083),
}


def get_state_centroid(state_name: str) -> Optional[Tuple[float, float]]:
    """Return latitude/longitude for a given state name (case-insensitive)."""

    if not state_name:
        return None
    return STATE_CENTROIDS.get(state_name.strip().lower())


def add_state_coordinates(
    df: pd.DataFrame,
    state_column: str = "statname",
    lat_col: str = "latitude",
    lon_col: str = "longitude",
) -> pd.DataFrame:
    """Add latitude/longitude columns to a dataframe based on state names."""

    df = df.copy()
    df[lat_col] = None
    df[lon_col] = None

    for idx, state in df[state_column].fillna("").items():
        cent = get_state_centroid(state)
        if cent:
            df.at[idx, lat_col] = cent[0]
            df.at[idx, lon_col] = cent[1]

    return df
