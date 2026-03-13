import pandas as pd

from src.geodata import add_state_coordinates, get_state_centroid


def test_get_state_centroid_returns_lat_lon():
    centroid = get_state_centroid("Gujarat")
    assert centroid is not None
    assert isinstance(centroid, tuple)
    assert len(centroid) == 2


def test_add_state_coordinates_adds_columns():
    df = pd.DataFrame({"statname": ["Gujarat", "Bihar", "Unknown"]})
    result = add_state_coordinates(df, state_column="statname")
    assert "latitude" in result.columns
    assert "longitude" in result.columns
    assert pd.notna(result.loc[0, "latitude"])
    assert pd.isna(result.loc[2, "latitude"])
