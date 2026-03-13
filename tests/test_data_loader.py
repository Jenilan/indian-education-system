import pandas as pd

from src import data_loader
from src.data_loader import standardize_district_schema


def test_standardize_district_schema_renames_and_coerces_percent():
    raw = pd.DataFrame(
        {
            "State_Name": ["Gujarat"],
            "District_Name": ["Ahmedabad"],
            "Male_Literacy_Rate": [0.9],
            "Female_Literacy_Rate": [0.8],
        }
    )
    df = standardize_district_schema(raw, dataset_year="2025-2026")
    assert "statname" in df.columns
    assert "district" in df.columns
    assert "male_lit" in df.columns and "female_lit" in df.columns
    assert df.loc[0, "male_lit"] == 90.0
    assert df.loc[0, "female_lit"] == 80.0
    assert "overall_li" in df.columns
    assert df.loc[0, "overall_li"] == 85.0
    assert "year" in df.columns


def test_load_local_district_data_loads_sample(tmp_path):
    # Create a minimal CSV with expected columns.
    csv_path = tmp_path / "sample.csv"
    sample = pd.DataFrame(
        {
            "STATNAME": ["TestState"],
            "DISTRICT": ["TestDistrict"],
            "OVERALL_LI": [80],
            "MALE_LIT": [85],
            "FEMALE_LIT": [75],
        }
    )
    sample.to_csv(csv_path, index=False)

    df = data_loader.load_local_district_data(path=str(csv_path))
    assert isinstance(df, pd.DataFrame)
    assert "statname" in df.columns
    assert "overall_li" in df.columns
    # should have at least one row from the sample data
    assert df.shape[0] == 1
