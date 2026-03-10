import pandas as pd

from src import data_loader


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
