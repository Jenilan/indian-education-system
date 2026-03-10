import pandas as pd

from src.cleaning import clean_text_columns, impute_median


def test_clean_text_columns_removes_special_chars():
    df = pd.DataFrame({"name": ["A!@#", " B "], "age": [10, 20]})
    cleaned = clean_text_columns(df)
    assert cleaned.loc[0, "name"] == "A"
    assert cleaned.loc[1, "name"] == "B"


def test_impute_median_fills_nans():
    df = pd.DataFrame({"value": [1, None, 3]})
    imputed = impute_median(df, ["value"])
    assert imputed["value"].isna().sum() == 0
    assert imputed.loc[1, "value"] == 2
