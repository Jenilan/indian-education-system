import pandas as pd

from src.analysis import forecast_time_series, gender_gap_ttest


def test_gender_gap_ttest_returns_stats():
    df = pd.DataFrame({"male_lit": [80, 85, 90], "female_lit": [75, 78, 82]})
    result = gender_gap_ttest(df)
    assert "t_stat" in result and "p_value" in result
    assert isinstance(result["p_value"], float)


def test_forecast_time_series_returns_future_years():
    df = pd.DataFrame({"year": [2015, 2016, 2017], "value": [10, 12, 14]})
    forecast = forecast_time_series(df, year_column="year", value_column="value", periods=2)
    assert forecast["year"].max() == 2019
    assert forecast.shape[0] > df.shape[0]
