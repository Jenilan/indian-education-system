import pandas as pd

from src.analysis import gender_gap_ttest


def test_gender_gap_ttest_returns_stats():
    df = pd.DataFrame({"male_lit": [80, 85, 90], "female_lit": [75, 78, 82]})
    result = gender_gap_ttest(df)
    assert "t_stat" in result and "p_value" in result
    assert isinstance(result["p_value"], float)
