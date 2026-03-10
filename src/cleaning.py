"""Data cleaning utilities for the Indian Education System project."""

from __future__ import annotations

import numpy as np
import pandas as pd


def clean_column_names(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names to be snake_case and free of special characters."""

    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]+", "", regex=True)
    )
    return df


def drop_irrelevant_columns(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Drop columns that are not needed for analysis."""

    return df.drop(columns=[c for c in columns if c in df.columns], errors="ignore")


def impute_median(df: pd.DataFrame, columns: list[str]) -> pd.DataFrame:
    """Impute missing values in given columns using the median."""

    df = df.copy()
    for col in columns:
        if col in df.columns:
            median = df[col].median(skipna=True)
            df[col] = df[col].fillna(median)
    return df


def clean_text_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Clean string/object columns by stripping whitespace and removing symbols."""

    df = df.copy()
    for col in df.select_dtypes(include=["object"]):
        df[col] = (
            df[col]
            .astype(str)
            .str.strip()
            .str.replace(r"[^A-Za-z0-9 _]+", "", regex=True)
            .replace({"": np.nan})
        )
    return df


def filter_year_range(df: pd.DataFrame, start: int, end: int, year_column: str = "year") -> pd.DataFrame:
    """Filter a dataframe to be within a year range."""

    if year_column not in df.columns:
        return df

    return df.loc[(df[year_column] >= start) & (df[year_column] <= end)]


def reorder_categories(df: pd.DataFrame, cat_column: str, ordering: list[str]) -> pd.DataFrame:
    """Reorder a categorical column to a specific order."""

    if cat_column not in df.columns:
        return df

    df = df.copy()
    df[cat_column] = pd.Categorical(df[cat_column], categories=ordering, ordered=True)
    return df
