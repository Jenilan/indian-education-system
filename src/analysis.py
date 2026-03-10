"""Analysis and modeling utilities for the Indian Education System project."""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np
import pandas as pd
from scipy.stats import ttest_ind
from sklearn.cluster import KMeans
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression, Ridge
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


LOGGER = logging.getLogger(__name__)


@dataclass
class RegressionResult:
    model_name: str
    model: object
    r2: float
    rmse: float
    mae: float
    y_true: np.ndarray
    y_pred: np.ndarray


def train_regression_models(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.3,
    random_state: int = 42,
    scale: bool = True,
) -> dict[str, RegressionResult]:
    """Train a few regression models on the dataset.

    Returns a dict keyed by model name.
    """

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )

    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
    else:
        X_train_scaled = X_train.values
        X_test_scaled = X_test.values

    results: dict[str, RegressionResult] = {}

    for model_name, model in [
        ("LinearRegression", LinearRegression()),
        ("Ridge", Ridge(alpha=1.0, random_state=random_state)),
        ("RandomForest", RandomForestRegressor(n_estimators=100, random_state=random_state, n_jobs=-1)),
    ]:
        logger = LOGGER.getChild(model_name)
        logger.info("Training %s", model_name)

        m = model.fit(X_train_scaled, y_train)
        y_pred = m.predict(X_test_scaled)

        mse = mean_squared_error(y_test, y_pred)
        rmse = np.sqrt(mse)
        mae = mean_absolute_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)

        results[model_name] = RegressionResult(
            model_name=model_name,
            model=m,
            r2=r2,
            rmse=rmse,
            mae=mae,
            y_true=y_test.to_numpy(),
            y_pred=y_pred,
        )

    return results


def kmeans_region_clustering(
    df: pd.DataFrame,
    feature_columns: list[str],
    n_clusters: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    """Run KMeans clustering on a subset of features and return the dataframe with cluster labels."""

    df = df.copy()
    X = df[feature_columns].copy()
    X = X.fillna(X.median())
    kmeans = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    df["cluster_label"] = kmeans.fit_predict(X)
    return df


def describe_cluster_centers(df: pd.DataFrame, feature_columns: list[str]) -> pd.DataFrame:
    """Return summary stats for each cluster center."""

    if "cluster_label" not in df.columns:
        raise ValueError("DataFrame must contain a 'cluster_label' column.")

    return df.groupby("cluster_label")[feature_columns].mean().reset_index()


def gender_gap_ttest(
    df: pd.DataFrame,
    male_column: str = "male_lit",
    female_column: str = "female_lit",
) -> dict[str, float]:
    """Perform a t-test to compare male and female literacy distributions.

    Returns a dict containing the t-statistic and p-value.
    """

    if male_column not in df.columns or female_column not in df.columns:
        raise ValueError("DataFrame must contain both male and female literacy columns.")

    male = df[male_column].dropna()
    female = df[female_column].dropna()

    t_stat, p_value = ttest_ind(male, female, equal_var=False, nan_policy="omit")

    return {"t_stat": float(t_stat), "p_value": float(p_value)}
