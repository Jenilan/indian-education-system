"""Visualization helpers for the Indian Education System project."""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import seaborn as sns

import pandas as pd


def plot_correlation_matrix(df: pd.DataFrame, figsize: tuple[int, int] = (12, 10)) -> None:
    """Plot a static correlation matrix using seaborn."""

    numeric = df.select_dtypes(include=[np.number])
    corr = numeric.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, fmt=".2f")
    plt.title("Correlation Matrix")
    plt.tight_layout()


def interactive_scatter_literacy(df: pd.DataFrame, x_col: str = "male_lit", y_col: str = "female_lit") -> go.Figure:
    """Interactive scatter plot of male vs female literacy rates."""

    district_col = None
    for c in ["district", "districts", "distname", "dist_nm"]:
        if c in df.columns:
            district_col = c
            break

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="statname" if "statname" in df.columns else None,
        hover_data=[district_col] if district_col else None,
        title="Male vs Female Literacy Rates",
        labels={x_col: "Male Literacy (%)", y_col: "Female Literacy (%)"},
    )
    fig.update_traces(marker=dict(size=8, opacity=0.7), selector=dict(mode="markers"))
    fig.update_layout(legend_title_text="State")
    return fig


def plot_state_literacy_bar(df: pd.DataFrame, state_column: str = "statname", value_column: str = "overall_li") -> go.Figure:
    """Interactive bar chart of average literacy by state."""

    state_avg = df.groupby(state_column)[value_column].mean().sort_values(ascending=False).reset_index()
    fig = px.bar(
        state_avg,
        x=value_column,
        y=state_column,
        orientation="h",
        title="Average Overall Literacy Rate by State",
        labels={value_column: "Average Literacy Rate (%)", state_column: "State"},
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"))
    return fig


def plot_model_performance(result) -> go.Figure:
    """Plot actual vs predicted values from a regression result."""

    fig = px.scatter(
        x=result.y_true,
        y=result.y_pred,
        labels={"x": "Actual", "y": "Predicted"},
        title=f"{result.model_name} - Actual vs Predicted",
    )
    minv = min(result.y_true.min(), result.y_pred.min())
    maxv = max(result.y_true.max(), result.y_pred.max())
    fig.add_shape(
        type="line",
        x0=minv,
        x1=maxv,
        y0=minv,
        y1=maxv,
        line=dict(color="red", dash="dash"),
    )
    return fig


def plot_state_bubble_map(
    df: pd.DataFrame,
    value_column: str = "overall_li",
    state_column: str = "statname",
    lat_column: str = "latitude",
    lon_column: str = "longitude",
) -> go.Figure:
    """Plot an interactive bubble map for Indian states based on a value."""

    map_df = df.dropna(subset=[lat_column, lon_column, value_column]).copy()
    if map_df.empty:
        raise ValueError("No geographic data available for mapping.")

    fig = px.scatter_geo(
        map_df,
        lat=lat_column,
        lon=lon_column,
        size=value_column,
        color=value_column,
        hover_name=state_column,
        hover_data={
            value_column: True,
            lat_column: False,
            lon_column: False,
        },
        title=f"India State-level {value_column.replace('_', ' ').title()}",
        projection="natural earth",
        scope="asia",
    )

    fig.update_geos(
        lataxis_range=[6, 37],
        lonaxis_range=[68, 98],
        showcountries=True,
        countrycolor="LightGray",
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

    return fig


def plot_time_series(
    df: pd.DataFrame,
    x_col: str = "year",
    y_col: str = "value",
    title: str | None = None,
) -> go.Figure:
    """Plot a simple time series line chart."""

    if x_col not in df.columns or y_col not in df.columns:
        raise ValueError("Dataframe must contain the specified x and y columns.")

    fig = px.line(df.sort_values(x_col), x=x_col, y=y_col, markers=True, title=title)
    fig.update_layout(xaxis_title=x_col.title(), yaxis_title=y_col.title())
    return fig


def plot_state_choropleth(
    df: pd.DataFrame,
    geojson_path: str,
    value_column: str = "overall_li",
    state_column: str = "statname",
    featureidkey: str = "properties.state",
    title: str | None = None,
) -> go.Figure:
    """Plot a choropleth map of states using a GeoJSON file."""

    import json

    with open(geojson_path, "r", encoding="utf-8") as f:
        geojson = json.load(f)

    # Normalize the state names for matching.
    df = df.copy()
    df[state_column] = df[state_column].astype(str).str.strip()

    if state_column not in df.columns:
        raise ValueError(f"DataFrame must contain {state_column}.")

    if value_column not in df.columns:
        raise ValueError(f"DataFrame must contain {value_column}.")

    fig = px.choropleth(
        df,
        geojson=geojson,
        locations=state_column,
        color=value_column,
        featureidkey=featureidkey,
        hover_name=state_column,
        title=title or f"Choropleth of {value_column}",
        color_continuous_scale="Viridis",
        labels={value_column: value_column.replace("_", " ")},
    )

    fig.update_geos(
        fitbounds="locations",
        visible=False,
    )
    fig.update_layout(margin=dict(l=10, r=10, t=40, b=10))

    return fig
