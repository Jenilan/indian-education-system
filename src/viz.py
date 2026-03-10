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

    fig = px.scatter(
        df,
        x=x_col,
        y=y_col,
        color="statname" if "statname" in df.columns else None,
        hover_data=["districts"] if "districts" in df.columns else None,
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
