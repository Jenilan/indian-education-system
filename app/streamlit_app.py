"""Streamlit dashboard for the Indian Education System project."""

from __future__ import annotations

import logging
import os
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Ensure the project root is on sys.path so `src` can be imported when running
# from the `app/` directory (e.g., `streamlit run app/streamlit_app.py`).
ROOT_DIR = Path(__file__).resolve().parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.insert(0, str(ROOT_DIR))

from src import analysis, data_loader, utils, viz


LOGGER = logging.getLogger(__name__)


def _load_local_dataset() -> tuple[dict[str, object] | None, pd.DataFrame | None]:
    """Attempt to load the local district-level dataset."""

    try:
        df = data_loader.load_local_district_data()
        return None, df
    except FileNotFoundError as exc:
        return {"error": str(exc)}, None


def _render_overview(df: pd.DataFrame) -> None:
    st.header("👁️ Overview")
    st.markdown(
        "Use the charts below to explore literacy rates by state and gender gaps across districts."
    )

    if "overall_li" in df.columns and "male_lit" in df.columns and "female_lit" in df.columns:
        fig_scatter = viz.interactive_scatter_literacy(df)
        st.plotly_chart(fig_scatter, use_container_width=True)

    if "statname" in df.columns and "overall_li" in df.columns:
        fig_bar = viz.plot_state_literacy_bar(df)
        st.plotly_chart(fig_bar, use_container_width=True)


def _render_analysis(df: pd.DataFrame) -> None:
    st.header("📊 Analysis")

    if "male_lit" in df.columns and "female_lit" in df.columns:
        st.subheader("Gender gap test")
        stats = analysis.gender_gap_ttest(df, male_column="male_lit", female_column="female_lit")
        st.metric("T-statistic", f"{stats['t_stat']:.3f}")
        st.metric("P-value", f"{stats['p_value']:.3g}")
        st.markdown(
            "This t-test compares distributions of male/female literacy rates. A low p-value suggests a significant difference."
        )

    st.subheader("Cluster districts")
    feature_cols = [c for c in ["overall_li", "male_lit", "female_lit"] if c in df.columns]
    if len(feature_cols) < 2:
        st.warning("Not enough features present in this dataset to run clustering.")
        return

    n_clusters = st.slider("Number of clusters", 2, 8, 3)
    cluster_df = analysis.kmeans_region_clustering(df, feature_cols, n_clusters=n_clusters)

    # Determine a safe district column name for display
    district_col = None
    for candidate in ["districts", "district", "distname", "dist_nm"]:
        if candidate in cluster_df.columns:
            district_col = candidate
            break

    display_cols = ["statname"]
    if district_col:
        display_cols.append(district_col)
    display_cols.append("cluster_label")

    st.dataframe(cluster_df[display_cols].head(12))
    st.markdown("Cluster centers (mean values):")
    st.dataframe(analysis.describe_cluster_centers(cluster_df, feature_cols))


def _render_downloads(df: pd.DataFrame) -> None:
    st.header("💾 Download & Export")

    csv_bytes = df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download cleaned dataset (CSV)",
        data=csv_bytes,
        file_name="indian_education_cleaned.csv",
        mime="text/csv",
    )

    st.markdown("---")
    st.subheader("Download charts")

    try:
        fig_bar = viz.plot_state_literacy_bar(df)
        bar_bytes = fig_bar.to_image(format="png")
        st.download_button(
            label="Download state literacy bar chart",
            data=bar_bytes,
            file_name="state_literacy.png",
            mime="image/png",
        )

        fig_scatter = viz.interactive_scatter_literacy(df)
        scatter_bytes = fig_scatter.to_image(format="png")
        st.download_button(
            label="Download gender gap scatter chart",
            data=scatter_bytes,
            file_name="gender_gap.png",
            mime="image/png",
        )
    except Exception:
        st.warning(
            "Chart export requires `kaleido` to be installed. Run `pip install kaleido` to enable downloads."
        )

    st.markdown("---")
    st.subheader("Generate PDF report")

    if st.button("Generate report"):
        from src.report import generate_pdf_report

        report_dir = utils.project_root() / "reports"
        report_dir.mkdir(exist_ok=True)

        report_path = report_dir / "education_report.pdf"
        narrative = (
            "This report summarizes key literacy statistics from the loaded dataset. "
            "It includes a state-level literacy ranking and gender gap analysis."
        )

        saved_figures: dict[str, str] = {}
        try:
            fig_bar = viz.plot_state_literacy_bar(df)
            fig_scatter = viz.interactive_scatter_literacy(df)

            bar_path = report_dir / "state_literacy.png"
            scatter_path = report_dir / "gender_gap.png"
            fig_bar.write_image(str(bar_path), scale=2)
            fig_scatter.write_image(str(scatter_path), scale=2)
            saved_figures = {
                "State literacy rankings": str(bar_path),
                "Gender literacy gap": str(scatter_path),
            }
        except Exception:
            st.warning(
                "Could not export chart images — make sure `kaleido` is installed so charts can be embedded in the PDF."
            )

        generate_pdf_report(
            output_path=str(report_path),
            title="Indian Education System - Summary Report",
            narrative=narrative,
            figure_paths=saved_figures,
        )

        st.success(f"Report created: {report_path}")


def _render_about() -> None:
    st.header("ℹ️ About")
    st.markdown(
        """
        **Indian Education System Explorer** is designed to provide actionable insights into
        India’s education landscape.

        ### Key features
        - Local district-level analysis from CSV datasets.
        - Live indicator trends fetched from the World Bank API.
        - Clustering and statistical testing to reveal patterns.
        - Downloadable charts, datasets, and PDF reports.

        ### Data sources
        - *Local*: District-level CSV (e.g., education department datasets).
        - *Global*: World Bank API indicators.

        ### Notes on ethics
        - Data quality varies by region; underreporting is common in some areas.
        - Avoid making policy decisions without considering the context behind the numbers.
        """
    )


def main() -> None:
    st.set_page_config(
        page_title="Indian Education System Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📚 Indian Education System Explorer")

    page = st.sidebar.radio(
        "Page", ["Overview", "Analysis", "Download", "About"]
    )

    source = st.sidebar.radio(
        "Data Source", ["Local CSV (district-level)", "World Bank (country-level)"]
    )

    df = None
    error = None

    if source.startswith("Local"):
        st.sidebar.markdown(
            "**Local dataset requirements**: Place a CSV file named `2015_16_Districtwise.csv` in the `data/` directory."
        )
        error, df = _load_local_dataset()
        if error:
            st.sidebar.error(error["error"])

    if page == "Overview":
        if df is None:
            st.warning("Please load a local dataset to view the overview.")
        else:
            _render_overview(df)

    elif page == "Analysis":
        if df is None:
            st.warning("Please load a local dataset to run the analysis.")
        else:
            _render_analysis(df)

    elif page == "Download":
        if df is None:
            st.warning("Please load a local dataset to access downloads.")
        else:
            _render_downloads(df)

    else:
        _render_about()


if __name__ == "__main__":
    main()
