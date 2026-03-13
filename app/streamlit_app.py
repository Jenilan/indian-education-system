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

from src import analysis, data_loader, geodata, utils, viz


LOGGER = logging.getLogger(__name__)


@st.cache_data(show_spinner=False)
def _load_local_dataset(path: str | None = None) -> tuple[dict[str, object] | None, pd.DataFrame | None]:
    """Attempt to load the local district-level dataset."""

    try:
        df = data_loader.load_district_dataset(path)
        return None, df
    except FileNotFoundError as exc:
        return {"error": str(exc)}, None
    except Exception as exc:
        return {"error": f"Failed to load dataset: {exc}"}, None


@st.cache_data(show_spinner=False)
def _load_uploaded_dataset(uploaded_file) -> pd.DataFrame:
    """Load and clean a CSV uploaded via Streamlit."""

    df = pd.read_csv(uploaded_file)
    return data_loader.clean_local_district_df(df)


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

    if "statname" in df.columns:
        st.subheader("State-level map")
        df_with_geo = geodata.add_state_coordinates(df, state_column="statname")

        metric_options = [
            c
            for c in ["overall_li", "male_lit", "female_lit", "p_sc_pop", "p_st_pop"]
            if c in df_with_geo.columns
        ]
        if metric_options:
            metric = st.selectbox("Choose metric for map", metric_options, index=0)
            try:
                fig_map = viz.plot_state_bubble_map(
                    df_with_geo, value_column=metric, state_column="statname"
                )
                st.plotly_chart(fig_map, use_container_width=True)
            except ValueError as exc:
                st.warning(str(exc))
        else:
            st.info("Map requires state-level numeric metrics (e.g., literacy or population).")


def _render_analysis(df: pd.DataFrame) -> None:
    st.header("📊 Analysis")

    if "statname" in df.columns:
        st.subheader("State drilldown")
        states = sorted(df["statname"].dropna().unique().tolist())
        state_choice = st.selectbox("Select a state", ["All"] + states, index=0)

        if state_choice != "All":
            state_df = df[df["statname"] == state_choice]
            st.markdown(f"### {state_choice} — District-level snapshot")
            st.write(
                "Showing district-level literacy and enrollment statistics for the selected state."
            )
            if "overall_li" in state_df.columns:
                top_n = st.slider("Show top N districts by overall literacy", 3, 20, 10)
                top_districts = (
                    state_df.sort_values("overall_li", ascending=False)
                    .head(top_n)[[c for c in ["district", "distname", "DISTRICT", "overall_li"] if c in state_df.columns]]
                )
                st.dataframe(top_districts.reset_index(drop=True))

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

        # Create a summary table of literacy by state to include in the report.
        state_summary = None
        if "statname" in df.columns and "overall_li" in df.columns:
            state_summary = (
                df.groupby("statname")["overall_li"]
                .mean()
                .reset_index()
                .sort_values("overall_li", ascending=False)
                .rename(columns={"overall_li": "avg_literacy"})
            )

        generate_pdf_report(
            output_path=str(report_path),
            title="Indian Education System - Summary Report",
            narrative=narrative,
            figure_paths=saved_figures,
            tables={"Top states by average literacy": state_summary} if state_summary is not None else None,
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


def _render_map(df: pd.DataFrame) -> None:
    st.header("🗺️ Map Explorer")
    st.markdown(
        "Use map visualizations to compare state-level performance across indicators. "
        "Switch between bubbles and choropleth to find insights quickly."
    )

    if "statname" not in df.columns:
        st.warning("Map view requires a `statname` column in the dataset.")
        return

    df_geo = geodata.add_state_coordinates(df, state_column="statname")
    metric_options = [
        c
        for c in ["overall_li", "male_lit", "female_lit", "p_sc_pop", "p_st_pop"]
        if c in df_geo.columns
    ]
    if not metric_options:
        st.warning("No supported numeric columns found for mapping.")
        return

    map_type = st.radio("Map type", ["Bubble", "Choropleth"], horizontal=True)
    metric = st.selectbox("Metric to map", metric_options, index=0)

    if map_type == "Bubble":
        try:
            fig = viz.plot_state_bubble_map(
                df_geo, value_column=metric, state_column="statname"
            )
            st.plotly_chart(fig, use_container_width=True)
        except ValueError as ex:
            st.warning(str(ex))

    else:
        geojson_path = utils.resource_path("data", "india_states.geojson")
        if not geojson_path.exists():
            st.error(
                "Choropleth map requires a GeoJSON file at `data/india_states.geojson`. "
                "Please ensure the file exists in the project folder."
            )
            return

        try:
            fig = viz.plot_state_choropleth(
                df_geo,
                geojson_path=str(geojson_path),
                value_column=metric,
                state_column="statname",
                title=f"State-level {metric.replace('_', ' ').title()} (Choropleth)",
            )
            st.plotly_chart(fig, use_container_width=True)
        except Exception as ex:
            st.warning(f"Could not render choropleth map: {ex}")


def _render_world_bank() -> None:
    st.header("🌍 World Bank Indicator Explorer")
    st.markdown(
        "Explore time-series indicators for India from the World Bank. "
        "Choose an indicator and the year range to visualize trends."
    )

    indicator = st.selectbox(
        "Indicator",
        options=list(data_loader.WORLD_BANK_INDICATORS.keys()),
        format_func=lambda k: f"{k} — {data_loader.WORLD_BANK_INDICATORS[k]}",
    )

    current_year = pd.Timestamp.now().year
    start_year, end_year = st.slider(
        "Year range", 1990, current_year, (2000, min(current_year, 2022))
    )

    try:
        with st.spinner("Fetching World Bank data..."):
            wb_df = data_loader.fetch_world_bank_indicator(
                indicator, start_year=start_year, end_year=end_year
            )

        if wb_df.empty:
            st.warning("No data returned for this indicator/time period.")
            return

        st.dataframe(wb_df)

        fig = viz.plot_time_series(
            wb_df,
            x_col="year",
            y_col=indicator,
            title=f"{data_loader.WORLD_BANK_INDICATORS[indicator]} ({indicator})",
        )
        st.plotly_chart(fig, use_container_width=True)

        if st.checkbox("Add simple linear forecast", value=False):
            forecast_df = analysis.forecast_time_series(
                wb_df, year_column="year", value_column=indicator, periods=5
            )
            fig_forecast = viz.plot_time_series(
                forecast_df,
                x_col="year",
                y_col=indicator,
                title=f"{data_loader.WORLD_BANK_INDICATORS[indicator]} + Forecast",
            )
            st.plotly_chart(fig_forecast, use_container_width=True)

        csv_bytes = wb_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download indicator data (CSV)",
            data=csv_bytes,
            file_name=f"{indicator}_{start_year}_{end_year}.csv",
            mime="text/csv",
        )

    except Exception as exc:
        st.error(f"Unable to fetch World Bank data: {exc}")


def main() -> None:
    st.set_page_config(
        page_title="Indian Education System Explorer",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    st.title("📚 Indian Education System Explorer")

    page = st.sidebar.radio(
        "Page",
        [
            "Overview",
            "Map",
            "Analysis",
            "World Bank",
            "Download",
            "About",
        ],
    )

    source = st.sidebar.radio(
        "Data Source", ["Local CSV (district-level)", "World Bank (country-level)"]
    )

    df = None
    error = None
    uploaded = None

    if source.startswith("Local"):
        st.sidebar.markdown(
            "**Local dataset**: upload a CSV, or provide a local file path (best for very large files). "
            "Supported: `.csv`, `.xlsx`, `.parquet`."
        )

        default_path = os.environ.get("EDU_LOCAL_DATA_PATH", "")
        data_dir = utils.data_dir()
        if data_dir.exists():
            local_options = sorted(
                [str(p) for p in data_dir.glob("*.csv")]
                + [str(p) for p in data_dir.glob("*.xlsx")]
                + [str(p) for p in data_dir.glob("*.xls")]
                + [str(p) for p in data_dir.glob("*.parquet")]
            )
        else:
            local_options = []

        path_choice = None
        if local_options:
            path_choice = st.sidebar.selectbox(
                "Or choose a dataset from `data/`",
                options=[""] + local_options,
                index=0,
            )

        path_input = st.sidebar.text_input(
            "Or paste full dataset path",
            value=default_path,
            help="Tip: set env var `EDU_LOCAL_DATA_PATH` to persist this. Quotes are OK; they’ll be stripped.",
        ).strip()

        uploaded = st.sidebar.file_uploader(
            "Or upload a district-level CSV (smaller files)", type=["csv"]
        )

        if uploaded is not None:
            df = _load_uploaded_dataset(uploaded)
        else:
            resolved_path = path_input or (path_choice if path_choice else None)
            error, df = _load_local_dataset(resolved_path)
            if error:
                st.sidebar.error(error["error"])

    if page == "Overview":
        if df is None:
            st.warning("Please load a local dataset to view the overview.")
        else:
            _render_overview(df)

    elif page == "Map":
        if df is None:
            st.warning("Please load a local dataset to view the map.")
        else:
            _render_map(df)

    elif page == "Analysis":
        if df is None:
            st.warning("Please load a local dataset to run the analysis.")
        else:
            _render_analysis(df)

    elif page == "World Bank":
        _render_world_bank()

    elif page == "Download":
        if df is None:
            st.warning("Please load a local dataset to access downloads.")
        else:
            _render_downloads(df)

    else:
        _render_about()


if __name__ == "__main__":
    main()
