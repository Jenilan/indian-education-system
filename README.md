# Indian Education System Explorer

[![Deploy on Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io/)

> **One-click deploy:** Click the badge above, choose **"New app"**, and link this repo to deploy the dashboard instantly.

A structured Python project that analyzes and visualizes key education metrics for India.

This repository is organized around the **Data Story** framework: **Collect → Clean → Analyze → Visualize → Communicate**. It includes:

- ✅ **Data sourcing** via local CSV files or live World Bank API metrics
- ✅ **Modular code** in `src/` for loading, cleaning, modeling, and visualizing
- ✅ **Interactive dashboard** using Streamlit (`app/streamlit_app.py`)
- ✅ **Statistical/ML analysis** (regression, clustering) using scikit-learn
- ✅ **Reusable notebooks** for exploratory analysis (`notebooks/`)
- ✅ **Tests** with `pytest` (`tests/`)
- ✅ **CI workflow** using GitHub Actions (`.github/workflows/ci.yml`)


## Project Structure

- `src/` - Core modules for data loading, cleaning, analysis, and visualization
- `data/` - Place datasets here (e.g., `2015_16_Districtwise.csv`)
- `app/` - Streamlit dashboard
- `notebooks/` - Exploratory notebooks
- `tests/` - Unit tests


## Quick Start

### Deploy to Streamlit Cloud (one-click)

If you'd like to host this dashboard publicly, create a free Streamlit Cloud account and connect your GitHub repo. Then: 1) add your dataset file to the repo (or use the World Bank mode), 2) set the main file to `app/streamlit_app.py`, and 3) press **Deploy**.

> Tip: If you don't want to commit the raw CSV, you can keep your dataset locally and use the World Bank API mode in the app.


1. **Install dependencies**

```bash
python -m pip install -r requirements.txt
```

2. (Optional) **Install dev tooling**

```bash
python -m pip install -r dev-requirements.txt
```

3. (Optional) **Install and enable pre-commit hooks**

```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```

4. (Optional) **Enable auto deployment to Streamlit Cloud**

1. Add the following GitHub Secrets to your repo:
   - `STREAMLIT_CLOUD_TOKEN`
   - `STREAMLIT_APP_ID`
2. The pipeline will run on push to `main` and attempt to deploy using the Streamlit Cloud API.


```bash
python -m pip install pre-commit
pre-commit install
pre-commit run --all-files
```
3. **Add your dataset**

Place your district-level CSV in `data/2015_16_Districtwise.csv`.

4. **Run the Streamlit dashboard**

```bash
streamlit run app/streamlit_app.py
```

5. **Run the CLI preview**

```bash
python main.py --local-data data/2015_16_Districtwise.csv
python main.py --fetch-world-bank
```


## What’s Inside

### Data Sources
- **Local CSV**: District-level dataset (e.g., literacy, enrollment). The project currently uses a placeholder dataset but is designed to work with real government or Kaggle files.
- **World Bank API**: Live fetching of indicators like enrollment rates and adult literacy.

### Analyses Included
- **Exploratory Data Analysis (EDA)**: correlations, distributions, and state-level comparisons.
- **Machine Learning**: regression (Linear, Ridge, Random Forest) and clustering (KMeans) to find education patterns.
- **Dashboard**: Interactive charts and charts generated with Plotly.
- **Reporting (future)**: Export model summaries and findings to PDF.


## Project Evolution / Resume Boosters

- ✅ Added **API integration** and fallback to local data
- ✅ Added **Streamlit interactive dashboard** (deployable to Streamlit Cloud)
- ✅ Added **statistical analysis and clustering** for actionable insights
- ✅ Structured as a **modular package** for reuse
- ✅ Added **testing** and **documentation** for maintainability


## Notes on Data Ethics & Accessibility

- Use color-blind friendly palettes in plots.
- Be mindful of biases in datasets (e.g., under-reporting in rural regions).
- Store any sensitive or personally identifiable data securely and anonymize when possible.


---

If you want to level this project up further, consider:
- Adding **state-level mapping** (GeoJSON) and choropleth maps
- Building a **PDF report generator** (ReportLab) summarizing key insights
- Adding **automated data validation tests** for new uploads
