"""Data loading utilities for the Indian Education System project.

This module provides functions to load local datasets and optionally fetch
publicly available education indicators from the World Bank API.

The code is written to be robust in environments where internet access may not
be available, falling back to local CSV assets when necessary.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests

from .utils import data_dir, safe_read_csv


LOGGER = logging.getLogger(__name__)


def _clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Make column names python-friendly (snake_case)."""

    df = df.copy()
    df.columns = (
        df.columns.str.strip()
        .str.lower()
        .str.replace(" ", "_", regex=False)
        .str.replace(r"[^a-z0-9_]+", "", regex=True)
    )
    return df


def _read_table(path: Path) -> pd.DataFrame:
    """Read a tabular dataset from CSV/XLSX/Parquet with safe defaults."""

    if not path.exists():
        raise FileNotFoundError(f"Could not find dataset at {path}")

    suffix = path.suffix.lower()
    if suffix in {".csv", ".txt"}:
        # Encoding in public datasets is inconsistent; try a couple common ones.
        last_exc: Exception | None = None
        for encoding in ("utf-8", "utf-8-sig", "cp1252", "latin-1"):
            try:
                return safe_read_csv(
                    str(path),
                    encoding=encoding,
                    low_memory=False,
                )
            except Exception as exc:  # pragma: no cover (depends on local file)
                last_exc = exc
        assert last_exc is not None
        raise last_exc

    if suffix in {".xlsx", ".xls"}:
        # Excel files are frequently misnamed in the wild:
        # - sometimes an .xlsx is saved as .xls
        # - sometimes a CSV is saved as .xls
        # We sniff the first chunk of bytes to decide how to read it.
        header = b""
        head_chunk = b""
        try:
            with path.open("rb") as f:
                header = f.read(8)
                head_chunk = header + f.read(2048)
        except Exception:  # pragma: no cover
            header = b""
            head_chunk = b""

        # XLSX files are ZIP containers starting with PK.
        looks_like_xlsx = header.startswith(b"PK")

        # CSV often starts with UTF-8 BOM + column text, and contains commas/newlines early.
        # Example: b"\xef\xbb\xbfAC_YEAR,DISTCD,..."
        looks_like_text_csv = False
        if head_chunk:
            sample = head_chunk.lstrip(b"\xef\xbb\xbf").strip()
            # Heuristic: lots of printable bytes and at least one comma + newline soon.
            printable = sum(32 <= b <= 126 or b in (9, 10, 13) for b in sample[:512])
            if printable / max(1, len(sample[:512])) > 0.9 and (b"," in sample[:512]) and (b"\n" in sample[:2048] or b"\r" in sample[:2048]):
                looks_like_text_csv = True

        if looks_like_text_csv:
            return safe_read_csv(str(path), encoding="utf-8-sig", low_memory=False)

        try:
            if suffix == ".xlsx" or looks_like_xlsx:
                return pd.read_excel(path, engine="openpyxl")
            # .xls (legacy format) requires xlrd
            return pd.read_excel(path, engine="xlrd")
        except ImportError as exc:  # pragma: no cover
            if suffix == ".xlsx":
                raise ImportError(
                    "Reading .xlsx files requires `openpyxl`. Install it with: pip install openpyxl"
                ) from exc
            raise ImportError(
                "Reading .xls files requires `xlrd`. Install it with: pip install xlrd"
            ) from exc
        except ValueError as exc:  # pragma: no cover
            # If the extension says xls but content looks like xlsx, try openpyxl as fallback.
            msg = str(exc).lower()
            if suffix == ".xls" and ("bof record" in msg or "unsupported format" in msg or "corrupt" in msg):
                try:
                    return pd.read_excel(path, engine="openpyxl")
                except Exception:
                    pass
            raise

    if suffix in {".parquet"}:
        try:
            return pd.read_parquet(path)
        except ImportError as exc:  # pragma: no cover
            raise ImportError(
                "Reading Parquet files requires `pyarrow` or `fastparquet`. "
                "Install one with: pip install pyarrow"
            ) from exc

    raise ValueError(
        f"Unsupported dataset format: {suffix}. Supported: .csv, .xlsx, .parquet"
    )


def normalize_dataset_path(path: str | Path | None) -> Path | None:
    """Normalize a user-provided dataset path.

    Streamlit text inputs often include accidental surrounding quotes. This helper
    strips common wrappers and expands ~.
    """

    if path is None:
        return None
    if isinstance(path, Path):
        return path

    p = str(path).strip()
    if not p:
        return None

    # Strip surrounding quotes: "C:\..." or 'C:\...'
    if (p.startswith('"') and p.endswith('"')) or (p.startswith("'") and p.endswith("'")):
        p = p[1:-1].strip()

    # Remove stray leading/trailing quotes (common copy/paste)
    p = p.strip('"').strip("'").strip()
    if not p:
        return None

    return Path(p).expanduser()


def _first_present(df: pd.DataFrame, candidates: list[str]) -> str | None:
    """Return first matching column name from candidates, else None."""

    for c in candidates:
        if c in df.columns:
            return c
    return None


def _coerce_percent(s: pd.Series) -> pd.Series:
    """Convert a numeric series to percent scale when values look like proportions."""

    if s.empty:
        return s
    out = pd.to_numeric(s, errors="coerce")
    finite = out.dropna()
    if finite.empty:
        return out
    # Heuristic: if most values are in [0, 1.2], treat as proportions.
    q95 = float(finite.quantile(0.95))
    if 0 <= q95 <= 1.2:
        return out * 100.0
    return out


def standardize_district_schema(
    df: pd.DataFrame, column_map: dict[str, str] | None = None, dataset_year: str | int | None = None
) -> pd.DataFrame:
    """Normalize arbitrary district-wise datasets into the app's expected schema.

    Output columns (when available):
    - statname: state/UT name
    - district: district name
    - overall_li: overall literacy rate (%)
    - male_lit: male literacy rate (%)
    - female_lit: female literacy rate (%)
    - year: dataset year (optional)
    """

    df = _clean_columns(df)
    df = df.copy()

    # Optional explicit mapping: {"statname": "state_name", "district": "district_name", ...}
    if column_map:
        inv = {v: k for k, v in column_map.items()}
        rename: dict[str, str] = {}
        for src_col, dst_col in inv.items():
            if src_col in df.columns:
                rename[src_col] = dst_col
        if rename:
            df = df.rename(columns=rename)

    # Auto-detect common variants if still missing.
    if "statname" not in df.columns:
        c = _first_present(
            df,
            [
                "state",
                "state_name",
                "stateut",
                "state_ut",
                "stateunionterritory",
                "state_union_territory",
                "statename",
                "ut_name",
                "name_of_stateut",
            ],
        )
        if c:
            df = df.rename(columns={c: "statname"})

    if "district" not in df.columns:
        c = _first_present(
            df,
            [
                "district_name",
                "districts",
                "distname",
                "dist_nm",
                "districtnm",
                "district",
                "name_of_district",
            ],
        )
        if c:
            df = df.rename(columns={c: "district"})

    if "overall_li" not in df.columns:
        c = _first_present(
            df,
            [
                "overall_literacy",
                "overall_literacy_rate",
                "literacy_rate",
                "literacy_rate_total",
                "total_literacy",
                "overall_li",
                "overall_lit",
                "literacy_total",
            ],
        )
        if c:
            df = df.rename(columns={c: "overall_li"})

    if "male_lit" not in df.columns:
        c = _first_present(
            df,
            [
                "male_literacy",
                "male_literacy_rate",
                "literacy_rate_male",
                "male_lit",
                "m_literacy",
            ],
        )
        if c:
            df = df.rename(columns={c: "male_lit"})

    if "female_lit" not in df.columns:
        c = _first_present(
            df,
            [
                "female_literacy",
                "female_literacy_rate",
                "literacy_rate_female",
                "female_lit",
                "f_literacy",
            ],
        )
        if c:
            df = df.rename(columns={c: "female_lit"})

    # Normalize types and percent scaling.
    for col in ("overall_li", "male_lit", "female_lit"):
        if col in df.columns:
            df[col] = _coerce_percent(df[col])

    if "overall_li" not in df.columns and {"male_lit", "female_lit"}.issubset(df.columns):
        df["overall_li"] = (pd.to_numeric(df["male_lit"], errors="coerce") + pd.to_numeric(df["female_lit"], errors="coerce")) / 2.0

    if dataset_year is not None and "year" not in df.columns:
        df["year"] = dataset_year

    return df


def load_local_district_data(path: Optional[str] = None) -> pd.DataFrame:
    """Load a local district-level dataset.

    Parameters
    ----------
    path:
        Path to a CSV file. If None, will use `data/2015_16_Districtwise.csv` in the
        project data folder.

    Returns
    -------
    pd.DataFrame
        The loaded and cleaned dataset.
    """

    default_path = data_dir() / "2015_16_Districtwise.csv"
    csv_path = Path(path) if path else default_path

    df = _read_table(csv_path)
    return standardize_district_schema(df)


def clean_local_district_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean a district-level dataframe loaded from an arbitrary source."""

    return standardize_district_schema(df)


def load_district_dataset(
    path: str | Path | None,
    *,
    dataset_year: str | int | None = None,
    column_map: dict[str, str] | None = None,
) -> pd.DataFrame:
    """Load a district-wise dataset from disk and standardize its schema.

    This is the recommended entry point for custom datasets (including 2025–2026).
    """

    if path is None:
        return load_local_district_data(None)
    p = normalize_dataset_path(path)
    if p is None:
        return load_local_district_data(None)
    df = _read_table(p)
    return standardize_district_schema(df, column_map=column_map, dataset_year=dataset_year)


def fetch_world_bank_indicator(
    indicator: str,
    country_code: str = "IND",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
    per_page: int = 1000,
) -> pd.DataFrame:
    """Fetch a single indicator from the World Bank API.

    Parameters
    ----------
    indicator:
        The World Bank indicator code (e.g., "SE.PRM.ENRR" for primary school
        enrollment).
    country_code:
        Country ISO code (default: "IND").
    start_year, end_year:
        Optional year filters to limit the result set.

    Returns
    -------
    pandas.DataFrame
        A time series of the indicator.

    Notes
    -----
    The World Bank API returns results in pages; this function will fetch all pages.
    """

    url = "https://api.worldbank.org/v2/country/{country}/indicator/{indicator}".format(
        country=country_code, indicator=indicator
    )

    params: Dict[str, object] = {
        "format": "json",
        "per_page": per_page,
    }
    if start_year:
        params["date"] = f"{start_year}:{end_year or ''}".strip(":")

    response = requests.get(url, params=params, timeout=30)
    response.raise_for_status()

    data = response.json()
    # data[0] is metadata, data[1] is list of observations
    if not data or len(data) < 2:
        raise ValueError("Unexpected response from World Bank API.")

    rows = data[1]
    df = pd.json_normalize(rows)
    # Keep only the fields the rest of the pipeline expects.
    if "date" in df.columns:
        df["year"] = df["date"].astype(int)
    df = df.rename(columns={"value": indicator, "country.value": "country"})
    return df[["country", "year", indicator]]


WORLD_BANK_INDICATORS: dict[str, str] = {
    "SE.ADT.LITR.ZS": "Adult literacy rate, population 15+ years (% of people ages 15 and above)",
    "SE.PRM.ENRR": "School enrollment, primary (% gross)",
    "SE.SEC.ENRR": "School enrollment, secondary (% gross)",
    "SE.TER.ENRR": "School enrollment, tertiary (% gross)",
    "SE.PRM.CMPT.ZS": "Primary completion rate (% of relevant age group)",
    "SE.PRM.ENRL": "School enrollment, primary (number)",
}


def fetch_world_bank_indicators(
    indicators: List[str],
    country_code: str = "IND",
    start_year: Optional[int] = None,
    end_year: Optional[int] = None,
) -> pd.DataFrame:
    """Fetch multiple World Bank indicators and join them into a single dataframe."""

    dfs = []
    for indicator in indicators:
        LOGGER.debug("Fetching indicator %s", indicator)
        df_i = fetch_world_bank_indicator(
            indicator=indicator,
            country_code=country_code,
            start_year=start_year,
            end_year=end_year,
        )
        dfs.append(df_i)

    merged = dfs[0]
    for df_i in dfs[1:]:
        merged = merged.merge(df_i, on=["country", "year"], how="outer")

    return merged
