import streamlit as st
import pandas as pd
from pathlib import Path

DATA_DIR = Path(__file__).parent.parent / "data"


APP_COLUMNS = [
    "nct_id", "title", "phase", "status", "enrollment",
    "start_date", "completion_date", "primary_completion_date",
    "sponsor", "sponsor_type", "conditions", "countries",
    "therapeutic_area", "duration_months",
]


@st.cache_data(ttl=3600)
def load_trials() -> pd.DataFrame:
    """Load pre-processed clinical trials dataset (excluding large text columns)."""
    path = DATA_DIR / "trials_processed.parquet"
    available = pd.read_parquet(path, columns=None).columns.tolist()
    cols = [c for c in APP_COLUMNS if c in available]
    df = pd.read_parquet(path, columns=cols)
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "completion_date" in df.columns:
        df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    return df


@st.cache_data(ttl=3600)
def load_trial_text() -> pd.DataFrame:
    """Load nct_id + eligibility_criteria from the full dataset (for text feature extraction)."""
    path = DATA_DIR / "trials_processed.parquet"
    return pd.read_parquet(path, columns=["nct_id", "eligibility_criteria"])


@st.cache_data(ttl=3600)
def load_protocols() -> pd.DataFrame | None:
    """Load pre-scored protocol complexity data, recovering therapeutic_area if missing."""
    path = DATA_DIR / "protocols_scored.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if len(df) == 0:
        return None

    # Fix: score_protocols.py groupby(include_groups=False) dropped therapeutic_area
    # for most rows. Recover from the main dataset by nct_id join.
    if "therapeutic_area" not in df.columns or df["therapeutic_area"].isna().sum() > len(df) * 0.5:
        trials_path = DATA_DIR / "trials_processed.parquet"
        if trials_path.exists():
            trials_ta = pd.read_parquet(trials_path, columns=["nct_id", "therapeutic_area"])
            df = df.drop(columns=["therapeutic_area"], errors="ignore")
            df = df.merge(trials_ta, on="nct_id", how="left")

    return df


def get_therapeutic_areas(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique therapeutic areas."""
    if "therapeutic_area" in df.columns:
        return sorted(df["therapeutic_area"].dropna().unique().tolist())
    return []


def get_phases(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique phases."""
    if "phase" in df.columns:
        return sorted(df["phase"].dropna().unique().tolist())
    return []


def get_statuses(df: pd.DataFrame) -> list[str]:
    """Return sorted list of unique study statuses."""
    if "status" in df.columns:
        return sorted(df["status"].dropna().unique().tolist())
    return []


@st.cache_data(ttl=3600)
def get_sample_trials() -> pd.DataFrame | None:
    """Return NCT ID, title, therapeutic area, and phase for scored protocols (for dropdown picker)."""
    protocols = load_protocols()
    if protocols is None:
        return None
    cols = ["nct_id", "title", "therapeutic_area", "phase", "complexity_score"]
    available = [c for c in cols if c in protocols.columns]
    return protocols[available].copy()


@st.cache_data(ttl=3600)
def load_trial_criteria(nct_id: str) -> str | None:
    """Load eligibility criteria text for a single trial from the scored protocols dataset."""
    protocols = load_protocols()
    if protocols is not None and "eligibility_criteria" in protocols.columns:
        match = protocols[protocols["nct_id"] == nct_id]
        if len(match) > 0:
            return match.iloc[0]["eligibility_criteria"]
    # Fall back to main dataset
    path = DATA_DIR / "trials_processed.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path, columns=["nct_id", "eligibility_criteria"])
    match = df[df["nct_id"] == nct_id]
    if len(match) > 0:
        return match.iloc[0]["eligibility_criteria"]
    return None
