import streamlit as st
import pandas as pd
from utils.data_loader import load_trials, get_therapeutic_areas, get_phases, get_statuses
from utils.charts import (
    trial_volume_trend, phase_distribution, geographic_map,
    enrollment_histogram, sponsor_treemap,
)
from utils.styles import inject_custom_css, section_label, byline

st.set_page_config(page_title="Trial Landscape", page_icon="\U0001F30D", layout="wide")
inject_custom_css()
byline()
section_label("Exploratory Data Analysis")
st.title("Trial Landscape Across 7 Therapeutic Areas")

# Load data
df = load_trials()

# --- Hero Insight ---
total = len(df)
ta_count = df["therapeutic_area"].nunique()
dur = df.dropna(subset=["duration_months"])
dur = dur[dur["duration_months"] > 0]
ta_medians = dur.groupby("therapeutic_area")["duration_months"].median().sort_values(ascending=False)
longest_ta = ta_medians.index[0] if len(ta_medians) > 0 else "Oncology"
longest_median = int(ta_medians.iloc[0]) if len(ta_medians) > 0 else 35
overall_median = int(dur["duration_months"].median()) if len(dur) > 0 else 20

onc_count = len(df[df["therapeutic_area"] == "Oncology"])
onc_pct = int(100 * onc_count / total) if total > 0 else 16

st.markdown(
    f"> **{total:,} trials across {ta_count} therapeutic areas** — "
    f"{longest_ta} leads in trial duration at a median {longest_median} months, "
    f"versus {overall_median} months across all areas."
)

st.markdown(
    f"**Methods:** `ClinicalTrials.gov API v2` \u2192 `pandas` data pipeline \u2192 "
    f"therapeutic area classification \u2192 duration computation \u2192 {total:,} trials "
    f"across {ta_count} areas. This page is exploratory data analysis — the machine "
    f"learning investigation starts on the next page."
)

# --- Sidebar Filters ---
st.sidebar.header("Filters")

ta_options = get_therapeutic_areas(df)
selected_ta = st.sidebar.multiselect("Therapeutic Area", ta_options, default=ta_options)

phase_options = get_phases(df)
selected_phases = st.sidebar.multiselect("Phase", phase_options, default=phase_options)

status_options = get_statuses(df)
selected_statuses = st.sidebar.multiselect("Study Status", status_options, default=status_options)

sponsor_types = sorted(df["sponsor_type"].dropna().unique().tolist())
selected_sponsor_types = st.sidebar.multiselect("Sponsor Type", sponsor_types, default=sponsor_types)

# Date range
min_date = df["start_date"].min()
max_date = df["start_date"].max()
if pd.notna(min_date) and pd.notna(max_date):
    date_range = st.sidebar.slider(
        "Start Date Range",
        min_value=min_date.year,
        max_value=max_date.year,
        value=(min_date.year, max_date.year),
    )
else:
    date_range = (2020, 2026)

# --- Apply Filters ---
mask = (
    df["therapeutic_area"].isin(selected_ta)
    & df["phase"].isin(selected_phases)
    & df["status"].isin(selected_statuses)
    & df["sponsor_type"].isin(selected_sponsor_types)
)

date_mask = df["start_date"].notna() & (df["start_date"].dt.year >= date_range[0]) & (df["start_date"].dt.year <= date_range[1])
mask = mask & (date_mask | df["start_date"].isna())

filtered = df[mask]

# --- KPI Cards ---
st.divider()
k1, k2, k3, k4 = st.columns(4)
k1.metric("Total Trials", f"{len(filtered):,}")
k2.metric("Median Enrollment", f"{int(filtered['enrollment'].median()):,}" if filtered['enrollment'].notna().any() else "N/A")

recruiting = filtered[filtered["status"].str.contains("Recruit", case=False, na=False)]
k3.metric("Actively Recruiting", f"{len(recruiting):,}")

top_sponsor = filtered["sponsor"].value_counts().index[0] if len(filtered) > 0 else "N/A"
top_sponsor_display = top_sponsor[:30] + "..." if len(str(top_sponsor)) > 30 else top_sponsor
k4.metric("Top Sponsor", top_sponsor_display)

# --- Primary Charts: Volume Trend + Phase Distribution ---
st.divider()

c1, c2 = st.columns([3, 2])
with c1:
    st.plotly_chart(trial_volume_trend(filtered), use_container_width=True)
with c2:
    phase_fig, na_count = phase_distribution(filtered)
    st.plotly_chart(phase_fig, use_container_width=True)
    st.caption(f"Excludes {na_count:,} observational / non-phased studies")

# --- Enrollment Histogram ---
st.plotly_chart(enrollment_histogram(filtered), use_container_width=True)

# --- Secondary: Geography & Sponsors ---
with st.expander("Explore Geography & Sponsors"):
    st.plotly_chart(geographic_map(filtered), use_container_width=True)
    st.plotly_chart(sponsor_treemap(filtered), use_container_width=True)

# --- Data Table ---
with st.expander("View Raw Data"):
    display_cols = ["nct_id", "title", "therapeutic_area", "phase", "status",
                    "enrollment", "start_date", "sponsor", "sponsor_type"]
    st.dataframe(filtered[display_cols].head(500), use_container_width=True, hide_index=True)

# --- Connector ---
st.divider()
col_left, col_right = st.columns([4, 1])
with col_left:
    st.markdown(
        f"**Everyone in CROs knows {longest_ta} takes the longest.** "
        "The harder question: can you predict how long a trial will take "
        "*before it starts* — and what data actually helps?"
    )
with col_right:
    st.page_link(
        "pages/2_Enrollment_Duration.py",
        label="See the investigation \u2192",
        icon="\U0001F4CA",
    )
