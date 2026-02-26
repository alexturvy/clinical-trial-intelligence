import json
from pathlib import Path

import streamlit as st
from utils.data_loader import load_trials, load_protocols
from utils.styles import inject_custom_css, hero_title, card_html

st.set_page_config(
    page_title="Clinical Trial Intelligence Platform",
    page_icon="\U0001F9EC",
    layout="wide",
    initial_sidebar_state="expanded",
)
inject_custom_css()

# --- Load data for dynamic stats ---
df = load_trials()
protocols = load_protocols()

DATA_DIR = Path(__file__).parent / "data"
with open(DATA_DIR / "model_results.json") as f:
    _model_results = json.load(f)
with open(DATA_DIR / "risk_model_results.json") as f:
    _risk_model = json.load(f)

# Compute dynamic stats from JSON
_struct_best = max(v["mean"] for v in _model_results["results"]["structured"].values())
_text_best = max(v["mean"] for v in _model_results["results"]["text_enriched"].values())
_improvement_ratio = _text_best / _struct_best if _struct_best > 0 else _text_best / 0.01

_effects = _risk_model["directional_effects"]
# For "mo/mention" display, only use count-based features (not binary flags)
_binary_features = {"performance_status_req", "contraception_req"}
_top_positive = max(
    ((k, v) for k, v in _effects.items()
     if k not in ("text_length", "word_count") and k not in _binary_features),
    key=lambda x: x[1],
)

total_trials = len(df)
ta_count = df["therapeutic_area"].nunique()
dur = df.dropna(subset=["duration_months"])
dur = dur[dur["duration_months"] > 0]
phase3_onc = dur[
    (dur["phase"].str.contains("Phase 3", na=False))
    & (dur["therapeutic_area"] == "Oncology")
]
onc_phase3_median = int(phase3_onc["duration_months"].median()) if len(phase3_onc) > 0 else 49

# --- Hero ---
hero_title(
    "Phase, enrollment, and disease area explain 4% of trial duration.",
    "NLP on eligibility criteria explains 6\u00D7 more.",
)

st.markdown(
    f"An investigation across **{total_trials:,}** clinical trials asking: what data actually "
    f"predicts trial duration? We tested three machine learning approaches on structured "
    f"registration fields versus language from eligibility criteria — and found the signal "
    f"lives in the text, not the metadata."
)

st.caption(
    "Python \u00B7 ClinicalTrials.gov API \u00B7 scikit-learn \u00B7 "
    "TF-IDF / TruncatedSVD \u00B7 Claude API \u00B7 Plotly \u00B7 Streamlit"
)

st.divider()

# --- Four Page Cards ---
c1, c2, c3, c4 = st.columns(4)

with c1:
    st.markdown(
        card_html(
            f"{total_trials:,} Trials, {ta_count} Therapeutic Areas",
            f"{total_trials:,}",
            "Volume, geography, phase distribution, and sponsors "
            "across the therapeutic areas where CROs operate",
            "ClinicalTrials.gov API \u2192 pandas",
        ),
        unsafe_allow_html=True,
    )
    st.page_link("pages/1_Trial_Landscape.py", label="Explore the landscape", icon="\U0001F30D")

with c2:
    st.markdown(
        card_html(
            f"Metadata Fails. Text Works {_improvement_ratio:.0f}\u00D7 Better.",
            f"R\u00B2 = {_text_best:.2f}",
            f"Three ML models on structured fields: R\u00B2 \u2248 {_struct_best:.2f}. "
            f"Add NLP features from eligibility criteria: R\u00B2 \u2248 {_text_best:.2f}",
            "scikit-learn \u2192 TF-IDF \u2192 cross-validation",
        ),
        unsafe_allow_html=True,
    )
    st.page_link("pages/2_Enrollment_Duration.py", label="See the investigation", icon="\U0001F4CA")

with c3:
    st.markdown(
        card_html(
            "16 Text Features That Predict Delay",
            f"+{_effects.get('performance_status_req', 3.0):.1f} mo",
            "Performance status requirements, prior therapy restrictions, "
            "genetic testing — each adds measurable months. Interactive risk profiler inside.",
            "NLP feature engineering \u2192 Claude API",
        ),
        unsafe_allow_html=True,
    )
    st.page_link("pages/3_Protocol_Complexity.py", label="Try the risk profiler", icon="\U0001F9EA")

with c4:
    st.markdown(
        card_html(
            "AI Opportunity Roadmap",
            "16 use cases",
            "The pattern — extracting structured signals from unstructured text — "
            "extends to costing, enrollment, amendments, and beyond",
            "Investigation \u2192 strategic roadmap",
        ),
        unsafe_allow_html=True,
    )
    st.page_link("pages/4_AI_Opportunity_Framework.py", label="View the roadmap", icon="\U0001F916")

# --- Data Source ---
st.divider()
st.markdown(
    '<p class="data-source">Data sourced from '
    '<a href="https://clinicaltrials.gov/">ClinicalTrials.gov</a>. '
    'Trial data reflects publicly registered studies and may not represent all clinical activity.</p>',
    unsafe_allow_html=True,
)
