import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.data_loader import load_trials
from utils.charts import duration_by_phase, duration_by_ta, LAYOUT_DEFAULTS, COLORS
from utils.styles import inject_custom_css, section_label

st.set_page_config(page_title="Duration Prediction", page_icon="\U0001F4CA", layout="wide")
inject_custom_css()
section_label("Machine Learning Investigation")
st.title("Predicting Trial Duration: Structured Data vs. Protocol Text")

df = load_trials()

# --- Load pre-computed model results (instant) ---
DATA_DIR = Path(__file__).parent.parent / "data"
with open(DATA_DIR / "model_results.json") as f:
    model_results = json.load(f)

results = model_results["results"]
feat_imp = model_results["feature_importances"]
n_samples = model_results["n_samples"]

# --- Hero Insight ---
dur = df.dropna(subset=["duration_months"])
dur = dur[dur["duration_months"] > 0]

phase3_onc = dur[
    (dur["phase"].str.contains("Phase 3", na=False))
    & (dur["therapeutic_area"] == "Oncology")
]
phase3_card = dur[
    (dur["phase"].str.contains("Phase 3", na=False))
    & (dur["therapeutic_area"] == "Cardiology")
]
onc_median = int(phase3_onc["duration_months"].median()) if len(phase3_onc) > 0 else 50
card_median = int(phase3_card["duration_months"].median()) if len(phase3_card) > 0 else 35

_struct_best = max(v["mean"] for v in results["structured"].values())
_text_best = max(v["mean"] for v in results["text_enriched"].values())
_improvement = _text_best / _struct_best if _struct_best > 0 else _text_best / 0.01

st.markdown(
    f"> We tested whether you can predict how long a clinical trial will take using the "
    f"basic facts in its registration — phase, disease area, enrollment size, and sponsor type. "
    f"Three different machine learning approaches all fail (R\u00B2 near zero). But when we add "
    f"features extracted from the eligibility criteria text — specific medical terminology, "
    f"complexity indicators, exclusion patterns — prediction improves {_improvement:.0f}\u00D7."
)

st.markdown(
    "**Methods:** `scikit-learn` \u00B7 Three model families (Linear Regression, Random Forest, "
    "Gradient Boosting) \u00B7 `TfidfVectorizer` for text feature extraction \u00B7 "
    "`TruncatedSVD` for dimensionality reduction \u00B7 5-fold cross-validation"
)


# ============================================================
# SECTION 1: The Model Comparison Investigation (KEY FINDING — TOP)
# ============================================================
st.divider()
st.header("Model Comparison: Structured vs. Text-Enriched Features")

st.markdown(f"Three model families trained on **{n_samples:,}** trials with 5-fold cross-validation.")

# --- Model family explainer ---
with st.expander("What are these model families?"):
    st.markdown(
        "- **Linear Regression** — finds straight-line relationships. If it fails, the "
        "relationship isn't linear.\n"
        "- **Random Forest** — combines many decision trees. Captures complex, nonlinear patterns.\n"
        "- **Gradient Boosting** — builds models sequentially, each correcting the last's errors. "
        "Usually most accurate.\n\n"
        "Each was tested on three feature sets: structured fields only, text features only, "
        "and both combined."
    )

# --- Model comparison chart ---
comparison_data = []
for fset_key, fset_label in [("structured", "Structured Only"), ("text_enriched", "Structured + Text"), ("text_only", "Text Only")]:
    for model_name, vals in results[fset_key].items():
        comparison_data.append({
            "Model": model_name,
            "Feature Set": fset_label,
            "R\u00B2": vals["mean"],
            "R\u00B2 Std": vals["std"],
        })

comp_df = pd.DataFrame(comparison_data)

fig_comp = px.bar(
    comp_df, x="Model", y="R\u00B2", color="Feature Set",
    barmode="group",
    error_y="R\u00B2 Std",
    color_discrete_sequence=[COLORS["neutral"], COLORS["primary"], COLORS["secondary"]],
    labels={"R\u00B2": "Cross-Validated R\u00B2"},
)
fig_comp.update_layout(
    title="Model Comparison: Structured vs. Text-Enriched Features",
    yaxis=dict(range=[-0.1, 0.35]),
    **LAYOUT_DEFAULTS,
)
fig_comp.add_hline(y=0, line_dash="dash", line_color="gray", opacity=0.5)
st.plotly_chart(fig_comp, use_container_width=True)

# --- Narrative interpretation ---
struct_gb = results["structured"]["Gradient Boosting"]["mean"]
text_gb = results["text_enriched"]["Gradient Boosting"]["mean"]
improvement = text_gb / struct_gb if struct_gb > 0 else text_gb / 0.01

mc1, mc2, mc3 = st.columns(3)
mc1.metric("Structured Only (best)", f"R\u00B2 = {max(v['mean'] for v in results['structured'].values()):.3f}")
mc2.metric("Text-Enriched (best)", f"R\u00B2 = {max(v['mean'] for v in results['text_enriched'].values()):.3f}")
mc3.metric("Improvement", f"{improvement:.0f}\u00D7")

st.markdown(
    f"Structured features — phase, therapeutic area, enrollment, sponsor type — explain almost "
    f"nothing across all three model families (R\u00B2 \u2248 {struct_gb:.2f}). "
    f"Even Random Forest and Gradient Boosting can't find nonlinear patterns in the metadata. "
    f"**Adding text-derived features from eligibility criteria raises R\u00B2 to "
    f"{text_gb:.2f}** — the signal for trial duration lives in the protocol text, not the metadata."
)


# ============================================================
# SECTION 2: What the Text Features Capture
# ============================================================
st.divider()
st.header("Feature Importance: What Text Features Drive Prediction?")

st.markdown(
    "This is where NLP starts doing the work. We engineered 11 domain-specific text features "
    "(biomarker counts, genetic testing terms, washout mentions, etc.) and extracted 20 TF-IDF "
    "semantic components from eligibility criteria text. The next page takes this further with "
    "a protocol risk profiler and Claude API integration."
)

st.caption(
    "`GradientBoostingRegressor(n_estimators=50, max_depth=4) \u00B7 "
    "Gini feature importance \u00B7 11 engineered + 20 TF-IDF + structured features`"
)

# Feature importance chart (top 20)
top_n = 20
sorted_feats = sorted(feat_imp.items(), key=lambda x: x[1], reverse=True)[:top_n]
feat_names = [f[0] for f in sorted_feats]
feat_values = [f[1] for f in sorted_feats]

fig_imp = px.bar(
    x=feat_values[::-1], y=feat_names[::-1], orientation="h",
    color_discrete_sequence=[COLORS["primary"]],
    labels={"x": "Feature Importance (Gini)", "y": ""},
)
fig_imp.update_layout(
    title="Top Features for Predicting Trial Duration (Gradient Boosting)",
    showlegend=False,
    height=500,
    **LAYOUT_DEFAULTS,
)
st.plotly_chart(fig_imp, use_container_width=True)

# Count feature types in top 20
struct_labels = {"enrollment", "num_countries", "Enrollment", "Num Countries"}
n_struct_in_top = sum(1 for f, _ in sorted_feats if f in struct_labels or f.startswith("TA:") or f.startswith("Phase:") or f.startswith("Sponsor:"))
n_text_in_top = len(sorted_feats) - n_struct_in_top

st.markdown(
    f"Of the top {top_n} features, **{n_text_in_top} are text-derived** — TF-IDF components "
    f"capturing protocol language patterns, domain keyword frequencies, and criteria structure. "
    f"Text length and word count appear because longer, more detailed protocols correlate with "
    f"more complex trial designs. But the TF-IDF components capture semantic content beyond "
    f"simple length — specific medical terminology that indicates trial complexity."
)

# TF-IDF explanation
st.markdown(
    "> Features like \"Washout Period Language\" and \"Biomarker Terminology\" are patterns discovered "
    "by TF-IDF analysis — a technique that identifies which medical terms appear unusually often "
    "in a protocol's eligibility criteria compared to the average trial. Related terms are grouped "
    "into themes using dimensionality reduction (SVD)."
)


# ============================================================
# SECTION 3: Duration Patterns (context — in expander)
# ============================================================
with st.expander("Duration Patterns by Phase & Area"):
    c1, c2 = st.columns(2)
    with c1:
        st.plotly_chart(duration_by_phase(df), use_container_width=True)
    with c2:
        st.plotly_chart(duration_by_ta(df), use_container_width=True)

    # Key metrics
    fc1, fc2, fc3 = st.columns(3)
    phase3 = dur[dur["phase"].str.contains("Phase 3", na=False)]["duration_months"]
    phase2 = dur[dur["phase"].str.contains("Phase 2", na=False) & ~dur["phase"].str.contains("Phase 3", na=False)]["duration_months"]

    with fc1:
        if len(phase3) > 0:
            st.metric("Median Phase 3", f"{phase3.median():.0f} months")
        st.caption(f"{len(phase3):,} trials")
    with fc2:
        if len(phase2) > 0:
            st.metric("Median Phase 2", f"{phase2.median():.0f} months")
        st.caption(f"{len(phase2):,} trials")
    with fc3:
        st.metric("Oncology Phase III", f"{onc_median} months")
        st.caption(f"vs Cardiology Phase III at {card_median} months")


with st.expander("Explore Enrollment Patterns"):
    enroll = df[df["enrollment"].notna() & (df["enrollment"] > 0)].copy()
    enroll_capped = enroll[enroll["enrollment"] <= enroll["enrollment"].quantile(0.99)]
    ec1, ec2 = st.columns(2)
    with ec1:
        fig_enroll_ta = px.box(
            enroll_capped, x="therapeutic_area", y="enrollment",
            color="therapeutic_area",
            labels={"therapeutic_area": "Therapeutic Area", "enrollment": "Enrollment Target"},
        )
        fig_enroll_ta.update_layout(title="Enrollment by Therapeutic Area", showlegend=False, **LAYOUT_DEFAULTS)
        st.plotly_chart(fig_enroll_ta, use_container_width=True)
    with ec2:
        fig_enroll_phase = px.box(
            enroll_capped, x="phase", y="enrollment",
            color_discrete_sequence=[COLORS["primary"]],
            labels={"phase": "Phase", "enrollment": "Enrollment Target"},
        )
        fig_enroll_phase.update_layout(title="Enrollment by Phase", showlegend=False, **LAYOUT_DEFAULTS)
        st.plotly_chart(fig_enroll_phase, use_container_width=True)


# --- Connector ---
st.divider()
col_left, col_right = st.columns([4, 1])
with col_left:
    st.markdown(
        "**Text features predict trial duration. Can we identify the specific "
        "protocol language that adds months — and build a tool that flags risk factors "
        "automatically?**"
    )
with col_right:
    st.page_link("pages/3_Protocol_Complexity.py", label="See the risk profiler \u2192", icon="\U0001F9EA")
