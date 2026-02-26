import json
from pathlib import Path

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px

from utils.data_loader import load_trials, load_protocols, get_sample_trials, load_trial_criteria
from utils.charts import LAYOUT_DEFAULTS, COLORS, TA_COLORS
from utils.text_features import extract_text_features
from utils.nlp_analyzer import score_criteria
from utils.styles import inject_custom_css, section_label, byline

st.set_page_config(page_title="Protocol Complexity & NLP", page_icon="\U0001F9EA", layout="wide")
inject_custom_css()
byline()
section_label("NLP Feature Engineering + Claude API")
st.title("Protocol Risk Profiling: 16 Text Features That Predict Delay")

# --- Load pre-computed results (instant) ---
DATA_DIR = Path(__file__).parent.parent / "data"
with open(DATA_DIR / "risk_model_results.json") as f:
    risk_model = json.load(f)

tsne_path = DATA_DIR / "protocol_tsne.parquet"
tsne_data = pd.read_parquet(tsne_path) if tsne_path.exists() else None

# --- Hero Insight ---
effects = risk_model["directional_effects"]
# Separate binary flags from count-based features for display
_binary_flags = {"performance_status_req", "contraception_req"}
_count_effects = sorted(
    ((k, v) for k, v in effects.items()
     if k not in ("text_length", "word_count") and k not in _binary_flags and v > 0),
    key=lambda x: x[1], reverse=True,
)
# Top binary effect
_top_binary = max(
    ((k, v) for k, v in effects.items() if k in _binary_flags and v > 0),
    key=lambda x: x[1], default=None,
)

st.markdown(
    f"> **Protocol text contains quantifiable duration risk signals invisible to structured metadata.** "
    f"After controlling for therapeutic area and phase: requiring a "
    f"performance status screen adds **{effects.get('performance_status_req', 3.0):+.1f} months**, "
    f"each additional {_count_effects[0][0].replace('_', ' ')} adds **{_count_effects[0][1]:+.1f} months**, "
    f"and each {_count_effects[1][0].replace('_', ' ')} adds **{_count_effects[1][1]:+.1f} months**."
)

st.markdown(
    "**Methods:** Domain-specific `NLP feature engineering` (16 text features: keyword extraction, "
    "procedure detection, prior therapy patterns, performance status) \u00B7 "
    "Residual analysis (removing TA/phase effects) \u00B7 `LinearRegression` for directional effects \u00B7 "
    "`GradientBoosting` for feature importance \u00B7 `Claude API` for semantic protocol scoring"
)


# ============================================================
# SECTION 1: What We Measured and Why
# ============================================================
st.divider()
st.header("NLP Feature Engineering: What We Measured and Why")

st.markdown(
    "CRO feasibility teams currently estimate duration using phase, disease area, and "
    "enrollment — the same fields that explain only 4% of variance. But eligibility "
    "criteria contain specific clinical requirements that directly drive screening "
    "difficulty, site workflow complexity, and patient availability."
)

# Dynamically build the feature list from actual effects, sorted by magnitude
_positive_sorted = sorted(
    ((k, v) for k, v in effects.items()
     if k not in ("text_length", "word_count", "numbered_criteria") and v > 0.1),
    key=lambda x: x[1], reverse=True,
)

_feature_descriptions = {
    "performance_status_req": "Trials requiring ECOG or Karnofsky screening target sicker patients with more complex baseline assessments",
    "prior_therapy_mentions": "More prior treatment requirements narrow the eligible population and extend screening",
    "genetic_mentions": "Genetic screening (BRCA, HER2, PD-L1) needs specialized labs and limits the eligible population",
    "comorbidity_exclusions": "More comorbidity exclusions narrow the recruitment funnel",
    "procedure_mentions": "Biopsies, imaging, and procedures add screening visits and scheduling complexity",
    "inclusion_mentions": "More inclusion criteria = narrower funnel, longer recruitment",
    "exclusion_mentions": "More exclusion criteria = more patients screened out",
    "biomarker_mentions": "Biomarker thresholds add screening visits and lab coordination",
    "organ_fn_mentions": "Hepatic/renal panels require scheduling and may disqualify patients",
    "lab_value_count": "Specific numeric thresholds increase screening rigor",
}

feature_bullets = []
for feat, val in _positive_sorted[:6]:  # Top 6 positive effects
    name = feat.replace("_", " ").title()
    desc = _feature_descriptions.get(feat, "")
    feature_bullets.append(f"- **{name}** (+{val:.1f} months each): {desc}")

st.markdown("\n".join(feature_bullets))

st.markdown(
    "This isn't a black-box prediction — each feature maps to an operational reality "
    "that CRO project managers already recognize intuitively."
)

# What the model captures vs. doesn't
with st.expander("What the model covers vs. what it can't"):
    st.markdown(
        "**What the model captures (~24% of duration variance):**\n"
        "- Protocol complexity signals extracted from eligibility criteria text\n"
        "- Therapeutic area and phase baselines\n"
        "- Enrollment size and geographic scope\n\n"
        "**What explains the remaining ~76% (not in eligibility text):**\n"
        "- **Site-level factors:** Site selection quality, investigator experience, "
        "geographic recruitment capacity\n"
        "- **Regulatory timelines:** IRB/ethics review, country-specific approval processes\n"
        "- **Operational execution:** Recruitment strategy, competing trials for same patients, "
        "CRA staffing\n"
        "- **Sponsor decisions:** Budget changes, strategic pivots, M&A activity\n"
        "- **Protocol amendments:** Mid-trial changes that extend timelines\n"
        "- **External events:** Pandemics, supply chain disruption\n\n"
        "The 24% that eligibility text captures is the **protocol-design signal** — "
        "the complexity baked into the trial before a single patient is screened. "
        "The remaining 76% is **operational execution** — factors that become "
        "knowable only after the trial starts. Both matter, but only the first "
        "is available at the feasibility stage when CROs make their commitments."
    )

st.caption(
    "`Residual analysis \u00B7 LinearRegression on TA/phase residuals \u00B7 "
    f"GradientBoostingRegressor feature importance \u00B7 {risk_model['n_samples']:,} trials \u00B7 "
    "domain-specific NLP feature engineering`"
)


# ============================================================
# SECTION 2: Duration Risk Factors
# ============================================================
st.divider()
st.header("Directional Effects on Trial Duration")
st.markdown(
    "After accounting for therapeutic area and phase (the known drivers), "
    "these text-derived features predict additional trial months."
)

# Directional effects chart
effect_data = pd.DataFrame([
    {"Feature": k.replace("_", " ").title(), "Months per Mention": v}
    for k, v in sorted(effects.items(), key=lambda x: x[1], reverse=True)
])

fig_effects = px.bar(
    effect_data, x="Months per Mention", y="Feature", orientation="h",
    color="Months per Mention",
    color_continuous_scale=["#4FC0D0", "#6C757D", "#E63946"],
    color_continuous_midpoint=0,
    labels={"Months per Mention": "Additional Months per Mention"},
)
fig_effects.update_layout(
    title="Text Feature Effects on Trial Duration (Beyond TA/Phase Baseline)",
    showlegend=False,
    coloraxis_showscale=False,
    height=400,
    **LAYOUT_DEFAULTS,
)
st.plotly_chart(fig_effects, use_container_width=True)

st.markdown(
    "Each bar shows how many additional months one more mention of that text feature predicts, "
    "**holding therapeutic area and phase constant**. Performance status requirements and "
    "prior therapy restrictions are the strongest positive signals — they indicate protocol "
    "designs targeting sicker, more heavily pre-treated patient populations that are harder "
    "to find and screen."
)

st.markdown(
    "**What this means for feasibility:** Current CRO costing models treat all "
    "Phase 3 Oncology trials as roughly equivalent. But an Oncology Phase 3 trial "
    "requiring ECOG screening, prior therapy documentation, BRCA testing, and organ "
    "function panels will run months longer than one with simple inclusion criteria — "
    "and this model quantifies exactly how many months."
)


# ============================================================
# SECTION 3: Protocol Risk Profiler
# ============================================================
st.divider()
st.header("Interactive Risk Profiler")
st.markdown(
    "Select a trial to see its duration risk decomposition — which text features "
    "push its predicted duration above or below baseline for its therapeutic area and phase. "
    "The **instant profiler** uses the NLP model from this investigation; the **Claude API analysis** "
    "(available below) adds LLM-powered semantic scoring for deeper protocol review."
)

st.caption(
    "`NLP feature extraction \u00B7 baseline comparison (TA/phase median) \u00B7 "
    "text-derived risk contribution \u00B7 optional Claude API for semantic analysis`"
)

# Trial picker with filter-then-select
protocols = load_protocols()
sample_trials = get_sample_trials()

if sample_trials is not None and len(sample_trials) > 0:
    if "complexity_score" in sample_trials.columns:
        sample_sorted = sample_trials.sort_values("complexity_score", ascending=False).reset_index(drop=True)
    else:
        sample_sorted = sample_trials.reset_index(drop=True)

    # Filter-then-select pattern
    ta_list = sorted(sample_sorted["therapeutic_area"].dropna().unique().tolist())
    phase_list = sorted(sample_sorted["phase"].dropna().unique().tolist())

    f1, f2 = st.columns(2)
    with f1:
        ta_filter = st.selectbox("Filter by Therapeutic Area", ["All"] + ta_list)
    with f2:
        phase_filter = st.selectbox("Filter by Phase", ["All"] + phase_list)

    filtered_trials = sample_sorted.copy()
    if ta_filter != "All":
        filtered_trials = filtered_trials[filtered_trials["therapeutic_area"] == ta_filter]
    if phase_filter != "All":
        filtered_trials = filtered_trials[filtered_trials["phase"] == phase_filter]

    st.caption(f"Showing {len(filtered_trials)} of {len(sample_sorted)} trials")

    display_options = []
    for _, row in filtered_trials.iterrows():
        title_short = str(row.get("title", ""))[:70]
        ta = row.get("therapeutic_area", "")
        phase = row.get("phase", "")
        score = row.get("complexity_score", "")
        score_str = f" | Score: {score}" if score != "" else ""
        display_options.append(f"{row['nct_id']} | {ta} | {phase}{score_str} \u2014 {title_short}")

    selected = st.selectbox("Select a trial:", ["(Select a trial)"] + display_options)

    if selected != "(Select a trial)":
        nct_id = selected.split(" | ")[0]
        criteria = load_trial_criteria(nct_id)

        if criteria:
            # Get trial metadata
            trial_row = filtered_trials[filtered_trials["nct_id"] == nct_id].iloc[0]
            ta = trial_row.get("therapeutic_area", "Unknown")
            phase = trial_row.get("phase", "Unknown")

            # Compute text features for this trial (instant — just regex counting)
            trial_df = pd.DataFrame([{"eligibility_criteria": criteria}])
            trial_feats = extract_text_features(trial_df)

            # Baseline for this TA/phase
            baselines = risk_model["baselines"]
            baseline_match = [b for b in baselines if b["therapeutic_area"] == ta and b["phase"] == phase]
            baseline_months = baseline_match[0]["median"] if baseline_match else None

            # Compute risk contributions
            contributions = {}
            for feat, coef in effects.items():
                feat_val = trial_feats[feat].iloc[0] if feat in trial_feats.columns else 0
                contributions[feat] = feat_val * coef

            total_text_risk = sum(contributions.values())

            # Display metrics
            if protocols is not None:
                actual_row = protocols[protocols["nct_id"] == nct_id]
                actual_dur = actual_row["duration_months"].values[0] if len(actual_row) > 0 and pd.notna(actual_row["duration_months"].values[0]) else None
            else:
                actual_dur = None

            # Compute model-adjusted estimate
            model_estimate = None
            if baseline_months is not None:
                model_estimate = baseline_months + total_text_risk

            mc1, mc2, mc3, mc4 = st.columns(4)
            if baseline_months is not None:
                mc1.metric(f"TA/Phase Median", f"{baseline_months:.0f} months",
                           help=f"Median duration for all {ta} {phase} trials in the dataset")
            mc2.metric("Text-Derived Adjustment", f"{total_text_risk:+.1f} months",
                       help="Additional months predicted by protocol text complexity")
            if model_estimate is not None:
                mc3.metric("Model-Adjusted Estimate", f"{model_estimate:.0f} months",
                           help="TA/Phase median + text-derived adjustment")
            if actual_dur is not None and actual_dur > 0:
                mc4.metric("Recorded Duration", f"{actual_dur:.0f} months",
                           help="Start date to completion date as registered on ClinicalTrials.gov")

            # --- Interpretation of model vs actual ---
            if actual_dur is not None and actual_dur > 0 and model_estimate is not None:
                gap = actual_dur - model_estimate
                if abs(gap) > 10:
                    st.caption(
                        f"*The model-adjusted estimate ({model_estimate:.0f} mo) and recorded duration "
                        f"({actual_dur:.0f} mo) diverge by {abs(gap):.0f} months. This is expected — "
                        f"the model explains ~23% of duration variance (R\u00B2 = 0.23). "
                        f"Its value is identifying systematic risk patterns across trials, not "
                        f"predicting any single trial precisely. Factors like regulatory delays, "
                        f"site performance, and enrollment challenges aren't captured in eligibility text.*"
                    )
                elif abs(gap) > 3:
                    st.caption(
                        f"*Model-adjusted estimate ({model_estimate:.0f} mo) is within "
                        f"{abs(gap):.0f} months of actual ({actual_dur:.0f} mo).*"
                    )

            # --- Risk Level Classification ---
            if total_text_risk > 5:
                risk_level = "High"
                risk_color = "red"
                risk_msg = (
                    "This protocol's eligibility criteria are significantly more complex than typical "
                    f"trials in {ta} {phase}. Multiple text features contribute to extended timelines — "
                    "the model flags this as a higher-risk protocol for duration overruns."
                )
            elif total_text_risk > 2:
                risk_level = "Elevated"
                risk_color = "orange"
                risk_msg = (
                    f"Several criteria features push this trial above baseline for {ta} {phase}. "
                    "Protocol complexity is above average — factor additional screening time into feasibility."
                )
            elif total_text_risk > 0:
                risk_level = "Moderate"
                risk_color = "blue"
                risk_msg = (
                    f"Protocol complexity is near baseline for {ta} {phase}. "
                    "Text-derived risk factors are present but modest."
                )
            else:
                risk_level = "Low"
                risk_color = "green"
                risk_msg = (
                    "Simpler-than-average eligibility criteria suggest this trial may run at or below "
                    f"baseline duration for {ta} {phase}."
                )

            st.markdown(f"### Risk Level: :{risk_color}[{risk_level}]")
            st.markdown(risk_msg)

            # Contribution breakdown
            contrib_df = pd.DataFrame([
                {"Feature": k.replace("_", " ").title(), "Contribution (months)": v,
                 "Raw Count": trial_feats[k].iloc[0] if k in trial_feats.columns else 0}
                for k, v in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True)
                if abs(v) > 0.01
            ])

            if len(contrib_df) > 0:
                fig_contrib = px.bar(
                    contrib_df, x="Contribution (months)", y="Feature", orientation="h",
                    color="Contribution (months)",
                    color_continuous_scale=["#4FC0D0", "#6C757D", "#E63946"],
                    color_continuous_midpoint=0,
                    hover_data=["Raw Count"],
                )
                fig_contrib.update_layout(
                    title="Risk Contribution by Text Feature",
                    showlegend=False,
                    coloraxis_showscale=False,
                    height=350,
                    **LAYOUT_DEFAULTS,
                )
                st.plotly_chart(fig_contrib, use_container_width=True)

                # --- Plain-English summary of key drivers ---
                top_contributors = [
                    (k, contributions[k], trial_feats[k].iloc[0] if k in trial_feats.columns else 0)
                    for k in contributions
                    if abs(contributions[k]) > 0.5
                ]
                top_contributors.sort(key=lambda x: abs(x[1]), reverse=True)

                if top_contributors:
                    st.markdown("**Key risk drivers for this protocol:**")
                    for feat, val, count in top_contributors:
                        direction = "adds" if val > 0 else "reduces"
                        st.markdown(
                            f"- {feat.replace('_', ' ').title()}: {count:.0f} mentions "
                            f"\u2192 {direction} {abs(val):.1f} months to expected duration"
                        )
            else:
                st.info("No significant text-derived risk factors detected for this protocol.")

            # Show raw text in expander
            with st.expander("View Eligibility Criteria Text"):
                st.text(criteria[:3000])

# Custom criteria input
with st.expander("Or paste custom eligibility criteria"):
    criteria_input = st.text_area(
        "Eligibility criteria text:",
        height=200,
        placeholder="Inclusion Criteria:\n1. Adults aged 18-75...\n\nExclusion Criteria:\n1. Prior treatment with...",
    )

    col_local, col_claude = st.columns(2)
    with col_local:
        if st.button("Risk Profile (instant)", type="primary"):
            if criteria_input.strip():
                trial_df = pd.DataFrame([{"eligibility_criteria": criteria_input}])
                trial_feats = extract_text_features(trial_df)
                contributions = {f: trial_feats[f].iloc[0] * c for f, c in effects.items()}
                total_risk = sum(contributions.values())
                st.metric("Text-Derived Risk", f"{total_risk:+.1f} months above TA/phase baseline")
                for feat, contrib in sorted(contributions.items(), key=lambda x: abs(x[1]), reverse=True):
                    if abs(contrib) > 0.1:
                        st.caption(f"{feat.replace('_', ' ').title()}: {contrib:+.1f} months ({trial_feats[feat].iloc[0]:.0f} mentions)")

    with col_claude:
        if st.button("Claude API Analysis (detailed)"):
            if criteria_input.strip():
                with st.spinner("Analyzing with Claude API..."):
                    result = score_criteria(criteria_input)
                if result is None:
                    st.error("API key not configured.")
                else:
                    st.metric("Complexity Score", f"{result.get('complexity_score', 'N/A')} / 10")
                    st.markdown(f"**Rationale:** {result.get('complexity_rationale', 'N/A')}")
                    markers = result.get("complexity_markers", {})
                    if markers:
                        for marker, present in markers.items():
                            icon = "\u2705" if present else "\u274C"
                            st.caption(f"{icon} {marker.replace('_', ' ').title()}")


# ============================================================
# SECTION 4: Protocol Similarity Map
# ============================================================
with st.expander("Explore Protocol Similarity Map"):
    st.markdown(
        "Each dot below is a trial protocol. Protocols with similar eligibility criteria "
        "language appear close together — revealing clusters of protocol types that share "
        "design patterns. Color shows whether the trial ran longer (red) or shorter (blue) "
        "than expected for its therapeutic area and phase."
    )

    st.caption(
        "`TfidfVectorizer(ngram_range=(1,2), max_features=1000) \u2192 "
        "TruncatedSVD(n=20) + engineered features \u2192 t-SNE(perplexity=30)`"
    )

    if tsne_data is not None and "tsne_x" in tsne_data.columns:
        color_by = st.radio(
            "Color by:", ["Duration (months)", "Duration Residual", "Complexity Score"],
            horizontal=True,
        )
        color_map = {
            "Duration (months)": "duration_months",
            "Duration Residual": "duration_residual",
            "Complexity Score": "complexity_score",
        }
        color_col = color_map[color_by]

        plot_df = tsne_data.dropna(subset=[color_col, "tsne_x", "tsne_y"]).copy()
        fig_tsne = px.scatter(
            plot_df, x="tsne_x", y="tsne_y", color=color_col,
            color_continuous_scale="RdBu_r" if color_col == "duration_residual" else "Viridis",
            color_continuous_midpoint=0 if color_col == "duration_residual" else None,
            hover_data=["nct_id", "title", "therapeutic_area", "phase", "duration_months", "complexity_score"],
            labels={"tsne_x": "t-SNE 1", "tsne_y": "t-SNE 2"},
        )
        fig_tsne.update_layout(
            title="Protocol Similarity Map",
            height=500,
            **LAYOUT_DEFAULTS,
        )
        st.plotly_chart(fig_tsne, use_container_width=True)
    else:
        st.info("Embedding computation requires scored protocols data.")


# --- Connector ---
st.divider()
col_left, col_right = st.columns([4, 1])
with col_left:
    st.markdown(
        "**Protocol risk profiling is one AI application for clinical trial operations. "
        "Where else can AI create leverage across the CRO — and how do these findings "
        "reframe the opportunity?**"
    )
with col_right:
    st.page_link("pages/4_AI_Opportunity_Framework.py", label="View the AI roadmap \u2192", icon="\U0001F916")
