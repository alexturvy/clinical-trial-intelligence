import json
from pathlib import Path

import streamlit as st
import pandas as pd
import plotly.express as px

from utils.charts import opportunity_matrix, LAYOUT_DEFAULTS, COLORS
from utils.data_loader import load_trials
from utils.styles import inject_custom_css, section_label, byline

st.set_page_config(page_title="AI Opportunity Framework", page_icon="\U0001F916", layout="wide")
inject_custom_css()
byline()
section_label("Strategic Framework")
st.title("AI Opportunity Framework: 16 Applications Across 5 CRO Functions")

# --- Load pre-computed model results for dynamic stats ---
DATA_DIR = Path(__file__).parent.parent / "data"
with open(DATA_DIR / "model_results.json") as f:
    model_results = json.load(f)
with open(DATA_DIR / "risk_model_results.json") as f:
    risk_model = json.load(f)

struct_best = max(v["mean"] for v in model_results["results"]["structured"].values())
text_best = max(v["mean"] for v in model_results["results"]["text_enriched"].values())
improvement_ratio = text_best / struct_best if struct_best > 0 else text_best / 0.01

effects = risk_model["directional_effects"]
_binary_flags = {"performance_status_req", "contraception_req"}
top_positive = max(
    ((k, v) for k, v in effects.items()
     if k not in ("text_length", "word_count") and k not in _binary_flags),
    key=lambda x: x[1],
)

df = load_trials()
total_trials = len(df)

# Time-to-value color map
TTV_COLORS = {
    "3-6 months": "#2A9D8F",
    "6-12 months": "#1B6B93",
    "12-18 months": "#F4A261",
    "18+ months": "#E63946",
}

# --- AI Opportunities Data (recalibrated for honest assessment) ---
OPPORTUNITIES = [
    # Clinical Operations
    {"business_function": "Clinical Operations", "category": "Prediction", "use_case": "Site Performance Prediction",
     "description": "Predict site enrollment rates and identify underperforming sites early using historical trial data",
     "feasibility": 8, "impact": 9, "data_readiness": 8,
     "time_to_value": "3-6 months", "prerequisite": "Requires historical site enrollment data",
     "connection_to_investigation": None},
    {"business_function": "Clinical Operations", "category": "NLP", "use_case": "Protocol Amendment Analysis",
     "description": "NLP analysis of protocol amendments to identify patterns and reduce amendment frequency",
     "feasibility": 7, "impact": 8, "data_readiness": 6,
     "time_to_value": "6-12 months", "prerequisite": "Requires historical amendment tracking data",
     "connection_to_investigation": "This investigation proved that protocol text contains quantifiable duration signals — the same NLP features can measure amendment impact"},
    {"business_function": "Clinical Operations", "category": "Automation", "use_case": "Monitoring Visit Scheduling",
     "description": "Optimize monitoring visit schedules based on site risk scores and enrollment velocity",
     "feasibility": 7, "impact": 5, "data_readiness": 7,
     "time_to_value": "6-12 months", "prerequisite": "Requires CTMS integration for real-time site data",
     "connection_to_investigation": None},
    {"business_function": "Clinical Operations", "category": "Prediction", "use_case": "Enrollment Forecasting",
     "description": "ML models to forecast enrollment timelines and identify recruitment bottlenecks",
     "feasibility": 7, "impact": 9, "data_readiness": 7,
     "time_to_value": "6-12 months", "prerequisite": "Requires historical site enrollment data",
     "connection_to_investigation": "Directly extends this investigation — the same text features that predict duration can predict enrollment velocity"},

    # Medical & Safety
    {"business_function": "Medical & Safety", "category": "NLP", "use_case": "Adverse Event Classification",
     "description": "Automated MedDRA coding and severity classification of adverse event narratives",
     "feasibility": 6, "impact": 8, "data_readiness": 5,
     "time_to_value": "12-18 months", "prerequisite": "Requires validated training data and regulatory review",
     "connection_to_investigation": None},
    {"business_function": "Medical & Safety", "category": "NLP", "use_case": "Medical Writing Assistance",
     "description": "AI-assisted drafting of CSRs, IBs, and safety narratives with structured data integration",
     "feasibility": 5, "impact": 6, "data_readiness": 5,
     "time_to_value": "12-18 months", "prerequisite": "Requires quality review framework for LLM-generated regulated content",
     "connection_to_investigation": None},
    {"business_function": "Medical & Safety", "category": "Prediction", "use_case": "Safety Signal Detection",
     "description": "Early detection of safety signals using statistical and ML methods on accumulating data",
     "feasibility": 3, "impact": 10, "data_readiness": 4,
     "time_to_value": "18+ months", "prerequisite": "Highest regulatory bar — requires extensive validation and FDA alignment",
     "connection_to_investigation": None},

    # Laboratory Services
    {"business_function": "Laboratory Services", "category": "Optimization", "use_case": "Sample Logistics Optimization",
     "description": "Route optimization for biological sample shipments to minimize transit time and cost",
     "feasibility": 8, "impact": 4, "data_readiness": 7,
     "time_to_value": "3-6 months", "prerequisite": "Solved problem in other industries — primarily engineering",
     "connection_to_investigation": None},
    {"business_function": "Laboratory Services", "category": "Prediction", "use_case": "Lab Kit Demand Forecasting",
     "description": "Predict lab kit requirements per site based on enrollment and visit schedules",
     "feasibility": 8, "impact": 4, "data_readiness": 8,
     "time_to_value": "3-6 months", "prerequisite": "Standard demand forecasting — requires LIMS integration",
     "connection_to_investigation": None},
    {"business_function": "Laboratory Services", "category": "Automation", "use_case": "Lab Data Reconciliation",
     "description": "Automated matching and reconciliation of lab results with clinical data",
     "feasibility": 7, "impact": 5, "data_readiness": 6,
     "time_to_value": "6-12 months", "prerequisite": "Requires standardized lab data formats across vendors",
     "connection_to_investigation": None},

    # Business Development
    {"business_function": "Business Development", "category": "NLP", "use_case": "RFP Response Optimization",
     "description": "AI-assisted RFP responses using historical win/loss data and proposal templates",
     "feasibility": 6, "impact": 8, "data_readiness": 4,
     "time_to_value": "12-18 months", "prerequisite": "Requires structured win/loss data that most CROs don't track",
     "connection_to_investigation": None},
    {"business_function": "Business Development", "category": "Prediction", "use_case": "Competitive Intelligence",
     "description": "Monitor and analyze competitor trial activity to identify market opportunities",
     "feasibility": 8, "impact": 6, "data_readiness": 8,
     "time_to_value": "3-6 months", "prerequisite": "Public data from ClinicalTrials.gov — similar to this investigation's data pipeline",
     "connection_to_investigation": None},
    {"business_function": "Business Development", "category": "Prediction", "use_case": "Trial Costing Model",
     "description": "ML-based trial cost estimation using historical project data and protocol features",
     "feasibility": 5, "impact": 10, "data_readiness": 4,
     "time_to_value": "12-18 months", "prerequisite": "Requires clean historical cost data linked to protocol features",
     "connection_to_investigation": "Directly extends this investigation — duration is the largest cost driver, and our text-enriched model quantifies protocol complexity impact"},

    # Business Operations
    {"business_function": "Business Operations", "category": "Automation", "use_case": "Document Processing",
     "description": "Automated extraction and classification of regulatory documents, IRB approvals, contracts",
     "feasibility": 8, "impact": 5, "data_readiness": 7,
     "time_to_value": "3-6 months", "prerequisite": "Table stakes — many vendors already offer this",
     "connection_to_investigation": None},
    {"business_function": "Business Operations", "category": "Optimization", "use_case": "Resource Allocation",
     "description": "Optimize CRA and PM assignments across trials based on therapeutic expertise and workload",
     "feasibility": 4, "impact": 8, "data_readiness": 4,
     "time_to_value": "18+ months", "prerequisite": "Requires clean historical staffing data rarely tracked systematically",
     "connection_to_investigation": None},
    {"business_function": "Business Operations", "category": "NLP", "use_case": "Training Content Generation",
     "description": "Auto-generate therapeutic area training materials from protocol documents and literature",
     "feasibility": 7, "impact": 3, "data_readiness": 6,
     "time_to_value": "6-12 months", "prerequisite": "Low strategic value — incremental efficiency for L&D teams",
     "connection_to_investigation": None},
]

n_opportunities = len(OPPORTUNITIES)
n_functions = len(set(o["business_function"] for o in OPPORTUNITIES))

# Build DataFrame for scoring
opp_df_all = pd.DataFrame(OPPORTUNITIES)
opp_df_all["combined_score"] = (opp_df_all["feasibility"] + opp_df_all["impact"] * 1.5 + opp_df_all["data_readiness"]) / 3.5
top_two = opp_df_all.sort_values("combined_score", ascending=False).head(2)["use_case"].tolist()

# --- Hero + Transition Narrative ---
st.markdown(
    f"> The previous three pages demonstrated one complete AI/ML application: predicting "
    f"trial duration from protocol text. It worked because the data was available "
    f"(ClinicalTrials.gov), the signal was real (eligibility criteria predict duration "
    f"{improvement_ratio:.0f}\u00D7 better than metadata alone), and the output is actionable "
    f"(quantified risk by text feature). **Now: where else across CRO operations does "
    f"the same logic apply?**"
)

st.caption("`Strategic framework · feasibility × impact × data readiness scoring · discovery-to-production pipeline`")


# ============================================================
# SECTION 1: What This Investigation Proved
# ============================================================
st.divider()
st.header("What This Investigation Proved")

# Compute second positive effect for display
_positive_effects = sorted(
    ((k, v) for k, v in effects.items() if k not in ("text_length", "word_count") and v > 0),
    key=lambda x: x[1], reverse=True,
)

st.markdown(
    f"Three findings that reframe how CROs should think about AI:\n\n"
    f"1. **Structured trial data has almost no predictive power for duration** (R\u00B2 \u2248 {struct_best:.2f}). "
    f"Phase, disease area, enrollment, and sponsor — the fields in every CRO database — "
    f"explain {struct_best * 100:.0f}% of duration variance.\n\n"
    f"2. **Eligibility criteria text contains quantifiable risk signals** — each "
    f"{_positive_effects[0][0].replace('_', ' ')} adds ~{_positive_effects[0][1]:.1f} months, "
    f"each {_positive_effects[1][0].replace('_', ' ')} adds ~{_positive_effects[1][1]:.1f} months. "
    f"These signals are invisible to current CRO systems.\n\n"
    f"3. **NLP on protocol text improves prediction {improvement_ratio:.0f}\u00D7 over structured data.** The same "
    f"approach — extracting structured signals from unstructured text — applies across "
    f"CRO operations."
)

st.markdown(
    "The pattern that made this work — **unstructured text containing structured signals "
    "that current systems ignore** — repeats across CRO business functions. Protocols, "
    "amendments, AE narratives, regulatory documents, RFP responses: all contain "
    "extractable intelligence that drives better decisions."
)


# ============================================================
# SECTION 2: Scoring Methodology
# ============================================================
st.divider()
st.header("Scoring Methodology")

st.markdown(
    "Each opportunity is assessed on three dimensions, scored by evaluating the current "
    "state of CRO technology and data infrastructure:\n\n"
    "**Feasibility (1-10):** Can this be built with current technology?\n"
    "- 8-10: Existing tools and proven approaches; implementation is primarily engineering\n"
    "- 5-7: Requires some R&D; approach is known but adaptation needed\n"
    "- 1-4: Research-grade; regulatory, safety, or data challenges make deployment uncertain\n\n"
    "**Impact (1-10):** How much business value does this create?\n"
    "- 8-10: Directly affects revenue, timeline, or patient safety across multiple trials\n"
    "- 5-7: Measurable efficiency gains for specific teams or workflows\n"
    "- 1-4: Incremental improvement; nice-to-have rather than need-to-have\n\n"
    "**Data Readiness (1-10):** Does the data exist and is it usable?\n"
    "- 8-10: Structured, accessible data already collected in standard CRO systems\n"
    "- 5-7: Data exists but requires cleaning, linking, or access negotiation\n"
    "- 1-4: Data is fragmented, unstructured, or not systematically collected"
)


# ============================================================
# SECTION 3: Opportunity Scoring Matrix
# ============================================================
st.divider()
st.header("Opportunity Matrix")

st.markdown(
    "Bubble position shows feasibility vs. impact; size shows data readiness. "
    "Color shows time to value. **Top-right quadrant = quick wins.**"
)

# Toggle: All vs Connected
view_mode = st.radio(
    "View:",
    ["All Opportunities", "Connected to This Investigation"],
    horizontal=True,
)

if view_mode == "Connected to This Investigation":
    filtered_opps = [o for o in OPPORTUNITIES if o["connection_to_investigation"] is not None]
else:
    filtered_opps = OPPORTUNITIES

st.plotly_chart(opportunity_matrix(filtered_opps), use_container_width=True)


# ============================================================
# SECTION 4: Strategic Recommendations
# ============================================================
st.divider()
st.header("Strategic Recommendations")

# Sort by combined score
sorted_opps = sorted(OPPORTUNITIES, key=lambda o: (o["feasibility"] + o["impact"] * 1.5 + o["data_readiness"]) / 3.5, reverse=True)

start_here = [o for o in sorted_opps if o["time_to_value"] in ("3-6 months", "6-12 months")][:4]
build_toward = [o for o in sorted_opps if o["time_to_value"] in ("6-12 months", "12-18 months") and o not in start_here][:4]
monitor = [o for o in sorted_opps if o["time_to_value"] in ("18+ months",)]

st.subheader("Start Here (0-6 months)")
for o in start_here:
    conn = f" *— connects to this investigation*" if o["connection_to_investigation"] else ""
    st.markdown(
        f"- **{o['use_case']}** ({o['business_function']}): "
        f"{o['description']}{conn}"
    )

st.subheader("Build Toward (6-18 months)")
for o in build_toward:
    conn = f" *— connects to this investigation*" if o["connection_to_investigation"] else ""
    st.markdown(
        f"- **{o['use_case']}** ({o['business_function']}): "
        f"{o['prerequisite']}{conn}"
    )

st.subheader("Monitor (18+ months)")
for o in monitor:
    st.markdown(
        f"- **{o['use_case']}** ({o['business_function']}): "
        f"{o['prerequisite']}"
    )


# ============================================================
# SECTION 5: From Investigation to Application
# ============================================================
st.divider()
st.header("From Investigation to Application")

st.markdown(
    "The three highest-impact opportunities connect directly to this investigation's findings:"
)

st.markdown(
    "**Enrollment Forecasting** — The same text features that predict duration can predict "
    "enrollment velocity. Harder-to-screen protocols (more biomarker requirements, more "
    "exclusion criteria) enroll slower."
)

st.markdown(
    f"**Protocol Amendment Analysis** — The directional effects we quantified "
    f"({top_positive[0].replace('_', ' ')} +{top_positive[1]:.1f} mo/mention) "
    f"provide a baseline for measuring amendment impact. When an amendment adds three new biomarker "
    f"requirements, we can estimate the timeline cost."
)

st.markdown(
    "**Trial Costing Model** — Duration is the largest driver of trial cost. Our text-enriched "
    "duration model is a direct input to a costing model that accounts for protocol complexity, "
    "not just phase and disease area."
)


# ============================================================
# SECTION 6: Prioritized Opportunity Ranking
# ============================================================
st.divider()
st.header("Prioritized Ranking")
st.markdown("Combined score: **(Feasibility + Impact ×1.5 + Data Readiness) / 3.5**, weighted toward impact.")

opp_df = pd.DataFrame(OPPORTUNITIES)
opp_df["combined_score"] = (opp_df["feasibility"] + opp_df["impact"] * 1.5 + opp_df["data_readiness"]) / 3.5
opp_df = opp_df.sort_values("combined_score", ascending=False)

fig_rank = px.bar(
    opp_df, x="combined_score", y="use_case", orientation="h",
    color="time_to_value",
    color_discrete_map=TTV_COLORS,
    hover_data=["description", "feasibility", "impact", "data_readiness"],
    labels={"combined_score": "Priority Score", "use_case": "", "time_to_value": "Time to Value"},
)
fig_rank.update_layout(
    title="AI Opportunities Ranked by Priority",
    yaxis=dict(autorange="reversed"),
    height=500,
    **LAYOUT_DEFAULTS,
)
st.plotly_chart(fig_rank, use_container_width=True)


# --- Discovery to Production Pipeline ---
with st.expander("Discovery-to-Production Pipeline"):
    st.markdown(
        "Each opportunity follows a path from discovery through proof-of-concept to production deployment "
        "— the same pipeline this platform was built through."
    )

    PIPELINE_STAGES = [
        {
            "stage": "1. Discovery",
            "description": "Identify pain points through business team interviews and process observation",
            "gate": "Problem is specific, measurable, and affects >1 trial or function",
            "duration": "1-2 weeks",
            "icon": "\U0001F50D",
        },
        {
            "stage": "2. Evaluation",
            "description": "Assess data availability, technical feasibility, and business case",
            "gate": "Data exists and is accessible; ROI estimate > 2x development cost",
            "duration": "2-3 weeks",
            "icon": "\U0001F4CB",
        },
        {
            "stage": "3. Proof of Concept",
            "description": "Build minimal working prototype on representative data sample",
            "gate": "PoC demonstrates measurable improvement over current process",
            "duration": "4-8 weeks",
            "icon": "\U0001F6E0\uFE0F",
        },
        {
            "stage": "4. Validation",
            "description": "Test with real users on live (non-critical) workflows; gather feedback",
            "gate": "Users confirm value; error rate within acceptable bounds; IT security approved",
            "duration": "4-6 weeks",
            "icon": "\u2705",
        },
        {
            "stage": "5. Production",
            "description": "Deploy with monitoring, documentation, training, and feedback loops",
            "gate": "SLA defined; rollback plan in place; ongoing monitoring configured",
            "duration": "2-4 weeks",
            "icon": "\U0001F680",
        },
    ]

    for stage in PIPELINE_STAGES:
        with st.container():
            col_icon, col_content = st.columns([1, 10])
            with col_icon:
                st.markdown(f"### {stage['icon']}")
            with col_content:
                st.markdown(f"**{stage['stage']}** ({stage['duration']})")
                st.markdown(stage["description"])
                st.caption(f"Gate: {stage['gate']}")
        st.markdown("---")


# --- Detail Table ---
with st.expander("View Opportunity Details"):
    for _, row in opp_df.iterrows():
        st.markdown(f"**{row['use_case']}** — {row['business_function']} ({row['category']})")
        st.markdown(row["description"])
        dc1, dc2, dc3, dc4 = st.columns(4)
        dc1.metric("Feasibility", f"{row['feasibility']}/10")
        dc2.metric("Impact", f"{row['impact']}/10")
        dc3.metric("Data Readiness", f"{row['data_readiness']}/10")
        dc4.metric("Priority Score", f"{row['combined_score']:.1f}")
        if row.get("connection_to_investigation"):
            st.caption(f"*Investigation connection: {row['connection_to_investigation']}*")
        st.divider()


# --- Closing ---
st.divider()
st.markdown(
    f"This platform demonstrates one complete path through the pipeline — from {total_trials:,} raw "
    f"trial records through model comparison, NLP feature engineering, and risk profiling "
    f"to a prioritized AI roadmap. The same investigative approach applies to any "
    f"opportunity above."
)
