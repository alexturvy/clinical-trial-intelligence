import plotly.express as px
import plotly.graph_objects as go
import pandas as pd

# Consistent color palette — medical blue theme
COLORS = {
    "primary": "#1B6B93",
    "secondary": "#4FC0D0",
    "accent": "#A2D2FF",
    "warm": "#FF6B6B",
    "neutral": "#6C757D",
}

TA_COLORS = {
    "Oncology": "#E63946",
    "Cardiology": "#1B6B93",
    "Metabolic Disease": "#F4A261",
    "Endocrinology": "#2A9D8F",
    "CNS": "#7B2D8E",
    "Anti-Viral": "#457B9D",
    "Anti-Infective": "#E9C46A",
}

PHASE_ORDER = ["Early Phase 1", "Phase 1", "Phase 1/Phase 2", "Phase 2",
               "Phase 2/Phase 3", "Phase 3", "Phase 4", "Not Applicable"]

LAYOUT_DEFAULTS = dict(
    font=dict(family="'DM Sans', sans-serif", size=12, color="#2c3e50"),
    title_font=dict(family="'Newsreader', Georgia, serif", size=16, color="#1A1A2E"),
    plot_bgcolor="rgba(0,0,0,0)",
    paper_bgcolor="rgba(0,0,0,0)",
    margin=dict(l=40, r=20, t=50, b=40),
    hoverlabel=dict(bgcolor="white", font_size=12, font_family="'DM Sans', sans-serif"),
)


def trial_volume_trend(df: pd.DataFrame) -> go.Figure:
    """Horizontal bar chart of trial counts by therapeutic area."""
    ta_counts = (
        df["therapeutic_area"]
        .value_counts()
        .sort_values(ascending=True)
        .reset_index()
    )
    ta_counts.columns = ["therapeutic_area", "count"]

    fig = px.bar(
        ta_counts, x="count", y="therapeutic_area", orientation="h",
        color="therapeutic_area",
        color_discrete_map=TA_COLORS,
        labels={"count": "Number of Trials", "therapeutic_area": ""},
    )
    fig.update_layout(
        title="Trial Volume by Therapeutic Area",
        showlegend=False,
        **LAYOUT_DEFAULTS,
    )
    return fig


def phase_distribution(df: pd.DataFrame) -> tuple[go.Figure, int]:
    """Bar chart of trial counts by phase. Excludes 'Not Applicable'. Returns (fig, na_count)."""
    na_count = int((df["phase"] == "Not Applicable").sum())
    phase_df = df[df["phase"] != "Not Applicable"]
    order = [p for p in PHASE_ORDER if p != "Not Applicable"]

    phase_counts = phase_df["phase"].value_counts().reindex(order).dropna().reset_index()
    phase_counts.columns = ["phase", "count"]

    fig = px.bar(
        phase_counts, x="phase", y="count",
        color_discrete_sequence=[COLORS["primary"]],
        labels={"phase": "Phase", "count": "Number of Trials"},
    )
    fig.update_layout(title="Distribution by Trial Phase", showlegend=False, **LAYOUT_DEFAULTS)
    return fig, na_count


def geographic_map(df: pd.DataFrame) -> go.Figure:
    """Choropleth map of trial counts by country."""
    countries = df.dropna(subset=["countries"]).copy()
    # Explode countries list into individual rows
    countries = countries.assign(country=countries["countries"].str.split("|")).explode("country")
    country_counts = countries["country"].value_counts().reset_index()
    country_counts.columns = ["country", "count"]

    fig = px.choropleth(
        country_counts, locations="country", locationmode="country names",
        color="count", color_continuous_scale="Blues",
        labels={"count": "Number of Trials", "country": "Country"},
    )
    fig.update_layout(title="Global Trial Distribution", geo=dict(showframe=False), **LAYOUT_DEFAULTS)
    fig.update_layout(margin=dict(l=0, r=0, t=50, b=0))
    return fig


def enrollment_histogram(df: pd.DataFrame) -> go.Figure:
    """Bar chart of enrollment counts using clinically meaningful bins."""
    enroll = df["enrollment"].dropna()
    enroll = enroll[enroll > 0]

    bins = [0, 50, 100, 250, 500, 1000, 5000, float("inf")]
    labels = ["1–50", "51–100", "101–250", "251–500", "501–1,000", "1,001–5,000", "5,000+"]
    binned = pd.cut(enroll, bins=bins, labels=labels, right=True)
    bin_counts = binned.value_counts().reindex(labels).fillna(0).reset_index()
    bin_counts.columns = ["Enrollment Range", "Number of Trials"]

    fig = px.bar(
        bin_counts, x="Enrollment Range", y="Number of Trials",
        color_discrete_sequence=[COLORS["secondary"]],
    )
    fig.update_layout(title="Enrollment Distribution", showlegend=False, **LAYOUT_DEFAULTS)
    return fig


def sponsor_treemap(df: pd.DataFrame, top_n: int = 20) -> go.Figure:
    """Treemap of top sponsors by trial count."""
    sponsor_counts = df["sponsor"].value_counts().head(top_n).reset_index()
    sponsor_counts.columns = ["sponsor", "count"]

    fig = px.treemap(
        sponsor_counts, path=["sponsor"], values="count",
        color="count", color_continuous_scale="Blues",
    )
    fig.update_layout(title=f"Top {top_n} Sponsors by Trial Count", **LAYOUT_DEFAULTS)
    return fig


def duration_by_phase(df: pd.DataFrame) -> go.Figure:
    """Box plot of trial duration by phase."""
    dur = df.dropna(subset=["duration_months"]).copy()
    dur = dur[dur["duration_months"] > 0]
    dur = dur[dur["duration_months"] <= dur["duration_months"].quantile(0.99)]

    fig = px.box(
        dur, x="phase", y="duration_months", color="phase",
        category_orders={"phase": PHASE_ORDER},
        labels={"phase": "Phase", "duration_months": "Duration (Months)"},
    )
    fig.update_layout(title="Trial Duration by Phase", showlegend=False, **LAYOUT_DEFAULTS)
    return fig


def duration_by_ta(df: pd.DataFrame) -> go.Figure:
    """Box plot of trial duration by therapeutic area."""
    dur = df.dropna(subset=["duration_months"]).copy()
    dur = dur[dur["duration_months"] > 0]
    dur = dur[dur["duration_months"] <= dur["duration_months"].quantile(0.99)]

    fig = px.box(
        dur, x="therapeutic_area", y="duration_months", color="therapeutic_area",
        color_discrete_map=TA_COLORS,
        labels={"therapeutic_area": "Therapeutic Area", "duration_months": "Duration (Months)"},
    )
    fig.update_layout(title="Trial Duration by Therapeutic Area", showlegend=False, **LAYOUT_DEFAULTS)
    return fig


def feature_importance_chart(feature_names: list, importances: list) -> go.Figure:
    """Horizontal bar chart of feature importances from regression model."""
    fi = pd.DataFrame({"feature": feature_names, "importance": importances})
    fi = fi.sort_values("importance", ascending=True)

    fig = px.bar(
        fi, x="importance", y="feature", orientation="h",
        color_discrete_sequence=[COLORS["primary"]],
        labels={"importance": "Coefficient Magnitude", "feature": "Feature"},
    )
    fig.update_layout(title="What Predicts Longer Trials?", showlegend=False, **LAYOUT_DEFAULTS)
    return fig


def complexity_vs_duration(df: pd.DataFrame) -> go.Figure:
    """Scatter plot of protocol complexity score vs trial duration."""
    plot_df = df.dropna(subset=["complexity_score", "duration_months"]).copy()
    plot_df = plot_df[plot_df["duration_months"] > 0]

    fig = px.scatter(
        plot_df, x="complexity_score", y="duration_months",
        color="therapeutic_area", color_discrete_map=TA_COLORS,
        size="enrollment", size_max=15,
        hover_data=["nct_id", "title"],
        labels={
            "complexity_score": "Protocol Complexity Score",
            "duration_months": "Duration (Months)",
            "therapeutic_area": "Therapeutic Area",
        },
    )
    fig.update_layout(title="Protocol Complexity vs. Trial Duration", **LAYOUT_DEFAULTS)
    return fig


def complexity_by_ta(df: pd.DataFrame) -> go.Figure:
    """Box plot of complexity score by therapeutic area."""
    fig = px.box(
        df.dropna(subset=["complexity_score"]),
        x="therapeutic_area", y="complexity_score",
        color="therapeutic_area", color_discrete_map=TA_COLORS,
        labels={"therapeutic_area": "Therapeutic Area", "complexity_score": "Complexity Score"},
    )
    fig.update_layout(title="Protocol Complexity by Therapeutic Area", showlegend=False, **LAYOUT_DEFAULTS)
    return fig


TTV_COLORS = {
    "3-6 months": "#2A9D8F",
    "6-12 months": "#1B6B93",
    "12-18 months": "#F4A261",
    "18+ months": "#E63946",
}


def opportunity_matrix(data: list[dict]) -> go.Figure:
    """Bubble chart for AI opportunity scoring, colored by time to value."""
    odf = pd.DataFrame(data)

    # Ensure time_to_value exists (backward compat)
    if "time_to_value" not in odf.columns:
        odf["time_to_value"] = "6-12 months"

    fig = px.scatter(
        odf, x="feasibility", y="impact",
        size="data_readiness", color="time_to_value",
        color_discrete_map=TTV_COLORS,
        hover_name="use_case",
        custom_data=["business_function", "data_readiness", "time_to_value"],
        labels={
            "feasibility": "Feasibility Score",
            "impact": "Business Impact Score",
            "data_readiness": "Data Readiness",
            "time_to_value": "Time to Value",
        },
        size_max=20,
        category_orders={"time_to_value": ["3-6 months", "6-12 months", "12-18 months", "18+ months"]},
    )
    fig.update_traces(
        hovertemplate=(
            "<b>%{hovertext}</b><br>"
            "Business Function: %{customdata[0]}<br>"
            "Feasibility: %{x}/10 · Impact: %{y}/10<br>"
            "Data Readiness: %{customdata[1]}/10<br>"
            "Time to Value: %{customdata[2]}"
            "<extra></extra>"
        ),
    )
    fig.update_layout(
        title="AI Opportunity Matrix: Feasibility vs. Impact",
        xaxis=dict(range=[0, 10.5]),
        yaxis=dict(range=[0, 10.5]),
        **LAYOUT_DEFAULTS,
    )
    # Quadrant lines
    fig.add_hline(y=5, line_dash="dash", line_color="gray", opacity=0.5)
    fig.add_vline(x=5, line_dash="dash", line_color="gray", opacity=0.5)
    # Quadrant labels
    fig.add_annotation(x=8, y=10.2, text="Quick Wins", showarrow=False,
                       font=dict(size=14, color="gray"), opacity=0.6)
    fig.add_annotation(x=2.5, y=10.2, text="Strategic Bets", showarrow=False,
                       font=dict(size=14, color="gray"), opacity=0.6)
    fig.add_annotation(x=2.5, y=0.5, text="Low Priority", showarrow=False,
                       font=dict(size=14, color="gray"), opacity=0.6)
    fig.add_annotation(x=8, y=0.5, text="Feasible but Low Impact", showarrow=False,
                       font=dict(size=14, color="gray"), opacity=0.6)
    return fig
