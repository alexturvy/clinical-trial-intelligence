"""
Shared CSS injection for the Clinical Trial Intelligence Platform.

Design system: Editorial data journalism aesthetic.
- Newsreader (serif) for headers — editorial authority
- DM Sans (sans-serif) for body — clean readability
- JetBrains Mono for code/methods — technical credibility
"""

import streamlit as st


_CSS = """
<style>
/* ================================================================
   FONTS
   ================================================================ */
@import url('https://fonts.googleapis.com/css2?family=Newsreader:ital,opsz,wght@0,6..72,400;0,6..72,500;0,6..72,600;0,6..72,700;1,6..72,400;1,6..72,500&family=DM+Sans:ital,opsz,wght@0,9..40,400;0,9..40,500;0,9..40,600;0,9..40,700;1,9..40,400&family=JetBrains+Mono:wght@400;500&display=swap');

/* ================================================================
   GLOBAL TYPOGRAPHY
   ================================================================ */
html, body, [class*="css"], .stMarkdown, p, li {
    font-family: 'DM Sans', sans-serif !important;
    -webkit-font-smoothing: antialiased;
}

/* Preserve Material Icons/Symbols font for Streamlit icons */
[data-testid="stIconMaterial"],
.material-symbols-rounded,
[class*="material"] {
    font-family: 'Material Symbols Rounded' !important;
}

/* ================================================================
   HEADER HIERARCHY
   ================================================================ */
h1 {
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 600 !important;
    letter-spacing: -0.025em !important;
    line-height: 1.15 !important;
    color: #1A1A2E !important;
    margin-bottom: 0.3rem !important;
}

h2 {
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 500 !important;
    font-size: 1.6rem !important;
    letter-spacing: -0.02em !important;
    line-height: 1.25 !important;
    color: #1A1A2E !important;
    margin-top: 0.5rem !important;
}

h3 {
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 500 !important;
    font-size: 1.25rem !important;
    letter-spacing: -0.01em !important;
    color: #2c3e50 !important;
}

/* Subheaders (##### in markdown → h5) used for card titles */
h5 {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.95rem !important;
    line-height: 1.35 !important;
    color: #1A1A2E !important;
    letter-spacing: -0.01em !important;
}

/* ================================================================
   BLOCKQUOTES — Hero insight styling
   ================================================================ */
blockquote {
    border-left: 3px solid #1B6B93 !important;
    background: linear-gradient(135deg, rgba(27,107,147,0.05) 0%, rgba(42,157,143,0.03) 100%) !important;
    padding: 1.25rem 1.5rem !important;
    border-radius: 0 8px 8px 0 !important;
    margin: 0.75rem 0 1.25rem 0 !important;
}

blockquote p {
    font-size: 1.02rem !important;
    line-height: 1.65 !important;
    color: #2c3e50 !important;
}

/* ================================================================
   CODE / METHODS BADGES
   ================================================================ */
code {
    font-family: 'JetBrains Mono', 'Fira Code', monospace !important;
    font-size: 0.78rem !important;
    background: rgba(27,107,147,0.07) !important;
    color: #1B6B93 !important;
    padding: 0.15rem 0.45rem !important;
    border-radius: 4px !important;
    font-weight: 500 !important;
    letter-spacing: 0 !important;
    border: 1px solid rgba(27,107,147,0.1) !important;
}

/* ================================================================
   METRICS
   ================================================================ */
[data-testid="stMetricValue"] {
    font-family: 'Newsreader', Georgia, serif !important;
    font-weight: 600 !important;
    font-size: 1.9rem !important;
    color: #1B6B93 !important;
    letter-spacing: -0.02em !important;
}

[data-testid="stMetricLabel"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    text-transform: uppercase !important;
    font-size: 0.7rem !important;
    letter-spacing: 0.06em !important;
    color: #6C757D !important;
}

/* ================================================================
   DIVIDERS — Subtle gradient fade
   ================================================================ */
hr {
    border: none !important;
    height: 1px !important;
    background: linear-gradient(90deg, transparent 0%, #c8d6e0 20%, #c8d6e0 80%, transparent 100%) !important;
    margin: 2.25rem 0 !important;
    opacity: 0.7 !important;
}

/* ================================================================
   CAPTIONS — Methods/toolkit lines
   ================================================================ */
[data-testid="stCaptionContainer"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #8899a6 !important;
    letter-spacing: 0.01em !important;
}

/* ================================================================
   EXPANDERS — Cleaner look
   ================================================================ */
[data-testid="stExpander"] {
    border: 1px solid #e4eaf0 !important;
    border-radius: 8px !important;
    background: #fbfcfe !important;
}

[data-testid="stExpander"] summary {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    font-size: 0.92rem !important;
    color: #1A1A2E !important;
}

/* ================================================================
   SIDEBAR
   ================================================================ */
[data-testid="stSidebar"] {
    background: #f6f9fc !important;
}

[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.08em !important;
    color: #6C757D !important;
}

/* ================================================================
   PAGE LINKS — CTA buttons
   ================================================================ */
[data-testid="stPageLink-NavLink"] {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border: 1px solid #1B6B93 !important;
    border-radius: 6px !important;
    transition: all 0.15s ease !important;
}

/* ================================================================
   MARKDOWN BODY TEXT
   ================================================================ */
[data-testid="stMarkdown"] p {
    line-height: 1.65 !important;
    color: #2c3e50 !important;
}

[data-testid="stMarkdown"] li {
    line-height: 1.55 !important;
    color: #2c3e50 !important;
    margin-bottom: 0.3rem !important;
}

[data-testid="stMarkdown"] strong {
    color: #1A1A2E !important;
}

/* ================================================================
   RADIO BUTTONS — Horizontal toggle
   ================================================================ */
[data-testid="stRadio"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
}

/* ================================================================
   SELECTBOX
   ================================================================ */
[data-testid="stSelectbox"] label {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 500 !important;
    font-size: 0.88rem !important;
}

/* ================================================================
   BUTTONS
   ================================================================ */
.stButton > button {
    font-family: 'DM Sans', sans-serif !important;
    font-weight: 600 !important;
    border-radius: 6px !important;
    letter-spacing: 0.01em !important;
}

/* ================================================================
   CUSTOM COMPONENTS
   ================================================================ */

/* Hero title — used on Home page */
.hero-title {
    font-family: 'Newsreader', Georgia, serif !important;
    font-size: 2.4rem !important;
    line-height: 1.18 !important;
    font-weight: 600 !important;
    letter-spacing: -0.03em !important;
    color: #1A1A2E !important;
    margin-bottom: 0.5rem !important;
}

.hero-title .subdued {
    color: #5a6c7d !important;
    font-weight: 400 !important;
}

/* Section label — small uppercase label above headers */
.section-label {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    text-transform: uppercase !important;
    letter-spacing: 0.1em !important;
    color: #1B6B93 !important;
    margin-bottom: -0.5rem !important;
    display: block !important;
}

/* Card container styling */
.card-container {
    background: #f8fafe;
    border: 1px solid #e4eaf0;
    border-radius: 10px;
    padding: 1.4rem 1.2rem;
    min-height: 280px;
    transition: border-color 0.2s ease, box-shadow 0.2s ease;
}

.card-container:hover {
    border-color: #1B6B93;
    box-shadow: 0 2px 12px rgba(27,107,147,0.08);
}

.card-metric {
    font-family: 'Newsreader', Georgia, serif;
    font-size: 1.75rem;
    font-weight: 600;
    color: #1B6B93;
    letter-spacing: -0.02em;
    margin: 0.6rem 0 0.4rem 0;
    white-space: nowrap;
}

.card-title {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem;
    font-weight: 600;
    color: #1A1A2E;
    line-height: 1.35;
    margin-bottom: 0.4rem;
}

.card-description {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.83rem;
    color: #5a6c7d;
    line-height: 1.5;
    margin-bottom: 0.5rem;
}

.card-tech {
    font-family: 'JetBrains Mono', monospace;
    font-size: 0.72rem;
    color: #1B6B93;
    background: rgba(27,107,147,0.07);
    border: 1px solid rgba(27,107,147,0.1);
    border-radius: 4px;
    padding: 0.15rem 0.45rem;
    display: inline-block;
    margin-bottom: 0.7rem;
}

/* Connector (page transition) styling */
.connector {
    background: linear-gradient(135deg, rgba(27,107,147,0.04) 0%, rgba(42,157,143,0.04) 100%);
    border: 1px solid #e4eaf0;
    border-radius: 8px;
    padding: 1rem 1.5rem;
    margin-top: 0.5rem;
}

.connector p {
    margin: 0 !important;
    color: #2c3e50 !important;
}

/* Risk level badges */
.risk-high { color: #E63946; }
.risk-elevated { color: #F4A261; }
.risk-moderate { color: #1B6B93; }
.risk-low { color: #2A9D8F; }

/* Footer / data source */
.data-source {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: #8899a6;
    border-top: 1px solid #e4eaf0;
    padding-top: 1rem;
    margin-top: 1rem;
}

/* Byline — top right (inner pages) */
.byline {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.78rem;
    color: #8899a6;
    text-align: right;
    letter-spacing: 0.01em;
}

/* Byline — home page (larger) */
.byline-home {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.92rem;
    color: #5a6c7d;
    text-align: right;
    letter-spacing: 0.01em;
    font-weight: 500;
}

/* ================================================================
   SPACING & RHYTHM
   ================================================================ */

/* Add breathing room between major blocks */
.block-container {
    padding-top: 2rem !important;
    max-width: 1100px !important;
}

/* Tighten gap between section label and header */
[data-testid="stMarkdown"] + [data-testid="stHeading"] {
    margin-top: -0.25rem !important;
}

</style>
"""


def inject_custom_css():
    """Inject the platform's custom CSS. Call once at the top of each page."""
    st.markdown(_CSS, unsafe_allow_html=True)


def section_label(text: str):
    """Render a small uppercase label above a section header."""
    st.markdown(f'<span class="section-label">{text}</span>', unsafe_allow_html=True)


def hero_title(main: str, subdued: str):
    """Render the Home page hero title with a subdued second line."""
    st.markdown(
        f'<h1 class="hero-title">{main}<br>'
        f'<span class="subdued">{subdued}</span></h1>',
        unsafe_allow_html=True,
    )


def byline(home: bool = False):
    """Render 'Built by Alex Turvy, PhD' top-right. Larger on home page."""
    cls = "byline-home" if home else "byline"
    st.markdown(
        f'<p class="{cls}">Built by Alex Turvy, PhD</p>',
        unsafe_allow_html=True,
    )


def card_html(title: str, metric_value: str, description: str, tech: str) -> str:
    """Return HTML for a single card. Use inside st.markdown(unsafe_allow_html=True)."""
    return (
        f'<div class="card-container">'
        f'<div class="card-title">{title}</div>'
        f'<div class="card-metric">{metric_value}</div>'
        f'<div class="card-description">{description}</div>'
        f'<div class="card-tech">{tech}</div>'
        f'</div>'
    )
