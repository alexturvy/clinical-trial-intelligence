# Clinical Trial Intelligence Platform

A data-driven exploration of 18,000+ clinical trials across therapeutic areas central to full-service CRO operations — oncology, cardiology, metabolic disease, endocrinology, CNS, anti-viral, and anti-infective.

## Live Demo

[View the app on Streamlit Cloud](https://clinical-trial-intelligence.streamlit.app)

## Pages

### Trial Landscape
Interactive exploration of trial volume, geography, phases, sponsors, and enrollment. Filter by therapeutic area, phase, status, sponsor type, and date range.

### Enrollment & Duration Intelligence
Statistical analysis of trial timelines with a predictive model (scikit-learn) identifying features associated with longer trials.

### Protocol Complexity Analyzer
AI-powered analysis of eligibility criteria complexity using NLP (Claude API). Pre-scored dataset of 500 protocols plus a live scoring widget.

### AI Opportunity Framework
Strategic mapping of AI opportunities across CRO business functions — clinical ops, medical/safety, labs, BD, and business operations — with feasibility, impact, and data readiness scoring.

## Data

All trial data sourced from [ClinicalTrials.gov API v2](https://clinicaltrials.gov/data-api/api). Protocol complexity scores generated using Claude (Anthropic).

## Tech Stack

- Python, Streamlit
- pandas, Plotly, scikit-learn
- Claude API (Anthropic) for NLP
- ClinicalTrials.gov API v2

## Setup

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

To regenerate data:

```bash
python data/fetch_trials.py          # Fetch trials from ClinicalTrials.gov
ANTHROPIC_API_KEY=sk-... python data/score_protocols.py  # Score protocols with Claude
```

## License

MIT
