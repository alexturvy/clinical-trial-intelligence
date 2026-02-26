import json
import streamlit as st

try:
    from anthropic import Anthropic
except ImportError:
    Anthropic = None

ANALYSIS_PROMPT = """Analyze the following clinical trial eligibility criteria and return a JSON object with these fields:

1. "inclusion_count": number of inclusion criteria
2. "exclusion_count": number of exclusion criteria
3. "total_criteria": total number of criteria
4. "complexity_markers": object with boolean flags:
   - "prior_treatment_requirements": requires specific prior treatments
   - "biomarker_requirements": requires specific biomarker status
   - "comorbidity_exclusions": excludes based on comorbid conditions
   - "washout_periods": specifies washout periods for prior medications
   - "age_restrictions": has specific age requirements beyond standard adult
   - "organ_function_requirements": requires specific lab values or organ function
   - "genetic_requirements": requires genetic testing or specific mutations
5. "complexity_score": 1-10 score (10 = most complex) based on:
   - Number of criteria (more = more complex)
   - Specificity of requirements (biomarkers, genetics = high complexity)
   - Number of exclusion criteria relative to inclusion
   - Washout period requirements
6. "complexity_rationale": one-sentence explanation of the score
7. "readability_assessment": "low" | "medium" | "high" — how readable the criteria are for a site coordinator

Return ONLY valid JSON, no other text.

Eligibility Criteria:
{criteria_text}"""


def score_criteria(criteria_text: str) -> dict | None:
    """Score eligibility criteria using Claude API. Returns parsed JSON or None on failure."""
    if Anthropic is None:
        return None

    api_key = st.secrets.get("ANTHROPIC_API_KEY", "")
    if not api_key:
        return None

    client = Anthropic(api_key=api_key)
    message = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=1024,
        messages=[
            {"role": "user", "content": ANALYSIS_PROMPT.format(criteria_text=criteria_text[:8000])}
        ],
    )

    try:
        text = message.content[0].text
        # Strip markdown code fences if present
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except (json.JSONDecodeError, IndexError):
        return None
