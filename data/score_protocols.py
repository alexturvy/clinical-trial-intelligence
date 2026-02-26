#!/usr/bin/env python3
"""
Score protocol complexity for a sample of clinical trials using Claude API.
Run locally once to generate protocols_scored.parquet.
Requires ANTHROPIC_API_KEY environment variable.
"""

import json
import os
import time
from pathlib import Path

import pandas as pd
from anthropic import Anthropic

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
7. "readability_assessment": "low" | "medium" | "high"

Return ONLY valid JSON, no other text.

Eligibility Criteria:
{criteria_text}"""

SAMPLE_SIZE = 500


def score_one(client: Anthropic, criteria_text: str) -> dict | None:
    """Score a single protocol's eligibility criteria."""
    try:
        message = client.messages.create(
            model="claude-sonnet-4-20250514",
            max_tokens=1024,
            messages=[
                {"role": "user", "content": ANALYSIS_PROMPT.format(criteria_text=criteria_text[:8000])}
            ],
        )
        text = message.content[0].text
        if text.startswith("```"):
            text = text.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(text)
    except Exception as e:
        print(f"  Error scoring: {e}")
        return None


def main():
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        raise ValueError("Set ANTHROPIC_API_KEY environment variable")

    client = Anthropic(api_key=api_key)

    # Load trials
    trials_path = Path(__file__).parent / "trials_processed.parquet"
    df = pd.read_parquet(trials_path)

    # Sample trials that have eligibility criteria
    eligible = df[df["eligibility_criteria"].notna() & (df["eligibility_criteria"].str.len() > 100)].copy()
    print(f"Trials with eligibility criteria: {len(eligible)}")

    # Stratified sample by therapeutic area
    sample = eligible.groupby("therapeutic_area", group_keys=False).apply(
        lambda x: x.sample(min(len(x), SAMPLE_SIZE // 7), random_state=42),
        include_groups=False,
    )
    # If we didn't hit SAMPLE_SIZE, top up randomly
    if len(sample) < SAMPLE_SIZE:
        remaining = eligible[~eligible.index.isin(sample.index)]
        extra = remaining.sample(min(len(remaining), SAMPLE_SIZE - len(sample)), random_state=42)
        sample = pd.concat([sample, extra])

    print(f"Scoring {len(sample)} protocols...")

    results = []
    consecutive_failures = 0
    for i, (idx, row) in enumerate(sample.iterrows()):
        if (i + 1) % 25 == 0:
            print(f"  Progress: {i + 1}/{len(sample)} ({len(results)} scored)", flush=True)

        score = score_one(client, row["eligibility_criteria"])
        if score is None:
            consecutive_failures += 1
            if consecutive_failures >= 10:
                print(f"  Stopping early: {consecutive_failures} consecutive failures")
                break
            continue
        consecutive_failures = 0

        markers = score.get("complexity_markers", {})
        results.append({
            "nct_id": row["nct_id"],
            "title": row["title"],
            "therapeutic_area": row["therapeutic_area"],
            "phase": row["phase"],
            "enrollment": row["enrollment"],
            "duration_months": row.get("duration_months"),
            "status": row["status"],
            "inclusion_count": score.get("inclusion_count"),
            "exclusion_count": score.get("exclusion_count"),
            "total_criteria": score.get("total_criteria"),
            "complexity_score": score.get("complexity_score"),
            "complexity_rationale": score.get("complexity_rationale"),
            "readability": score.get("readability_assessment"),
            "has_prior_treatment_req": markers.get("prior_treatment_requirements", False),
            "has_biomarker_req": markers.get("biomarker_requirements", False),
            "has_comorbidity_exclusions": markers.get("comorbidity_exclusions", False),
            "has_washout_periods": markers.get("washout_periods", False),
            "has_age_restrictions": markers.get("age_restrictions", False),
            "has_organ_function_req": markers.get("organ_function_requirements", False),
            "has_genetic_req": markers.get("genetic_requirements", False),
            "eligibility_criteria": row["eligibility_criteria"][:5000],  # Truncate for storage
        })

        time.sleep(0.5)  # Rate limiting

    scored_df = pd.DataFrame(results)
    if len(scored_df) == 0:
        print("\nNo protocols were scored. Check your API key and credit balance.")
        return

    out_path = Path(__file__).parent / "protocols_scored.parquet"
    scored_df.to_parquet(out_path, index=False)
    print(f"\nSaved {len(scored_df)} scored protocols to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024:.0f} KB")
    print(f"Mean complexity score: {scored_df['complexity_score'].mean():.1f}")
    print(f"Score distribution:\n{scored_df['complexity_score'].describe()}")


if __name__ == "__main__":
    main()
