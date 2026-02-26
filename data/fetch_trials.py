#!/usr/bin/env python3
"""
Fetch clinical trials from ClinicalTrials.gov API v2.
Pulls trials across therapeutic areas relevant to full-service CROs.
Run locally once to generate trials_processed.parquet.
"""

import json
import time
from pathlib import Path

import pandas as pd
import requests

BASE_URL = "https://clinicaltrials.gov/api/v2/studies"

FIELDS = (
    "NCTId,BriefTitle,OfficialTitle,Phase,OverallStatus,EnrollmentCount,"
    "StartDate,PrimaryCompletionDate,CompletionDate,LeadSponsorName,"
    "Condition,LocationCountry,EligibilityCriteria"
)

# Therapeutic area queries — using terms that map to Medpace's focus areas
THERAPEUTIC_QUERIES = {
    "Oncology": "cancer OR neoplasm OR tumor OR oncology OR carcinoma OR lymphoma OR leukemia",
    "Cardiology": "cardiovascular OR heart failure OR cardiac OR coronary OR atrial fibrillation OR hypertension",
    "Metabolic Disease": "diabetes mellitus OR obesity OR metabolic syndrome OR dyslipidemia OR NASH OR NAFLD",
    "Endocrinology": "endocrine OR thyroid OR adrenal OR growth hormone OR osteoporosis OR hormonal",
    "CNS": "Alzheimer OR Parkinson OR epilepsy OR multiple sclerosis OR depression OR schizophrenia OR neuropathy",
    "Anti-Viral": "HIV OR hepatitis OR influenza OR antiviral OR COVID-19 OR RSV",
    "Anti-Infective": "bacterial infection OR antibiotic OR tuberculosis OR fungal infection OR sepsis OR antimicrobial",
}

# Limit per therapeutic area to keep total manageable (~3k each → ~20k total)
MAX_PER_TA = 3000


def fetch_studies(query_cond: str, max_results: int = MAX_PER_TA) -> list[dict]:
    """Fetch studies for a condition query, paginating through results."""
    studies = []
    params = {
        "format": "json",
        "pageSize": 1000,
        "query.cond": query_cond,
        "fields": FIELDS,
        "countTotal": "true",
        "sort": "StartDate:desc",
    }

    page = 0
    while len(studies) < max_results:
        page += 1
        resp = requests.get(BASE_URL, params=params, timeout=60)
        resp.raise_for_status()
        data = resp.json()

        batch = data.get("studies", [])
        if not batch:
            break

        studies.extend(batch)
        total = data.get("totalCount", "?")
        print(f"  Page {page}: fetched {len(batch)} studies (total so far: {len(studies)}, API total: {total})")

        next_token = data.get("nextPageToken")
        if not next_token or len(studies) >= max_results:
            break

        params["pageToken"] = next_token
        time.sleep(0.3)  # Be polite to the API

    return studies[:max_results]


def extract_record(study: dict, therapeutic_area: str) -> dict:
    """Flatten a study JSON into a flat record dict."""
    proto = study.get("protocolSection", {})
    ident = proto.get("identificationModule", {})
    status = proto.get("statusModule", {})
    design = proto.get("designModule", {})
    sponsor_mod = proto.get("sponsorCollaboratorsModule", {})
    conds = proto.get("conditionsModule", {})
    elig = proto.get("eligibilityModule", {})
    contacts = proto.get("contactsLocationsModule", {})

    # Phase — join if multiple
    phases = design.get("phases", [])
    phase_str = "/".join(p.replace("PHASE", "Phase ").replace("EARLY_", "Early ")
                         .replace("NA", "Not Applicable")
                         for p in phases) if phases else None

    # Enrollment
    enroll_info = design.get("enrollmentInfo", {})
    enrollment = enroll_info.get("count")

    # Dates
    start_raw = status.get("startDateStruct", {}).get("date")
    completion_raw = status.get("completionDateStruct", {}).get("date")
    primary_completion_raw = status.get("primaryCompletionDateStruct", {}).get("date")

    # Sponsor
    lead_sponsor = sponsor_mod.get("leadSponsor", {})
    sponsor_name = lead_sponsor.get("name")
    sponsor_class = lead_sponsor.get("class", "")  # INDUSTRY, NIH, FED, OTHER, NETWORK

    # Countries — deduplicate
    locations = contacts.get("locations", [])
    countries = sorted(set(loc.get("country", "") for loc in locations if loc.get("country")))

    # Conditions
    conditions = conds.get("conditions", [])

    return {
        "nct_id": ident.get("nctId"),
        "title": ident.get("briefTitle"),
        "official_title": ident.get("officialTitle"),
        "phase": phase_str,
        "status": status.get("overallStatus", "").replace("_", " ").title(),
        "enrollment": enrollment,
        "start_date": start_raw,
        "completion_date": completion_raw,
        "primary_completion_date": primary_completion_raw,
        "sponsor": sponsor_name,
        "sponsor_type": "Industry" if sponsor_class == "INDUSTRY" else (
            "Government" if sponsor_class in ("NIH", "FED") else "Academic/Other"
        ),
        "conditions": "|".join(conditions),
        "countries": "|".join(countries),
        "therapeutic_area": therapeutic_area,
        "eligibility_criteria": elig.get("eligibilityCriteria"),
    }


def compute_duration(df: pd.DataFrame) -> pd.DataFrame:
    """Add duration_months column from start_date to completion_date."""
    df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    df["primary_completion_date"] = pd.to_datetime(df["primary_completion_date"], errors="coerce")

    mask = df["start_date"].notna() & df["completion_date"].notna()
    df.loc[mask, "duration_months"] = (
        (df.loc[mask, "completion_date"] - df.loc[mask, "start_date"]).dt.days / 30.44
    ).round(1)

    return df


def main():
    all_records = []
    seen_nct_ids = set()

    for ta, query in THERAPEUTIC_QUERIES.items():
        print(f"\nFetching {ta}...")
        studies = fetch_studies(query)
        print(f"  Got {len(studies)} raw studies")

        for study in studies:
            record = extract_record(study, ta)
            nct_id = record["nct_id"]
            if nct_id and nct_id not in seen_nct_ids:
                seen_nct_ids.add(nct_id)
                all_records.append(record)

    print(f"\nTotal unique trials: {len(all_records)}")

    df = pd.DataFrame(all_records)
    df = compute_duration(df)

    # Basic cleaning
    df["enrollment"] = pd.to_numeric(df["enrollment"], errors="coerce")
    df = df.dropna(subset=["nct_id"])

    # Save
    out_path = Path(__file__).parent / "trials_processed.parquet"
    df.to_parquet(out_path, index=False)
    print(f"Saved {len(df)} trials to {out_path}")
    print(f"File size: {out_path.stat().st_size / 1024 / 1024:.1f} MB")

    # Summary stats
    print(f"\nTherapeutic areas: {df['therapeutic_area'].value_counts().to_dict()}")
    print(f"Phases: {df['phase'].value_counts().to_dict()}")
    print(f"Statuses: {df['status'].value_counts().head(5).to_dict()}")


if __name__ == "__main__":
    main()
