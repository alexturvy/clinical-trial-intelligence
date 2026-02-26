"""
Pre-compute all model results for Pages 2 & 3.
Run once offline: `python precompute.py`
Pages then load JSON/Parquet instead of training models on every visit.
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_score
from sklearn.manifold import TSNE

DATA_DIR = Path(__file__).parent / "data"

# ── Imports from project utils (no Streamlit dependency) ─────────────
sys.path.insert(0, str(Path(__file__).parent))
from utils.text_features import (
    extract_text_features,
    compute_tfidf_components,
    get_tfidf_component_labels,
    TEXT_FEATURE_NAMES,
    compute_tsne,
)


# ── Helpers ──────────────────────────────────────────────────────────

def _load_trials():
    path = DATA_DIR / "trials_processed.parquet"
    df = pd.read_parquet(path)
    if "start_date" in df.columns:
        df["start_date"] = pd.to_datetime(df["start_date"], errors="coerce")
    if "completion_date" in df.columns:
        df["completion_date"] = pd.to_datetime(df["completion_date"], errors="coerce")
    return df


def _load_protocols():
    path = DATA_DIR / "protocols_scored.parquet"
    if not path.exists():
        return None
    df = pd.read_parquet(path)
    if len(df) == 0:
        return None
    # Recover therapeutic_area if missing
    if "therapeutic_area" not in df.columns or df["therapeutic_area"].isna().sum() > len(df) * 0.5:
        trials_ta = pd.read_parquet(DATA_DIR / "trials_processed.parquet", columns=["nct_id", "therapeutic_area"])
        df = df.drop(columns=["therapeutic_area"], errors="ignore")
        df = df.merge(trials_ta, on="nct_id", how="left")
    return df


# Readable label mapping for TF-IDF components
TFIDF_LABEL_MAP = {
    "washout": "Washout Period Language",
    "wash-out": "Washout Period Language",
    "half-life": "Washout Period Language",
    "half life": "Washout Period Language",
    "biomarker": "Biomarker Terminology",
    "pd-l1": "Biomarker Terminology",
    "her2": "Biomarker Terminology",
    "egfr": "Biomarker Terminology",
    "mutation": "Genetic Testing Terms",
    "genetic": "Genetic Testing Terms",
    "genotype": "Genetic Testing Terms",
    "creatinine": "Organ Function Tests",
    "bilirubin": "Organ Function Tests",
    "hemoglobin": "Organ Function Tests",
    "hepatic": "Organ Function Tests",
    "platelet": "Lab Value Thresholds",
    "neutrophil": "Lab Value Thresholds",
    "ejection": "Cardiac Function Tests",
    "lvef": "Cardiac Function Tests",
    "pregnant": "Pregnancy / Reproductive",
    "pregnancy": "Pregnancy / Reproductive",
    "contraception": "Pregnancy / Reproductive",
    "diabetes": "Metabolic Conditions",
    "insulin": "Metabolic Conditions",
    "hba1c": "Metabolic Conditions",
    "tumor": "Tumor Characteristics",
    "malignant": "Tumor Characteristics",
    "metastatic": "Tumor Characteristics",
    "consent": "Consent & Capacity",
    "informed": "Consent & Capacity",
    "randomization": "Randomization Language",
    "randomized": "Randomization Language",
    "screening": "Screening & Enrollment",
    "baseline": "Screening & Enrollment",
    "visit": "Visit Schedule Language",
    "follow": "Follow-up Requirements",
    "infection": "Infection & Immunity",
    "hiv": "Infection & Immunity",
    "hepatitis": "Infection & Immunity",
    "renal": "Renal Function Tests",
    "kidney": "Renal Function Tests",
    "gfr": "Renal Function Tests",
}

# Clean labels for engineered features
ENGINEERED_LABEL_MAP = {
    "text_length": "Text Length",
    "word_count": "Word Count",
    "exclusion_mentions": "Exclusion Criteria Count",
    "inclusion_mentions": "Inclusion Criteria Count",
    "numbered_criteria": "Numbered Criteria",
    "biomarker_mentions": "Biomarker Terminology",
    "genetic_mentions": "Genetic Testing Terms",
    "washout_mentions": "Washout Period Language",
    "organ_fn_mentions": "Organ Function Tests",
    "lab_value_count": "Lab Value Thresholds",
    "time_constraint_count": "Time Constraints",
    "procedure_mentions": "Procedure Requirements",
    "prior_therapy_mentions": "Prior Therapy Restrictions",
    "comorbidity_exclusions": "Comorbidity Exclusions",
    "performance_status_req": "Performance Status Requirement",
    "contraception_req": "Contraception Requirement",
}


def _clean_tfidf_label(raw_label: str) -> str:
    """Convert 'tfidf_3 (washout/half-life/clearance)' to a readable name."""
    # Extract terms from parentheses
    if "(" in raw_label:
        terms_part = raw_label.split("(")[1].rstrip(")")
        terms = [t.strip() for t in terms_part.split("/")]
    else:
        return raw_label

    # Try to map any term to a readable label
    for term in terms:
        for keyword, label in TFIDF_LABEL_MAP.items():
            if keyword in term.lower():
                return label

    # Fallback: use the terms themselves, title-cased
    return " / ".join(t.title() for t in terms[:2]) + " Terms"


def _clean_feature_name(name: str) -> str:
    """Clean any feature name to human-readable form."""
    if name in ENGINEERED_LABEL_MAP:
        return ENGINEERED_LABEL_MAP[name]
    if name.startswith("tfidf_"):
        return name  # Will be handled separately with component labels
    # Category features from OneHotEncoder
    name = name.replace("therapeutic_area_", "TA: ")
    name = name.replace("phase_", "Phase: ")
    name = name.replace("sponsor_type_", "Sponsor: ")
    return name


# =====================================================================
# PAGE 2: Model Comparison
# =====================================================================

def precompute_page2():
    print("── Page 2: Model comparison ──")
    model_df = pd.read_parquet(DATA_DIR / "trials_processed.parquet")
    model_df["start_date"] = pd.to_datetime(model_df.get("start_date"), errors="coerce")
    model_df = model_df.dropna(
        subset=["duration_months", "enrollment", "phase", "therapeutic_area", "sponsor_type"]
    ).copy()
    model_df = model_df[model_df["duration_months"] > 0]
    model_df = model_df[model_df["duration_months"] <= model_df["duration_months"].quantile(0.99)]
    model_df = model_df[model_df["enrollment"] <= model_df["enrollment"].quantile(0.99)]
    model_df["num_countries"] = (
        model_df["countries"].fillna("").str.split("|").apply(lambda x: len([c for c in x if c]))
    )

    print(f"  Samples: {len(model_df)}")

    # Extract text features
    text_feats = extract_text_features(model_df)

    # TF-IDF components (fitted on FULL model_df — independent from Page 3)
    tfidf_df, tfidf_vec, svd_model = compute_tfidf_components(
        model_df["eligibility_criteria"], n_components=20
    )
    raw_comp_labels = get_tfidf_component_labels(tfidf_vec, svd_model, top_n=3)
    clean_comp_labels = [_clean_tfidf_label(lbl) for lbl in raw_comp_labels]

    model_df = pd.concat([
        model_df.reset_index(drop=True),
        text_feats.reset_index(drop=True),
        tfidf_df.reset_index(drop=True),
    ], axis=1)

    # ── Feature sets ──
    struct_features = ["enrollment", "num_countries", "phase", "therapeutic_area", "sponsor_type"]
    cat_features = ["phase", "therapeutic_area", "sponsor_type"]
    num_struct = ["enrollment", "num_countries"]
    tfidf_cols = [f"tfidf_{i}" for i in range(20)]
    text_feat_names = TEXT_FEATURE_NAMES + tfidf_cols
    num_all = num_struct + text_feat_names
    all_features = struct_features + text_feat_names

    y = model_df["duration_months"]

    # ── Preprocessors ──
    struct_prep = ColumnTransformer(transformers=[
        ("num", "passthrough", num_struct),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features),
    ])
    combined_prep = ColumnTransformer(transformers=[
        ("num", "passthrough", num_all),
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"), cat_features),
    ])
    text_only_prep = ColumnTransformer(transformers=[
        ("num", "passthrough", text_feat_names),
    ])

    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(n_estimators=50, max_depth=8, random_state=42, n_jobs=-1),
        "Gradient Boosting": GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42),
    }

    results = {"structured": {}, "text_enriched": {}, "text_only": {}}

    for name, model in models.items():
        print(f"  Training {name}...")

        # Structured only
        pipe_s = Pipeline([("prep", struct_prep), ("model", model)])
        scores_s = cross_val_score(pipe_s, model_df[struct_features], y, cv=5, scoring="r2")
        results["structured"][name] = {"mean": float(scores_s.mean()), "std": float(scores_s.std())}

        # Text enriched
        pipe_c = Pipeline([("prep", combined_prep), ("model", model)])
        scores_c = cross_val_score(pipe_c, model_df[all_features], y, cv=5, scoring="r2")
        results["text_enriched"][name] = {"mean": float(scores_c.mean()), "std": float(scores_c.std())}

        # Text only
        pipe_t = Pipeline([("prep", text_only_prep), ("model", model)])
        scores_t = cross_val_score(pipe_t, model_df[text_feat_names], y, cv=5, scoring="r2")
        results["text_only"][name] = {"mean": float(scores_t.mean()), "std": float(scores_t.std())}

    # ── Feature importance from best model ──
    print("  Computing feature importances...")
    best_pipe = Pipeline([
        ("prep", combined_prep),
        ("model", GradientBoostingRegressor(n_estimators=50, max_depth=4, random_state=42)),
    ])
    best_pipe.fit(model_df[all_features], y)
    cat_encoder = best_pipe.named_steps["prep"].named_transformers_["cat"]
    cat_names = cat_encoder.get_feature_names_out(cat_features).tolist()

    # Build readable feature names
    readable_num = []
    for f in num_all:
        if f.startswith("tfidf_"):
            idx = int(f.split("_")[1])
            readable_num.append(clean_comp_labels[idx] if idx < len(clean_comp_labels) else f)
        elif f in ENGINEERED_LABEL_MAP:
            readable_num.append(ENGINEERED_LABEL_MAP[f])
        else:
            readable_num.append(f)

    all_names = readable_num + [_clean_feature_name(n) for n in cat_names]
    importances = best_pipe.named_steps["model"].feature_importances_

    # Deduplicate: if multiple features map to same label, sum their importances
    feat_imp = {}
    for name_clean, imp in zip(all_names, importances):
        feat_imp[name_clean] = feat_imp.get(name_clean, 0) + float(imp)

    output = {
        "n_samples": len(model_df),
        "results": results,
        "feature_importances": feat_imp,
        "component_labels": clean_comp_labels,
    }

    out_path = DATA_DIR / "model_results.json"
    with open(out_path, "w") as f:
        json.dump(output, f, indent=2)
    print(f"  Saved → {out_path}")


# =====================================================================
# PAGE 3: Risk Model + t-SNE
# =====================================================================

def precompute_page3():
    print("── Page 3: Risk model + t-SNE ──")
    trials = _load_trials()
    model_df = pd.read_parquet(DATA_DIR / "trials_processed.parquet")
    model_df["start_date"] = pd.to_datetime(model_df.get("start_date"), errors="coerce")
    model_df = model_df.dropna(
        subset=["duration_months", "enrollment", "phase", "therapeutic_area", "sponsor_type"]
    ).copy()
    model_df = model_df[model_df["duration_months"] > 0]
    model_df = model_df[model_df["duration_months"] <= model_df["duration_months"].quantile(0.99)]
    model_df = model_df[model_df["enrollment"] <= model_df["enrollment"].quantile(0.99)]

    print(f"  Samples: {len(model_df)}")

    # Extract text features
    text_feats = extract_text_features(model_df)
    model_df = pd.concat([model_df.reset_index(drop=True), text_feats.reset_index(drop=True)], axis=1)

    # ── Baseline model: TA + phase only ──
    baseline_prep = ColumnTransformer(transformers=[
        ("cat", OneHotEncoder(drop="first", sparse_output=False, handle_unknown="ignore"),
         ["therapeutic_area", "phase"]),
    ])
    baseline_pipe = Pipeline([("prep", baseline_prep), ("model", LinearRegression())])
    baseline_pipe.fit(model_df[["therapeutic_area", "phase"]], model_df["duration_months"])
    model_df["baseline_pred"] = baseline_pipe.predict(model_df[["therapeutic_area", "phase"]])
    model_df["residual"] = model_df["duration_months"] - model_df["baseline_pred"]

    # ── Directional effects ──
    engineered_feats = [f for f in TEXT_FEATURE_NAMES if not f.startswith("tfidf_")]
    lr = LinearRegression()
    lr.fit(model_df[engineered_feats], model_df["residual"])
    directional_effects = {k: float(v) for k, v in zip(engineered_feats, lr.coef_)}

    # ── GB feature importance ──
    gb = GradientBoostingRegressor(n_estimators=100, max_depth=5, random_state=42)
    gb.fit(model_df[TEXT_FEATURE_NAMES], model_df["residual"])
    gb_importances = {k: float(v) for k, v in zip(TEXT_FEATURE_NAMES, gb.feature_importances_)}

    # ── Baselines by TA/phase ──
    baselines = model_df.groupby(["therapeutic_area", "phase"])["duration_months"].agg(["median", "count"]).reset_index()
    baselines_list = baselines.to_dict(orient="records")
    # Convert numpy types to native Python
    for row in baselines_list:
        row["median"] = float(row["median"])
        row["count"] = int(row["count"])

    risk_output = {
        "n_samples": len(model_df),
        "directional_effects": directional_effects,
        "gb_importances": gb_importances,
        "baselines": baselines_list,
        "residual_std": float(model_df["residual"].std()),
    }

    out_path = DATA_DIR / "risk_model_results.json"
    with open(out_path, "w") as f:
        json.dump(risk_output, f, indent=2)
    print(f"  Saved → {out_path}")

    # ── t-SNE on protocols (independent TF-IDF, fitted on protocols only) ──
    print("  Computing protocol t-SNE...")
    protocols = _load_protocols()
    if protocols is not None:
        proto_text_feats = extract_text_features(protocols)
        proto_tfidf_df, _, _ = compute_tfidf_components(protocols["eligibility_criteria"], n_components=20)

        combined = pd.concat([proto_text_feats.reset_index(drop=True), proto_tfidf_df.reset_index(drop=True)], axis=1)
        coords = compute_tsne(combined.values, perplexity=30)

        protocols = protocols.reset_index(drop=True)
        protocols["tsne_x"] = coords[:, 0]
        protocols["tsne_y"] = coords[:, 1]

        # Compute residuals for coloring
        ta_phase_medians = trials.dropna(subset=["duration_months"]).groupby(
            ["therapeutic_area", "phase"]
        )["duration_months"].median().to_dict()
        protocols["expected_duration"] = protocols.apply(
            lambda r: ta_phase_medians.get((r.get("therapeutic_area"), r.get("phase")), np.nan), axis=1
        )
        protocols["duration_residual"] = protocols["duration_months"] - protocols["expected_duration"]

        tsne_path = DATA_DIR / "protocol_tsne.parquet"
        protocols.to_parquet(tsne_path, index=False)
        print(f"  Saved → {tsne_path}")
    else:
        print("  Skipped t-SNE: no scored protocols found.")


# =====================================================================

if __name__ == "__main__":
    precompute_page2()
    precompute_page3()
    print("\nDone. Run `streamlit run Home.py` to verify.")
