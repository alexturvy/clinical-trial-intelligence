"""
NLP feature extraction from clinical trial eligibility criteria.
TF-IDF vectorization, domain keyword extraction, dimensionality reduction, and clustering.
"""

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE

# --- Domain keyword dictionaries ---

BIOMARKER_TERMS = [
    "biomarker", "pd-l1", "her2", "egfr", "brca", "kras", "braf",
    "alk", "ros1", "msi", "mmr", "ctdna", "cea", "psa", "ca-125",
    "tumor mutational burden", "tmb", "microsatellite",
    "immunohistochemistry", "ihc", "fish", "pcr", "ngs", "sequencing",
]

GENETIC_TERMS = [
    "genetic", "mutation", "variant", "allele", "genotype", "polymorphism",
    "chromosom", "translocation", "deletion", "amplification", "fusion",
    "germline", "somatic", "wild-type", "wild type", "heterozygous",
    "homozygous", "carrier",
]

WASHOUT_TERMS = [
    "washout", "wash-out", "half-life", "half life", "clearance period",
    "discontinue", "discontinued", "prior to enrollment",
    "before randomization", "days before", "weeks before", "months before",
]

ORGAN_FUNCTION_TERMS = [
    "creatinine", "bilirubin", "ast", "alt", "alkaline phosphatase",
    "hemoglobin", "platelet", "neutrophil", "anc", "inr",
    "egfr", "gfr", "ejection fraction", "lvef", "fev1", "dlco",
    "liver function", "renal function", "hepatic", "cardiac function",
    "bone marrow", "hematologic",
]

PROCEDURE_TERMS = [
    "biopsy", "biopsies", "blood draw", "venipuncture", "lumbar puncture",
    "bone marrow aspirat", "endoscopy", "colonoscopy", "bronchoscopy",
    "imaging", "mri", "ct scan", "pet scan", "x-ray", "ultrasound",
    "echocardiogram", "mammogram", "electrocardiogram", "ekg", "ecg",
]

PRIOR_THERAPY_TERMS = [
    "prior therapy", "prior treatment", "prior regimen", "previous therapy",
    "previous treatment", "first-line", "first line", "second-line",
    "second line", "third-line", "third line", "lines of therapy",
    "line of therapy", "prior systemic", "prior chemotherapy",
    "prior immunotherapy", "prior radiation",
]

COMORBIDITY_EXCL_TERMS = [
    "history of cancer", "history of malignancy", "prior malignancy",
    "history of cardiac", "history of hepatic", "history of renal",
    "history of stroke", "history of seizure", "history of psychiatric",
    "autoimmune disease", "active infection", "uncontrolled diabetes",
    "uncontrolled hypertension", "organ transplant",
]


def _count_keywords(text: str, keywords: list[str]) -> int:
    text_lower = text.lower()
    return sum(1 for kw in keywords if kw in text_lower)


def extract_text_features(df: pd.DataFrame, text_col: str = "eligibility_criteria") -> pd.DataFrame:
    """Extract NLP features from eligibility criteria text. Returns a DataFrame aligned to df's index."""
    text = df[text_col].fillna("")

    features = pd.DataFrame(index=df.index)
    features["text_length"] = text.str.len()
    features["word_count"] = text.str.split().str.len().fillna(0).astype(int)

    # Criteria structure
    features["exclusion_mentions"] = text.str.lower().str.count(r"exclusion|exclude|ineligible").fillna(0).astype(int)
    features["inclusion_mentions"] = text.str.lower().str.count(r"inclusion|include|eligible").fillna(0).astype(int)
    features["numbered_criteria"] = text.str.count(r"\n\s*\d+[.)\s]").fillna(0).astype(int)

    # Domain keyword counts
    features["biomarker_mentions"] = text.apply(lambda t: _count_keywords(t, BIOMARKER_TERMS))
    features["genetic_mentions"] = text.apply(lambda t: _count_keywords(t, GENETIC_TERMS))
    features["washout_mentions"] = text.apply(lambda t: _count_keywords(t, WASHOUT_TERMS))
    features["organ_fn_mentions"] = text.apply(lambda t: _count_keywords(t, ORGAN_FUNCTION_TERMS))

    # Specificity indicators
    features["lab_value_count"] = text.str.count(r"[<>≤≥]\s*\d+").fillna(0).astype(int)
    features["time_constraint_count"] = text.str.count(
        r"\d+\s*(days?|weeks?|months?|years?)\s*(before|after|prior|within)"
    ).fillna(0).astype(int)

    # Procedural complexity
    features["procedure_mentions"] = text.apply(lambda t: _count_keywords(t, PROCEDURE_TERMS))
    features["prior_therapy_mentions"] = text.apply(lambda t: _count_keywords(t, PRIOR_THERAPY_TERMS))
    features["comorbidity_exclusions"] = text.apply(lambda t: _count_keywords(t, COMORBIDITY_EXCL_TERMS))

    # Performance status (ECOG/Karnofsky — indicates how sick patients must be)
    features["performance_status_req"] = text.str.contains(
        r"ecog|karnofsky|who performance|eastern cooperative", case=False, regex=True
    ).astype(int)

    # Contraception requirements (adds screening burden)
    features["contraception_req"] = text.str.contains(
        r"contraception|contraceptive|childbearing potential|fertile", case=False, regex=True
    ).astype(int)

    return features


TEXT_FEATURE_NAMES = [
    "text_length", "word_count", "exclusion_mentions", "inclusion_mentions",
    "numbered_criteria", "biomarker_mentions", "genetic_mentions",
    "washout_mentions", "organ_fn_mentions", "lab_value_count", "time_constraint_count",
    "procedure_mentions", "prior_therapy_mentions", "comorbidity_exclusions",
    "performance_status_req", "contraception_req",
]


def compute_tfidf_components(
    texts: pd.Series, n_components: int = 20
) -> tuple[pd.DataFrame, TfidfVectorizer, TruncatedSVD]:
    """TF-IDF + TruncatedSVD on eligibility text. Returns (component_df, vectorizer, svd)."""
    texts_clean = texts.fillna("").astype(str)

    tfidf = TfidfVectorizer(
        max_features=1000,
        stop_words="english",
        ngram_range=(1, 2),
        min_df=5,
        max_df=0.8,
    )
    tfidf_matrix = tfidf.fit_transform(texts_clean)

    svd = TruncatedSVD(n_components=n_components, random_state=42)
    components = svd.fit_transform(tfidf_matrix)

    comp_df = pd.DataFrame(
        components,
        columns=[f"tfidf_{i}" for i in range(n_components)],
        index=texts.index,
    )
    return comp_df, tfidf, svd


def get_tfidf_component_labels(tfidf: TfidfVectorizer, svd: TruncatedSVD, top_n: int = 3) -> list[str]:
    """Map each SVD component to its top loading terms for readable feature names."""
    terms = tfidf.get_feature_names_out()
    labels = []
    for i, comp in enumerate(svd.components_):
        top_idx = comp.argsort()[-top_n:][::-1]
        top_terms = "/".join(terms[j] for j in top_idx)
        labels.append(f"tfidf_{i} ({top_terms})")
    return labels


def cluster_protocols(
    tfidf_matrix, n_clusters: int = 5
) -> tuple[np.ndarray, KMeans]:
    """K-Means clustering on TF-IDF/SVD components. Returns (labels, model)."""
    km = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    labels = km.fit_predict(tfidf_matrix)
    return labels, km


def compute_tsne(components: np.ndarray, perplexity: int = 30) -> np.ndarray:
    """t-SNE dimensionality reduction to 2D. Returns (n, 2) array."""
    tsne = TSNE(n_components=2, random_state=42, perplexity=perplexity)
    return tsne.fit_transform(components)


def get_cluster_top_terms(
    tfidf_matrix, cluster_labels: np.ndarray, tfidf: TfidfVectorizer, top_n: int = 6
) -> dict[int, list[str]]:
    """Get the most distinctive TF-IDF terms for each cluster."""
    terms = tfidf.get_feature_names_out()
    result = {}
    for c in sorted(set(cluster_labels)):
        cluster_idx = np.where(cluster_labels == c)[0]
        non_cluster_idx = np.where(cluster_labels != c)[0]
        cluster_mean = tfidf_matrix[cluster_idx].mean(axis=0).A1
        other_mean = tfidf_matrix[non_cluster_idx].mean(axis=0).A1
        diff = cluster_mean - other_mean
        top_idx = diff.argsort()[-top_n:][::-1]
        result[c] = [terms[i] for i in top_idx]
    return result
