"""SummEval ingestion and validation for evalcore.

Loads the human-annotated SummEval dataset (Fabbri et al., 2020) via the
free `mteb/summeval` Hugging Face mirror, reshapes it into a long-format
DataFrame (one row per article-model pair), and validates it against known
dataset invariants before anything downstream trusts it.

Dataset facts (verified against the dataset card, not assumed):
- 100 source articles (CNN/DailyMail), each with 16 machine-generated
  summaries and 11 unscored human reference summaries.
- Each machine summary has an averaged human score (across 3 experts +
  5 crowd workers) on a 1-5 Likert scale for coherence, consistency,
  fluency, and relevance.
- Total scored rows after reshaping: 100 * 16 = 1600.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, Field, field_validator

logger = logging.getLogger(__name__)

HF_DATASET_NAME = "mteb/summeval"
HF_SPLIT = "test"
EXPECTED_N_ARTICLES = 100
EXPECTED_N_MODELS = 16
SCORE_DIMENSIONS = ("coherence", "consistency", "fluency", "relevance")
SCORE_MIN, SCORE_MAX = 1.0, 5.0


class ScoredSummary(BaseModel):
    """One machine-generated summary with its averaged human scores."""

    doc_id: str
    model_index: int = Field(ge=0, lt=EXPECTED_N_MODELS)
    article: str
    summary: str
    coherence: float
    consistency: float
    fluency: float
    relevance: float

    @field_validator("coherence", "consistency", "fluency", "relevance")
    @classmethod
    def score_in_range(cls, v: float) -> float:
        if not (SCORE_MIN <= v <= SCORE_MAX):
            raise ValueError(f"score {v} outside expected range [{SCORE_MIN}, {SCORE_MAX}]")
        return v


@dataclass
class ValidationReport:
    n_rows: int
    n_unique_docs: int
    n_models_per_doc: dict[str, int]
    passed: bool
    errors: list[str]


def load_summeval_raw() -> "pd.DataFrame":
    """Load the raw (wide-format) SummEval dataset from the HF mirror."""
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_NAME, split=HF_SPLIT)
    return ds.to_pandas()


def reshape_to_long(raw_df: pd.DataFrame) -> pd.DataFrame:
    """Reshape wide SummEval rows (lists per article) into one row per
    (article, model) pair, which is the unit our statistical tests operate on.
    """
    records: list[dict] = []
    for _, row in raw_df.iterrows():
        n = len(row["machine_summaries"])
        for i in range(n):
            records.append(
                {
                    "doc_id": row["id"],
                    "model_index": i,
                    "article": row["text"],
                    "summary": row["machine_summaries"][i],
                    "coherence": float(row["coherence"][i]),
                    "consistency": float(row["consistency"][i]),
                    "fluency": float(row["fluency"][i]),
                    "relevance": float(row["relevance"][i]),
                }
            )
    return pd.DataFrame.from_records(records)


def validate_summeval_dataframe(df: pd.DataFrame) -> ValidationReport:
    """Check the reshaped dataframe against known dataset invariants.

    This is the guardrail that catches silent corruption (e.g. a future
    HF dataset version change) before it poisons downstream statistics.
    """
    errors: list[str] = []

    required_cols = {"doc_id", "model_index", "article", "summary", *SCORE_DIMENSIONS}
    missing_cols = required_cols - set(df.columns)
    if missing_cols:
        errors.append(f"missing columns: {missing_cols}")
        return ValidationReport(0, 0, {}, False, errors)

    if df[list(required_cols)].isnull().any().any():
        null_cols = df[list(required_cols)].columns[df[list(required_cols)].isnull().any()].tolist()
        errors.append(f"null values found in columns: {null_cols}")

    n_unique_docs = df["doc_id"].nunique()
    if n_unique_docs != EXPECTED_N_ARTICLES:
        errors.append(f"expected {EXPECTED_N_ARTICLES} unique docs, found {n_unique_docs}")

    models_per_doc = df.groupby("doc_id")["model_index"].nunique().to_dict()
    bad_docs = {d: c for d, c in models_per_doc.items() if c != EXPECTED_N_MODELS}
    if bad_docs:
        errors.append(f"docs with != {EXPECTED_N_MODELS} models: {bad_docs}")

    for dim in SCORE_DIMENSIONS:
        out_of_range = df[(df[dim] < SCORE_MIN) | (df[dim] > SCORE_MAX)]
        if len(out_of_range) > 0:
            errors.append(f"{len(out_of_range)} rows with {dim} outside [{SCORE_MIN},{SCORE_MAX}]")

    empty_summaries = df[df["summary"].str.strip() == ""]
    if len(empty_summaries) > 0:
        errors.append(f"{len(empty_summaries)} rows with empty summary text")

    passed = len(errors) == 0
    return ValidationReport(
        n_rows=len(df),
        n_unique_docs=n_unique_docs,
        n_models_per_doc=models_per_doc,
        passed=passed,
        errors=errors,
    )


def load_summeval() -> pd.DataFrame:
    """Public entry point: load, reshape, and validate SummEval.

    Raises RuntimeError if validation fails, so callers never silently
    work with a corrupted dataset.
    """
    logger.info("Loading SummEval from %s (split=%s)", HF_DATASET_NAME, HF_SPLIT)
    raw_df = load_summeval_raw()
    long_df = reshape_to_long(raw_df)
    report = validate_summeval_dataframe(long_df)
    if not report.passed:
        raise RuntimeError(f"SummEval validation failed: {report.errors}")
    logger.info(
        "SummEval loaded and validated: %d rows, %d unique docs",
        report.n_rows,
        report.n_unique_docs,
    )
    return long_df


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    df = load_summeval()
    print(df.head())
    print(f"\nShape: {df.shape}")
    print(f"\nScore summary:\n{df[list(SCORE_DIMENSIONS)].describe()}")
