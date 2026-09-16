"""Unit tests for evalcore.data.summeval — uses a synthetic fixture so
these tests run offline and fast, independent of the HF dataset mirror."""

import pandas as pd
import pytest

from evalcore.data.summeval import (
    EXPECTED_N_MODELS,
    reshape_to_long,
    validate_summeval_dataframe,
)


def make_synthetic_raw(n_docs: int = 3, n_models: int = EXPECTED_N_MODELS) -> pd.DataFrame:
    rows = []
    for d in range(n_docs):
        rows.append(
            {
                "id": f"doc-{d}",
                "text": f"article text {d}",
                "machine_summaries": [f"summary {d}-{m}" for m in range(n_models)],
                "human_summaries": [f"reference {d}-{h}" for h in range(11)],
                "coherence": [3.0] * n_models,
                "consistency": [4.0] * n_models,
                "fluency": [5.0] * n_models,
                "relevance": [2.5] * n_models,
            }
        )
    return pd.DataFrame(rows)


def test_reshape_produces_expected_row_count():
    raw = make_synthetic_raw(n_docs=3, n_models=16)
    long_df = reshape_to_long(raw)
    assert len(long_df) == 3 * 16


def test_reshape_preserves_score_values():
    raw = make_synthetic_raw(n_docs=1, n_models=2)
    long_df = reshape_to_long(raw)
    assert long_df.iloc[0]["coherence"] == 3.0
    assert long_df.iloc[0]["relevance"] == 2.5


def test_validation_passes_on_well_formed_data():
    raw = make_synthetic_raw(n_docs=100, n_models=16)
    long_df = reshape_to_long(raw)
    report = validate_summeval_dataframe(long_df)
    assert report.passed
    assert report.errors == []
    assert report.n_rows == 1600
    assert report.n_unique_docs == 100


def test_validation_catches_out_of_range_scores():
    raw = make_synthetic_raw(n_docs=2, n_models=16)
    long_df = reshape_to_long(raw)
    long_df.loc[0, "coherence"] = 7.0
    report = validate_summeval_dataframe(long_df)
    assert not report.passed
    assert any("coherence" in e for e in report.errors)


def test_validation_catches_missing_column():
    raw = make_synthetic_raw(n_docs=1, n_models=16)
    long_df = reshape_to_long(raw).drop(columns=["fluency"])
    report = validate_summeval_dataframe(long_df)
    assert not report.passed
    assert any("missing columns" in e for e in report.errors)


def test_validation_catches_wrong_model_count_per_doc():
    raw = make_synthetic_raw(n_docs=2, n_models=16)
    long_df = reshape_to_long(raw)
    long_df = long_df[~((long_df["doc_id"] == "doc-0") & (long_df["model_index"] == 15))]
    report = validate_summeval_dataframe(long_df)
    assert not report.passed
    assert any("!= 16" in e for e in report.errors)


@pytest.mark.network
def test_real_summeval_loads_and_validates():
    """Marked network — run explicitly with `pytest -m network`.
    Skipped by default so offline/CI-without-internet runs stay fast."""
    from evalcore.data.summeval import load_summeval

    df = load_summeval()
    assert len(df) == 1600
    assert df["doc_id"].nunique() == 100

