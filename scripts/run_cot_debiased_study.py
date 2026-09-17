"""CoT prompt + order-averaging debiasing combined, on the same 15 pairs
used throughout this study arc. Baselines to compare against:
  - bare-verdict, no debiasing: 66.7% inconsistent
  - CoT, no debiasing: 41.7% inconsistent
  - bare-verdict + debiasing: proven broken (100% of tie-breaks = coin flip)
This tests whether CoT's partial real signal is enough for order-averaging
to actually help, now that there's real content signal to average toward.
~15 pairs x 4 dimensions x 2-3 calls = 120-180 calls, ~15-20 min (CoT is
slower per call)."""

import csv
import os

import numpy as np

from evalcore.data.summeval import load_summeval
from evalcore.judge.bias import run_debiased_pairwise_judge
from evalcore.judge.pairwise_cot import build_cot_pairwise_prompt, parse_cot_pairwise_verdict

N_PAIRS = 15
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
OUTPUT_PATH = "results/cot_debiased_study.csv"
RANDOM_SEED = 7  # same seed as all prior position-bias/debiasing studies

df = load_summeval()
rng = np.random.default_rng(RANDOM_SEED)
doc_ids = df["doc_id"].unique()
sampled_docs = rng.choice(doc_ids, size=N_PAIRS, replace=False)

os.makedirs("results", exist_ok=True)
fieldnames = ["doc_id", "dimension", "model_a_idx", "model_b_idx", "winner", "agreement", "n_calls", "note"]

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i, doc_id in enumerate(sampled_docs):
        doc_rows = df[df["doc_id"] == doc_id]
        two_models = doc_rows.sample(n=2, random_state=int(rng.integers(0, 2**31 - 1)))
        row_a, row_b = two_models.iloc[0], two_models.iloc[1]

        for dimension in DIMENSIONS:
            result = run_debiased_pairwise_judge(
                row_a["article"], row_a["summary"], row_b["summary"], dimension,
                prompt_builder=build_cot_pairwise_prompt,
                verdict_parser=parse_cot_pairwise_verdict,
            )
            writer.writerow({
                "doc_id": doc_id,
                "dimension": dimension,
                "model_a_idx": row_a["model_index"],
                "model_b_idx": row_b["model_index"],
                "winner": result.winner,
                "agreement": result.agreement,
                "n_calls": result.n_calls,
                "note": result.note,
            })
            f.flush()
        print(f"  completed pair {i+1}/{N_PAIRS}: {doc_id}")

print(f"\nSaved to {OUTPUT_PATH}")

