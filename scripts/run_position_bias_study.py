"""Run the position-bias check across real SummEval pairs, all 4 dimensions.
~15 article/model-pair comparisons x 4 dimensions x 2 calls (original +
swapped order) = 120 calls, ~8-12 minutes."""

import csv
import os

import numpy as np

from evalcore.data.summeval import load_summeval
from evalcore.judge.bias import run_position_bias_check

N_PAIRS = 15
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
OUTPUT_PATH = "results/position_bias_study.csv"
RANDOM_SEED = 7

df = load_summeval()
rng = np.random.default_rng(RANDOM_SEED)
doc_ids = df["doc_id"].unique()
sampled_docs = rng.choice(doc_ids, size=N_PAIRS, replace=False)

os.makedirs("results", exist_ok=True)
fieldnames = ["doc_id", "dimension", "model_a_idx", "model_b_idx", "winner_original_order", "winner_swapped_order", "consistent", "note"]

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i, doc_id in enumerate(sampled_docs):
        doc_rows = df[df["doc_id"] == doc_id]
        two_models = doc_rows.sample(n=2, random_state=int(rng.integers(0, 2**31 - 1)))
        row_a, row_b = two_models.iloc[0], two_models.iloc[1]

        for dimension in DIMENSIONS:
            result = run_position_bias_check(row_a["article"], row_a["summary"], row_b["summary"], dimension)
            writer.writerow({
                "doc_id": doc_id,
                "dimension": dimension,
                "model_a_idx": row_a["model_index"],
                "model_b_idx": row_b["model_index"],
                "winner_original_order": result.winner_original_order,
                "winner_swapped_order": result.winner_swapped_order,
                "consistent": result.consistent,
                "note": result.note,
            })
            f.flush()
        print(f"  completed pair {i+1}/{N_PAIRS}: {doc_id}")

print(f"\nSaved to {OUTPUT_PATH}")

