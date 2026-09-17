"""Score original vs. verbose-variant summaries with the pointwise judge
across all 4 dimensions, on real SummEval summaries. ~15 summaries x 4
dimensions x 2 versions = 120 calls, ~8-12 minutes."""

import csv
import os

import numpy as np

from evalcore.data.summeval import load_summeval
from evalcore.judge.ollama_client import call_ollama
from evalcore.judge.pointwise import build_pointwise_prompt, parse_pointwise_score
from evalcore.judge.style_bias import compare_verbosity_scores, make_verbose_variant

N_SUMMARIES = 15
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
OUTPUT_PATH = "results/verbosity_bias_study.csv"
RANDOM_SEED = 11
MODEL = "llama3.1:latest"

df = load_summeval()
rng = np.random.default_rng(RANDOM_SEED)
sample = df.sample(n=N_SUMMARIES, random_state=int(rng.integers(0, 2**31 - 1)))

os.makedirs("results", exist_ok=True)
fieldnames = ["doc_id", "model_idx", "dimension", "original_score", "verbose_score", "score_delta", "length_ratio"]

with open(OUTPUT_PATH, "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    writer.writeheader()

    for i, (_, row) in enumerate(sample.iterrows()):
        original = row["summary"]
        verbose = make_verbose_variant(original, seed=i)

        for dimension in DIMENSIONS:
            orig_prompt = build_pointwise_prompt(row["article"], original, dimension)
            verbose_prompt = build_pointwise_prompt(row["article"], verbose, dimension)

            orig_score = parse_pointwise_score(call_ollama(orig_prompt, model=MODEL)).score
            verbose_score = parse_pointwise_score(call_ollama(verbose_prompt, model=MODEL)).score

            comparison = compare_verbosity_scores(orig_score, verbose_score, dimension, original, verbose)
            writer.writerow({
                "doc_id": row["doc_id"],
                "model_idx": row["model_index"],
                "dimension": dimension,
                "original_score": comparison.original_score,
                "verbose_score": comparison.verbose_score,
                "score_delta": comparison.score_delta,
                "length_ratio": round(comparison.length_ratio, 2),
            })
            f.flush()
        print(f"  completed summary {i+1}/{N_SUMMARIES}: {row['doc_id']}")

print(f"\nSaved to {OUTPUT_PATH}")

