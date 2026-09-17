"""Qualitative check: pull raw CoT reasoning text for a few coherence pairs
that flipped under order-swap, to see whether the model's stated reasoning
engages with actual content or is generic filler preceding a
position-driven verdict. Uses the SAME 3 pairs (by index into the position-
bias sample) that showed coherence inconsistency in the CoT study, so this
is a direct qualitative follow-up on already-flagged cases.
~3 pairs x 2 orders = 6 calls, ~2-3 minutes."""

import numpy as np

from evalcore.data.summeval import load_summeval
from evalcore.judge.ollama_client import call_ollama
from evalcore.judge.pairwise_cot import build_cot_pairwise_prompt

RANDOM_SEED = 7
MODEL = "llama3.1:latest"

# doc_ids that showed coherence inconsistency in the CoT study
FLAGGED_DOC_IDS = [
    "dm-test-5be0a9584b051175d9f4842a143b76385335d96a",
    "dm-test-c50d33e9749e7bb484d9b69c4f5fca35a3a50cb5",
    "dm-test-b5bc2ae78441e4ac08fb01823d5fc0f1627c3166",
]

df = load_summeval()
rng = np.random.default_rng(RANDOM_SEED)
doc_ids = df["doc_id"].unique()
sampled_docs = rng.choice(doc_ids, size=15, replace=False)

for i, doc_id in enumerate(sampled_docs):
    if doc_id not in FLAGGED_DOC_IDS:
        continue
    doc_rows = df[df["doc_id"] == doc_id]
    two_models = doc_rows.sample(n=2, random_state=int(rng.integers(0, 2**31 - 1)))
    row_a, row_b = two_models.iloc[0], two_models.iloc[1]

    print(f"\n{'='*80}\ndoc_id: {doc_id}\n{'='*80}")

    prompt_orig = build_cot_pairwise_prompt(row_a["article"], row_a["summary"], row_b["summary"], "coherence")
    response_orig = call_ollama(prompt_orig, model=MODEL)
    print(f"\n--- ORIGINAL ORDER (A=model_{row_a['model_index']}, B=model_{row_b['model_index']}) ---")
    print(response_orig)

    prompt_swap = build_cot_pairwise_prompt(row_a["article"], row_b["summary"], row_a["summary"], "coherence")
    response_swap = call_ollama(prompt_swap, model=MODEL)
    print(f"\n--- SWAPPED ORDER (A=model_{row_b['model_index']}, B=model_{row_a['model_index']}) ---")
    print(response_swap)

