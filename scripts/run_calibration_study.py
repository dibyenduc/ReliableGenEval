"""Run the judge-vs-human calibration study: sample 100 rows across the
full SummEval dataset (all 16 models, all 100 articles) and score each
on all 4 dimensions with the local LLM judge. ~400 calls, ~25-35 minutes
at current Ollama latency. Checkpointed -- safe to interrupt and rerun."""

import numpy as np

from evalcore.data.summeval import load_summeval
from evalcore.judge.batch import run_batch_judge_scoring

N_SAMPLES = 100
DIMENSIONS = ["coherence", "consistency", "fluency", "relevance"]
CHECKPOINT_PATH = "results/judge_calibration_raw.csv"
RANDOM_SEED = 42

df = load_summeval()
rng = np.random.default_rng(RANDOM_SEED)
sample_idx = rng.choice(len(df), size=N_SAMPLES, replace=False)
sample_df = df.iloc[sample_idx].reset_index(drop=True)

print(f"Sampled {len(sample_df)} rows across {sample_df['model_index'].nunique()} models "
      f"and {sample_df['doc_id'].nunique()} articles")
print(f"Scoring {len(sample_df) * len(DIMENSIONS)} (row, dimension) pairs -- this will take a while.")

run_batch_judge_scoring(sample_df, DIMENSIONS, CHECKPOINT_PATH)
print(f"\nDone. Results checkpointed to {CHECKPOINT_PATH}")

