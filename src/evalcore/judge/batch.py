"""Batch pointwise judge scoring with incremental CSV checkpointing.

Given ~3-5s per Ollama call, a batch of a few hundred calls takes many
minutes. Checkpointing after every call means an interruption or crash
loses at most the in-flight call, not the whole batch -- and reruns
automatically skip already-scored (doc_id, model_index, dimension) rows.
"""

from __future__ import annotations

import csv
import os

import pandas as pd

from evalcore.judge.ollama_client import call_ollama
from evalcore.judge.pointwise import build_pointwise_prompt, parse_pointwise_score

CHECKPOINT_COLUMNS = [
    "doc_id", "model_index", "dimension", "human_score",
    "raw_response", "judge_score", "parse_ok", "parse_note",
]


def load_existing_checkpoint(checkpoint_path: str) -> set[tuple[str, int, str]]:
    """Return the set of (doc_id, model_index, dimension) already scored."""
    if not os.path.exists(checkpoint_path):
        return set()
    existing = pd.read_csv(checkpoint_path)
    return set(zip(existing["doc_id"], existing["model_index"], existing["dimension"]))


def run_batch_judge_scoring(
    df: pd.DataFrame,
    dimensions: list[str],
    checkpoint_path: str,
    model: str = "llama3.1:latest",
) -> None:
    """Score every (row, dimension) pair in df with the pointwise judge,
    appending results to checkpoint_path as they complete. Skips pairs
    already present in the checkpoint file, so this is safely re-runnable.
    """
    already_done = load_existing_checkpoint(checkpoint_path)
    file_exists = os.path.exists(checkpoint_path)

    with open(checkpoint_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=CHECKPOINT_COLUMNS)
        if not file_exists:
            writer.writeheader()

        total = len(df) * len(dimensions)
        done_count = 0
        for _, row in df.iterrows():
            for dimension in dimensions:
                key = (row["doc_id"], row["model_index"], dimension)
                done_count += 1
                if key in already_done:
                    continue

                prompt = build_pointwise_prompt(row["article"], row["summary"], dimension)
                raw = call_ollama(prompt, model=model)
                parsed = parse_pointwise_score(raw)

                writer.writerow({
                    "doc_id": row["doc_id"],
                    "model_index": row["model_index"],
                    "dimension": dimension,
                    "human_score": row[dimension],
                    "raw_response": parsed.raw_response,
                    "judge_score": parsed.score if parsed.score is not None else "",
                    "parse_ok": parsed.parse_ok,
                    "parse_note": parsed.parse_note,
                })
                f.flush()

                if done_count % 10 == 0:
                    print(f"  progress: {done_count}/{total}")

