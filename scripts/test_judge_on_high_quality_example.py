"""Manual smoke test: run the pointwise judge on a HIGH-scoring SummEval
item (contrast case) to check whether judge scores actually vary with
quality, or are anchored near a fixed value regardless of content."""

from evalcore.data.summeval import load_summeval
from evalcore.judge.ollama_client import call_ollama
from evalcore.judge.pointwise import build_pointwise_prompt, parse_pointwise_score

df = load_summeval()
high_quality_row = df.loc[df["coherence"].idxmax()]

print(f"Selected row: doc_id={high_quality_row['doc_id']}, model_index={high_quality_row['model_index']}")
print(f"Human scores -- coherence: {high_quality_row['coherence']:.2f}, "
      f"consistency: {high_quality_row['consistency']:.2f}, "
      f"fluency: {high_quality_row['fluency']:.2f}, "
      f"relevance: {high_quality_row['relevance']:.2f}")

for dimension in ["coherence", "consistency", "fluency", "relevance"]:
    prompt = build_pointwise_prompt(high_quality_row["article"], high_quality_row["summary"], dimension)
    raw = call_ollama(prompt)
    parsed = parse_pointwise_score(raw)
    human_score = high_quality_row[dimension]
    print(f"\n{dimension}:")
    print(f"  raw LLM response: {raw!r}")
    print(f"  parsed score: {parsed.score} (parse_ok={parsed.parse_ok})")
    print(f"  human score: {human_score:.2f}")
