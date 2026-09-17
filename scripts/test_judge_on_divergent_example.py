"""Contrast test: score a SummEval item where human annotators show
extreme disagreement across dimensions (coherence=1, consistency=5,
fluency=5, relevance=3.33), to check if the judge differentiates or
collapses to one holistic score."""

from evalcore.data.summeval import load_summeval
from evalcore.judge.ollama_client import call_ollama
from evalcore.judge.pointwise import build_pointwise_prompt, parse_pointwise_score

df = load_summeval()
row = df[
    (df["doc_id"] == "dm-test-4a593dc4c7e0b4d09bfdc66bb315c47b54eb15df") & (df["model_index"] == 3)
].iloc[0]

print(f"Human scores -- coherence: {row['coherence']:.2f}, consistency: {row['consistency']:.2f}, "
      f"fluency: {row['fluency']:.2f}, relevance: {row['relevance']:.2f}")
print(f"\nSummary text: {row['summary']!r}\n")

for dimension in ["coherence", "consistency", "fluency", "relevance"]:
    prompt = build_pointwise_prompt(row["article"], row["summary"], dimension)
    raw = call_ollama(prompt)
    parsed = parse_pointwise_score(raw)
    print(f"{dimension}: raw={raw!r} parsed={parsed.score} (human={row[dimension]:.2f})")

