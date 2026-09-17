"""Find a SummEval item where human annotators clearly disagree across
dimensions -- a good stress test for whether the judge actually
differentiates dimensions or collapses them into one holistic score."""

from evalcore.data.summeval import load_summeval

df = load_summeval()
df["dimension_spread"] = df[["coherence", "consistency", "fluency", "relevance"]].max(axis=1) - df[
    ["coherence", "consistency", "fluency", "relevance"]
].min(axis=1)

top = df.sort_values("dimension_spread", ascending=False).head(5)
for _, row in top.iterrows():
    print(f"\ndoc_id={row['doc_id']}, model_index={row['model_index']}, spread={row['dimension_spread']:.2f}")
    print(f"  coherence={row['coherence']:.2f}  consistency={row['consistency']:.2f}  "
          f"fluency={row['fluency']:.2f}  relevance={row['relevance']:.2f}")
