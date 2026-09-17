"""Manual smoke test: run the pointwise judge on one real SummEval item
against the live Ollama server. Not a pytest test -- this hits the network
and a live model, run manually to sanity-check the wrapper end to end."""

from evalcore.data.summeval import load_summeval
from evalcore.judge.ollama_client import call_ollama
from evalcore.judge.pointwise import build_pointwise_prompt, parse_pointwise_score

df = load_summeval()
row = df.iloc[0]

for dimension in ["coherence", "consistency", "fluency", "relevance"]:
    prompt = build_pointwise_prompt(row["article"], row["summary"], dimension)
    raw = call_ollama(prompt)
    parsed = parse_pointwise_score(raw)
    human_score = row[dimension]
    print(f"\n{dimension}:")
    print(f"  raw LLM response: {raw!r}")
    print(f"  parsed score: {parsed.score} (parse_ok={parsed.parse_ok}, note={parsed.parse_note})")
    print(f"  human score: {human_score:.2f}")
