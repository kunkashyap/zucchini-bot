import logging
from app.graph.workflow import run_workflow

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")

test_queries = [
    "hi",
    "who are you?",
    "What is Python?",
    "Explain LangGraph in simple terms.",
    "Aap kaun ho aur kya kar sakte ho?",
]

print("==================================================================")
print("RUNNING ZUCCHINI PROMPT CONTAMINATION & IDENTITY TEST SUITE")
print("==================================================================")

for q in test_queries:
    print(f"\n>>> QUERY: '{q}'")
    result = run_workflow(query=q)
    print(f"LANGUAGE DETECTED: {result.get('language')}")
    print(f"CONTEXT CHUNKS RETRIEVED: {len(result.get('retrieved_documents', []))}")
    print(f"RESPONSE:\n{result.get('response')}")
    print("-" * 66)
