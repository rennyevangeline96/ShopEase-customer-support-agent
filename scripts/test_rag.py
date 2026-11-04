"""Step 3 check: builds both RAG indexes and runs sample searches.
Run from the project root:  python -m scripts.test_rag
Run it twice: the second run must say 'loaded from cache (no API call)'."""
from backend.rag import faiss_store
from backend.rag.faiss_store import get_store

CATALOG_QUERIES = [
    "my blender arrived broken",
    "my storefront website is not loading",
    "the monthly marketing report",
]
KB_QUERIES = [
    "Where is my order? It was supposed to arrive yesterday",
    "I want to return the shoes I bought last week, how do I get a refund",
    "I forgot my password and can't log in",
    "The blender I received is broken, can I get a replacement",
    "You charged my card three times",
    "Do you sell gift cards for a company of 500 employees with custom branding",
]


def run(store_name, queries):
    store = get_store(store_name)
    print(f"\n===== {store_name.upper()} =====")
    for q in queries:
        hits = store.search(q, k=3)
        verdict = "MATCH" if store.is_relevant(hits) else "NO MATCH"
        print(f"\nQ: {q}\n   -> {verdict}")
        for h in hits:
            print(f"   {h['score']:.3f}  {h['source']}  [{h['heading']}]")


if __name__ == "__main__":
    run("catalog", CATALOG_QUERIES)
    run("kb", KB_QUERIES)
    print(f"\nEmbedding usage this run: {faiss_store.usage['tokens']} tokens, {faiss_store.usage['calls']} API calls")
