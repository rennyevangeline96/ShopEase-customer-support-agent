"""Knowledge Base Agent: finds the relevant passages in the company documents.
Uses RAG 2 (policies, FAQs, solved tickets) through the search_kb tool."""
import asyncio

from autogen_agentchat.agents import AssistantAgent

from backend import config
from backend.context import TicketContext
from backend.rag.faiss_store import get_store

MAX_SEARCHES = 3  # per-ticket cap on search calls

SYSTEM_MESSAGE = """You are the Knowledge Base Agent of the ShopEase support desk.
Your only job is to search the company documents.

Read the SEARCH_QUERY line written by the Customer Query Agent and call the tool search_kb
exactly once with it. If that line is missing, use the customer's complaint as the query.
Do not write any other text. Do not answer the customer."""


def create_knowledge_agent(client, ctx: TicketContext) -> AssistantAgent:
    async def search_kb(query: str) -> str:
        """Search the ShopEase knowledge base (shipping, refund, account and login, product FAQ,
        billing and payments, previously solved tickets). Returns matching passages with the
        source file name, or NO_MATCH if no document answers the question."""
        if ctx.search_calls >= MAX_SEARCHES:
            return "KB_RESULT: LIMIT_REACHED\nSearch limit for this ticket was reached."
        ctx.search_calls += 1
        ctx.kb_searched = True

        store = get_store("kb")
        hits = await asyncio.to_thread(store.search, (query or ctx.complaint).strip(), 5)
        limit = lambda h: config.RELEVANCE_THRESHOLD if h["method"] == "vector" else config.KEYWORD_THRESHOLD
        good = [h for h in hits if h["score"] >= limit(h)]
        # Official policy files first, past solved tickets last
        good.sort(key=lambda h: h["source"] == "solved_tickets.md")
        good = good[:3]

        ctx.best_score = hits[0]["score"] if hits else 0.0
        ctx.kb_relevant = bool(good)
        ctx.kb_sources = list(dict.fromkeys(h["source"] for h in good))

        if not good:
            return (
                "KB_RESULT: NO_MATCH\n"
                f"BEST_SCORE: {ctx.best_score}\n"
                "SOURCES: none\n"
                "No document in the knowledge base answers this question."
            )
        parts = [
            "KB_RESULT: MATCH",
            f"BEST_SCORE: {ctx.best_score}",
            f"SOURCES: {', '.join(ctx.kb_sources)}",
            "",
        ]
        for i, h in enumerate(good, 1):
            parts.append(f"PASSAGE {i} (source: {h['source']}, section: {h['heading']}, score {h['score']}):")
            parts.append(h["text"])
            parts.append("")
        return "\n".join(parts).strip()

    return AssistantAgent(
        name="knowledge_agent",
        model_client=client,
        tools=[search_kb],
        reflect_on_tool_use=False,  # the message shown is the tool's exact output (no LLM rewrite)
        description="Searches the company documents and returns matching passages with source file names.",
        system_message=SYSTEM_MESSAGE,
    )
