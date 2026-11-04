"""Per-ticket working memory shared by the tools and the team.

One TicketContext is created for every complaint, so nothing leaks between customers.
Agents hand work to each other through messages; this object additionally records the
structured values (category, sources, confidence ...) so Python code can enforce the
escalation rules and write the log row without trusting the LLM to copy text.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field

ESCALATION_NOTICE = (
    "Your question has been sent to our customer support team, they will contact you shortly."
)
URGENCIES = {"low": "Low", "medium": "Medium", "high": "High"}


# Unicode ranges of non-Latin scripts -> language name used in the reply instruction
SCRIPTS = [
    ("Tamil", 0x0B80, 0x0BFF), ("Hindi", 0x0900, 0x097F), ("Telugu", 0x0C00, 0x0C7F),
    ("Kannada", 0x0C80, 0x0CFF), ("Malayalam", 0x0D00, 0x0D7F), ("Bengali", 0x0980, 0x09FF),
    ("Gujarati", 0x0A80, 0x0AFF), ("Punjabi", 0x0A00, 0x0A7F), ("Arabic", 0x0600, 0x06FF),
    ("Russian", 0x0400, 0x04FF), ("Japanese", 0x3040, 0x30FF), ("Korean", 0xAC00, 0xD7AF),
    ("Chinese", 0x4E00, 0x9FFF),
]


def detect_language(text: str) -> str:
    """Detect non-Latin scripts by Unicode range. Returns 'English' when the text is Latin
    script (the agent is still told to match the customer's language in that case)."""
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "English"
    best, best_n = "English", 0
    for name, lo, hi in SCRIPTS:
        n = sum(lo <= ord(c) <= hi for c in letters)
        if n > best_n:
            best, best_n = name, n
    return best if best_n / len(letters) >= 0.4 else "English"


def parse_fields(text: str) -> dict[str, str]:
    """Read 'KEY: value' lines (tolerates **bold** markdown)."""
    clean = text.replace("**", "")
    return {
        m.group(1).upper(): m.group(2).strip()
        for m in re.finditer(r"^\s*([A-Za-z_]+)\s*:\s*(.+?)\s*$", clean, re.M)
    }


@dataclass
class TicketContext:
    customer_name: str = ""
    order_id: str = ""
    complaint: str = ""
    language: str = "English"
    guardrail_flags: list = field(default_factory=list)
    guardrail_reasons: list = field(default_factory=list)

    # Query Agent output
    category: str = "Other"
    business_line: str = ""
    product: str = ""
    urgency: str = "Medium"
    sentiment: str = "Neutral"
    summary: str = ""
    search_query: str = ""
    catalog_hits: list = field(default_factory=list)

    # Knowledge Agent output
    search_calls: int = 0
    kb_searched: bool = False
    kb_relevant: bool = False
    best_score: float = 0.0
    kb_sources: list = field(default_factory=list)

    # Resolution Agent output
    reply: str = ""
    confidence: int | None = None

    # Log Agent output
    llm_escalate: bool = False
    llm_reason: str = ""
    logged: bool = False
    log_line: str = ""
    ticket_id: str = ""
    status: str = ""
    reason: str = ""
    final_reply: str = ""

    # -- parsing the agents' messages -------------------------------------
    def ingest(self, source: str, text: str) -> None:
        if source == "query_agent":
            f = parse_fields(text)
            self.category = f.get("CATEGORY", self.category)
            self.business_line = f.get("BUSINESS_LINE", self.business_line)
            self.product = f.get("PRODUCT", self.product)
            self.urgency = URGENCIES.get(f.get("URGENCY", "").lower(), "Medium")
            self.sentiment = f.get("SENTIMENT", self.sentiment).title()
            self.summary = f.get("SUMMARY", self.summary)
            self.search_query = f.get("SEARCH_QUERY", self.search_query)
        elif source == "resolution_agent":
            clean = text.replace("**", "")
            m = list(re.finditer(r"CONFIDENCE\s*:\s*(\d{1,3})", clean, re.I))
            self.confidence = min(int(m[-1].group(1)), 100) if m else None
            body = re.split(r"\n\s*CONFIDENCE\s*:", clean, flags=re.I)[0]
            body = re.sub(r"^\s*REPLY\s*:\s*", "", body, flags=re.I).strip()
            self.reply = body

    def ingest_messages(self, messages) -> None:
        for m in messages:
            source = getattr(m, "source", "")
            if source in ("query_agent", "resolution_agent") and hasattr(m, "to_text"):
                self.ingest(source, m.to_text())