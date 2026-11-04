"""Escalation rules. Enforced in Python so a ticket can never skip a rule because the
LLM forgot it. The Log Agent's own judgement can ADD an escalation but never remove one."""
from __future__ import annotations

import re

from backend import config
from backend.context import TicketContext

LEGAL = re.compile(
    r"\b(lawyer|attorney|advocate|legal(ly)?|sue|suing|lawsuit|court|police|consumer (forum|court))\b", re.I
)
FRAUD_PAYMENT = re.compile(
    r"\b(fraud|scam|unauthori[sz]ed|chargeback|dispute|double[- ]charged|my bank|"
    r"charged .{0,30}(twice|thrice|two|three|\d+) times|stole|stolen|hacked)\b", re.I
)
ANGRY = {"angry", "very angry", "furious", "outraged"}


def evaluate(ctx: TicketContext) -> list[str]:
    """Return the list of escalation reasons. Empty list means Resolved."""
    reasons: list[str] = []
    text = ctx.complaint

    if ctx.urgency.lower() == "high":
        reasons.append("High urgency")
    if ctx.sentiment.lower() in ANGRY:
        reasons.append(f"{ctx.sentiment} customer")
    if not ctx.kb_searched:
        reasons.append("Knowledge base was not searched")
    elif not ctx.kb_relevant:
        reasons.append("No matching document")
    if LEGAL.search(text):
        reasons.append("Legal threat")
    if FRAUD_PAYMENT.search(text):
        reasons.append("Fraud or payment dispute")
    if ctx.confidence is None:
        reasons.append("Missing confidence score")
    elif ctx.confidence < config.CONFIDENCE_THRESHOLD:
        reasons.append(f"Low confidence ({ctx.confidence}%)")
    reasons.extend(ctx.guardrail_reasons)
    if ctx.llm_escalate:
        reasons.append(f"Log Agent: {ctx.llm_reason or 'judged too complex'}")
    return reasons