"""Log Agent: decides whether the ticket needs a human, records it, and ends the run."""
from autogen_agentchat.agents import AssistantAgent

from backend import escalation, guardrails, ticket_log
from backend.context import ESCALATION_NOTICE, TicketContext

SYSTEM_MESSAGE = """You are the Log Agent of the ShopEase support desk. You are the last agent.
Read the whole conversation, then call the tool log_ticket exactly once.

Set escalate to true if a human should handle the ticket because ANY of these is true:
- urgency is High
- the customer is Angry or Very Angry
- the Knowledge Base Agent returned NO_MATCH
- the issue involves a legal threat, fraud, or a payment dispute
- the Resolution Agent's confidence is below 70
- the case is complicated for any other reason
Otherwise set escalate to false. Put a short reason in reason.
The system also double-checks these rules in code. After the tool call, write nothing else."""


def finalize_and_log(ctx: TicketContext) -> str:
    """Apply the escalation rules, build the final reply and write the CSV row.
    Idempotent: a ticket is only ever logged once."""
    if ctx.logged:
        return ctx.log_line

    out = guardrails.check_output(ctx.reply) if ctx.reply else guardrails.OutputResult()
    if out.blocked:
        ctx.reply = ""  # fall back to the safe holding reply below
    ctx.guardrail_flags.extend(out.flags)
    ctx.guardrail_reasons.extend(out.reasons)

    reasons = escalation.evaluate(ctx)
    status = "Escalated" if reasons else "Resolved"
    reason = " + ".join(reasons)

    reply = ctx.reply.strip()
    if not reply:
        name = ctx.customer_name or "Customer"
        reply = f"Dear {name},\n\nThank you for contacting ShopEase.\n\nWarm regards,\nShopEase Support Team"
    # Non-English replies carry the agent's own translation of the sentence.
    if (status == "Escalated" and ctx.language == "English"
            and "sent to our customer support team" not in reply.lower()):
        idx = reply.lower().rfind("warm regards")
        if idx > 0:  # keep the sign-off last
            reply = f"{reply[:idx].rstrip()}\n\n{ESCALATION_NOTICE}\n\n{reply[idx:]}"
        else:
            reply = f"{reply}\n\n{ESCALATION_NOTICE}"

    ticket_id = ticket_log.append_row({
        "customer_name": ctx.customer_name,
        "order_id": ctx.order_id,
        "complaint": ctx.complaint,
        "category": ctx.category,
        "urgency": ctx.urgency,
        "sentiment": ctx.sentiment,
        "summary": ctx.summary,
        "kb_sources": ", ".join(ctx.kb_sources) if ctx.kb_sources else "none",
        "response": reply,
        "confidence": "" if ctx.confidence is None else ctx.confidence,
        "status": status,
        "escalation_reason": reason,
        "guardrail_flags": "; ".join(ctx.guardrail_flags),
    })
    ctx.ticket_id, ctx.status, ctx.reason, ctx.final_reply = ticket_id, status, reason, reply
    ctx.logged = True
    ctx.log_line = f"LOGGED {ticket_id} | status: {status} | reason: {reason or 'none'}"
    return ctx.log_line


def create_log_agent(client, ctx: TicketContext) -> AssistantAgent:
    async def log_ticket(escalate: bool, reason: str = "") -> str:
        """Record the ticket in the log file and decide the final status. Set escalate to true
        when a human support agent must handle the ticket, and give a short reason."""
        ctx.llm_escalate = bool(escalate)
        ctx.llm_reason = (reason or "").strip()
        return finalize_and_log(ctx)

    return AssistantAgent(
        name="log_agent",
        model_client=client,
        tools=[log_ticket],
        reflect_on_tool_use=False,  # the tool output ("LOGGED TKT-...") is the final message
        description="Applies the escalation rules, records the ticket and ends the workflow.",
        system_message=SYSTEM_MESSAGE,
    )