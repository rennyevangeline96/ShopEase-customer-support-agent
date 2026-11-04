"""Human-in-the-loop: a support person approves an escalated reply through AutoGen's
UserProxyAgent.

Why a second, separate run: Streamlit re-runs the script on every click and cannot pause
in the middle of the live four-agent run to wait for a person. So the main run finishes
first and marks the ticket Escalated. When the support person clicks Approve, this small
AutoGen team runs: the UserProxyAgent ("support_reviewer") is asked to review the draft and
its input_func returns the reply the person edited in the UI. The text it returns is what
gets saved as the approved reply.
"""
from __future__ import annotations

import asyncio

from autogen_agentchat.agents import UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination
from autogen_agentchat.teams import RoundRobinGroupChat

REVIEWER = "support_reviewer"


async def _review(ticket_id: str, complaint: str, reasons: str, draft: str, edited: str) -> str:
    final_text = (edited or "").strip() or draft  # an empty box approves the AI draft as it is

    reviewer = UserProxyAgent(
        name=REVIEWER,
        description="A human support person who reviews and approves escalated replies.",
        input_func=lambda prompt: final_text,  # the person's decision, taken from the UI
    )
    team = RoundRobinGroupChat([reviewer], termination_condition=MaxMessageTermination(2))
    task = (
        f"ESCALATED TICKET {ticket_id} NEEDS HUMAN APPROVAL\n"
        f"Escalation reasons: {reasons}\n"
        f"Customer complaint: {complaint}\n"
        f"AI draft reply:\n{draft}\n"
        "Please approve the reply, or return an edited version."
    )
    result = await team.run(task=task)
    replies = [m for m in result.messages if getattr(m, "source", "") == REVIEWER]
    if not replies:
        raise RuntimeError("UserProxyAgent returned no reply")
    return replies[-1].to_text()


def approve_reply(ticket_id: str, complaint: str, reasons: str, draft: str, edited: str):
    """Run the approval through UserProxyAgent. Returns (approved_reply, used_user_proxy).
    If the AutoGen step fails for any reason, the person's edited reply is still saved so
    the support person's work is never lost (used_user_proxy is False in that case)."""
    try:
        return asyncio.run(_review(ticket_id, complaint, reasons, draft, edited)), True
    except Exception:
        return ((edited or "").strip() or draft), False