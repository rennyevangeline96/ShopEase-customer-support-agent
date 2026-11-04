"""Light guardrails.

INPUT  (before any LLM call): mask card numbers, CVV, OTP, Aadhaar and passwords, and flag
       prompt-injection attempts. Only the masked text is shown to the agents and logged.
OUTPUT (before the reply is logged/shown): block replies that contain contact details or
       links that are not in our documents, and flag money amounts that are not in our documents.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from functools import lru_cache

from backend import config

# ----------------------------------------------------------------------------
# Input guardrail
# ----------------------------------------------------------------------------
MASKS = [
    ("cvv", re.compile(r"\bcvv\s*[:=]?\s*\d{3,4}\b", re.I), "[CVV REMOVED]"),
    ("otp", re.compile(r"\b(?:otp|one[- ]time (?:password|code))\s*(?:is|:|=)?\s*\d{4,8}\b", re.I), "[OTP REMOVED]"),
    ("password", re.compile(r"\bpassword\s*(?:is|:|=)\s*\S+", re.I), "password [REMOVED]"),
    ("card_number", re.compile(r"\b\d(?:[ -]?\d){12,18}\b"), "[CARD NUMBER REMOVED]"),
    ("aadhaar", re.compile(r"\b\d{4}[ -]?\d{4}[ -]?\d{4}\b"), "[AADHAAR REMOVED]"),
]
INJECTION = re.compile(
    r"(ignore|disregard|forget|override)\s+(all\s+|any\s+)?(the\s+)?(previous|prior|above|earlier|your)\s+"
    r"(instructions?|rules?|prompts?)|system prompt|reveal\s+(your\s+)?(prompt|instructions)|"
    r"you are now|jailbreak|developer mode|act as (an? )?(admin|developer|root)",
    re.I,
)


@dataclass
class InputResult:
    text: str
    flags: list[str] = field(default_factory=list)


def check_input(text: str) -> InputResult:
    flags: list[str] = []
    for name, pattern, repl in MASKS:
        text, n = pattern.subn(repl, text)
        if n:
            flags.append(f"pii_masked:{name}")
    if INJECTION.search(text):
        flags.append("prompt_injection")
    return InputResult(text=text, flags=flags)


# ----------------------------------------------------------------------------
# Output guardrail
# ----------------------------------------------------------------------------
EMAIL = re.compile(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*")
URL = re.compile(r"https?://\S+|www\.\S+", re.I)
PHONE = re.compile(r"\+?\d[\d\s().-]{7,}\d")
AMOUNT = re.compile(r"(?:rs\.?|₹|inr)\s*([\d,]+(?:\.\d+)?)", re.I)


def _digits(s: str) -> str:
    return re.sub(r"\D", "", s)


@lru_cache(maxsize=1)
def _allowed() -> dict:
    """Contact details and amounts that appear in our own documents."""
    text = ""
    for d in (config.KB_DIR, config.CATALOG_DIR):
        for f in d.glob("*.md"):
            text += f.read_text(encoding="utf-8") + "\n"
    return {
        "emails": {e.lower().rstrip(".") for e in EMAIL.findall(text)},
        "phones": {_digits(p) for p in PHONE.findall(text)},
        "amounts": {a.replace(",", "") for a in AMOUNT.findall(text)},
    }


@dataclass
class OutputResult:
    blocked: bool = False
    reasons: list[str] = field(default_factory=list)  # shown as escalation reasons
    flags: list[str] = field(default_factory=list)    # stored in the log


def check_output(reply: str) -> OutputResult:
    ok = _allowed()
    res = OutputResult()

    bad_email = [e for e in EMAIL.findall(reply) if e.lower().rstrip(".") not in ok["emails"]]
    bad_phone = [p for p in PHONE.findall(reply) if _digits(p) not in ok["phones"]]
    bad_url = URL.findall(reply)
    if bad_email or bad_phone or bad_url:
        res.blocked = True
        res.flags.append("output_blocked:unofficial_contact")
        res.reasons.append("Reply blocked: contained contact details or links not in our documents")

    bad_amount = sorted({a.replace(",", "") for a in AMOUNT.findall(reply)} - ok["amounts"])
    if bad_amount:
        res.flags.append("output_unverified_amount")
        res.reasons.append(f"Unverified amount in reply: Rs. {', '.join(bad_amount)}")
    return res