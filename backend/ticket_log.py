"""CSV ticket log: one row per ticket, created automatically."""
from __future__ import annotations

import csv
import threading
from datetime import datetime

import pandas as pd

from backend import config

COLUMNS = [
    "ticket_id", "timestamp", "customer_name", "order_id", "complaint",
    "category", "urgency", "sentiment", "summary", "kb_sources",
    "response", "confidence", "status", "escalation_reason", "guardrail_flags",
    "reviewed_by", "reviewed_at",
]
_lock = threading.Lock()


def _next_id() -> str:
    last = 0
    if config.LOG_FILE.exists():
        try:
            ids = pd.read_csv(config.LOG_FILE, usecols=["ticket_id"], dtype=str, encoding="utf-8-sig")
            nums = ids["ticket_id"].str.extract(r"(\d+)")[0].dropna().astype(int)
            last = int(nums.max()) if len(nums) else 0
        except Exception:
            last = 0
    return f"TKT-{last + 1:04d}"


def _migrate_if_needed() -> None:
    """If an older log has fewer columns, rewrite it with the current columns."""
    if not config.LOG_FILE.exists():
        return
    old = pd.read_csv(config.LOG_FILE, dtype=str, encoding="utf-8-sig").fillna("")
    if list(old.columns) != COLUMNS:
        old.reindex(columns=COLUMNS, fill_value="").to_csv(config.LOG_FILE, index=False, encoding="utf-8-sig")


def append_row(row: dict) -> str:
    """Append a ticket and return its ticket_id."""
    with _lock:
        config.LOG_DIR.mkdir(parents=True, exist_ok=True)
        ticket_id = _next_id()
        row = {
            **row,
            "ticket_id": ticket_id,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        }
        _migrate_if_needed()
        new_file = not config.LOG_FILE.exists()
        # utf-8-sig on creation so Excel shows non-English text correctly
        with open(config.LOG_FILE, "w" if new_file else "a", newline="",
                  encoding="utf-8-sig" if new_file else "utf-8") as f:
            w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
            if new_file:
                w.writeheader()
            w.writerow(row)
        return ticket_id


def update_review(ticket_id: str, response: str, reviewed_by: str) -> bool:
    """Human-in-the-loop: store the support person's (edited) reply and who approved it."""
    with _lock:
        _migrate_if_needed()
        df = pd.read_csv(config.LOG_FILE, dtype=str, encoding="utf-8-sig").fillna("")
        mask = df["ticket_id"] == ticket_id
        if not mask.any():
            return False
        df.loc[mask, "response"] = response
        df.loc[mask, "reviewed_by"] = reviewed_by
        df.loc[mask, "reviewed_at"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        df.to_csv(config.LOG_FILE, index=False, encoding="utf-8-sig")
        return True


def read_log() -> pd.DataFrame:
    if not config.LOG_FILE.exists():
        return pd.DataFrame(columns=COLUMNS)
    df = pd.read_csv(config.LOG_FILE, dtype=str, encoding="utf-8-sig").fillna("")
    for col in COLUMNS:  # logs written by older versions may lack newer columns
        if col not in df.columns:
            df[col] = ""
    return df