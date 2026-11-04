"""ShopEase Customer Support Desk: Streamlit frontend.  Run:  streamlit run app.py"""
import asyncio
import html
import io

import pandas as pd
import streamlit as st

from backend import config, ticket_log
from backend.agents.human_review import approve_reply
from backend.samples import SAMPLES
from backend.team import stream_ticket

st.set_page_config(page_title="ShopEase Support Desk", page_icon="🛍️", layout="wide")

AGENTS = {
    "query_agent": ("🔎", "Customer Query Agent"),
    "knowledge_agent": ("📚", "Knowledge Base Agent"),
    "resolution_agent": ("✍️", "Resolution Agent"),
    "log_agent": ("🗂️", "Log Agent"),
    "guardrail": ("🛡️", "Input Guardrail"),
}


# ----------------------------------------------------------------------------
# Rendering helpers
# ----------------------------------------------------------------------------
def render_event(ev: dict) -> None:
    icon, name = AGENTS.get(ev.get("agent", ""), ("🤖", ev.get("agent", "")))
    if ev["type"] == "tool_call":
        st.caption(f"{icon} {name} → calling tool `{ev['tool']}`")
    elif ev["type"] == "tool_result":
        with st.expander(f"🔧 Tool result · {ev['tool']} (RAG 1: company catalog)", expanded=False):
            st.text(ev["content"])
    elif ev["type"] == "guardrail":
        with st.expander(f"{icon} {name}", expanded=False):
            st.text(ev["content"])
    elif ev["type"] == "agent":
        with st.expander(f"{icon} {name}", expanded=True):
            st.text(ev["content"])
    elif ev["type"] == "error":
        st.error(f"Something went wrong: {ev['content']}")


def render_result(r: dict) -> None:
    st.subheader("Final reply")
    escalated = r["status"] == "Escalated"
    color, bg, label = ("#d32f2f", "#fdecea", "🔴 Escalated") if escalated else ("#2e7d32", "#e8f5e9", "🟢 Resolved")
    conf = "n/a" if r["confidence"] is None else f"{r['confidence']}%"
    st.markdown(
        f"<span style='background:{color};color:white;padding:4px 14px;border-radius:14px;"
        f"font-weight:600'>{label}</span> &nbsp; **{r['ticket_id']}** · {r['category']} · "
        f"Urgency **{r['urgency']}** · Sentiment **{r['sentiment']}** · Confidence **{conf}**",
        unsafe_allow_html=True,
    )
    reply_html = html.escape(r["reply"]).replace("\n", "<br>")
    st.markdown(
        f"<div style='background:{bg};border-left:6px solid {color};padding:16px 20px;"
        f"border-radius:6px;margin:10px 0;color:#222'>{reply_html}</div>",
        unsafe_allow_html=True,
    )
    if escalated:
        st.error(f"Escalated to human support: {r['reason']}")
    sources = ", ".join(r["sources"]) if r["sources"] else "none (no matching document)"
    st.caption(f"Sources used: {sources} · Business line: {r['business_line'] or '-'} · "
               f"Product: {r['product'] or '-'} · Best match score: {r['best_score']} · "
               f"Guardrail flags: {', '.join(r.get('guardrail_flags') or []) or 'none'}")


async def run_live(container, name: str, order: str, complaint: str) -> None:
    st.session_state.events, st.session_state.result = [], None
    async for ev in stream_ticket(name, order, complaint):
        if ev["type"] == "result":
            st.session_state.result = ev
        else:
            st.session_state.events.append(ev)
            with container:
                render_event(ev)


def apply_sample() -> None:
    s = SAMPLES.get(st.session_state.sample)
    if s:
        st.session_state.name, st.session_state.order, st.session_state.complaint = s


# ----------------------------------------------------------------------------
# Sidebar
# ----------------------------------------------------------------------------
with st.sidebar:
    st.title("🛍️ ShopEase")
    st.caption("AI Customer Support Desk")
    st.subheader("System status")
    kb_ready = (config.INDEX_DIR / "kb.index").exists()
    st.markdown(
        f"- LLM: `{config.CHAT_MODEL}`\n"
        f"- Embeddings: `{config.EMBEDDING_MODEL}`\n"
        f"- Vector DB: `FAISS` ({'index built' if kb_ready else 'builds on first ticket'})\n"
        f"- Team: `SelectorGroupChat`\n"
        f"- Stops on: `LOGGED` or {config.MAX_MESSAGES} messages\n"
        f"- API key: {'✅ loaded' if config.OPENAI_API_KEY else '❌ missing (.env)'}"
    )

st.session_state.setdefault("name", "")
st.session_state.setdefault("order", "")
st.session_state.setdefault("complaint", "")

tab_new, tab_log, tab_dash, tab_human = st.tabs(
    ["📝 New Ticket", "📒 Ticket Log", "📊 Dashboard", "🧑‍💼 Human Review"]
)

# ----------------------------------------------------------------------------
# Tab 1: new ticket
# ----------------------------------------------------------------------------
with tab_new:
    st.header("Submit a customer complaint")
    st.selectbox("Test with a sample", ["(type your own)"] + list(SAMPLES), key="sample", on_change=apply_sample)
    with st.container(border=True):
        c1, c2 = st.columns(2)
        c1.text_input("Customer name (optional)", key="name")
        c2.text_input("Order ID (optional)", key="order")
        st.text_area("Complaint", key="complaint", height=120)
        submit = st.button("Submit", type="primary")

    st.subheader("Agent timeline")
    timeline = st.container()
    if submit:
        if not st.session_state.complaint.strip():
            st.warning("Please enter a complaint first.")
        elif not config.OPENAI_API_KEY:
            st.error("OPENAI_API_KEY is missing. Add it to your .env file and restart.")
        else:
            asyncio.run(run_live(timeline, st.session_state.name, st.session_state.order,
                                 st.session_state.complaint))
    elif st.session_state.get("events"):
        with timeline:
            for ev in st.session_state.events:
                render_event(ev)
    if st.session_state.get("result"):
        render_result(st.session_state.result)

# ----------------------------------------------------------------------------
# Tab 2: ticket log
# ----------------------------------------------------------------------------
df = ticket_log.read_log()
with tab_log:
    st.header(f"Ticket log ({len(df)} tickets)")
    if df.empty:
        st.info("No tickets yet. Submit a complaint in the New Ticket tab.")
    else:
        f1, f2, f3 = st.columns(3)
        status_f = f1.multiselect("Status", sorted(df["status"].unique()))
        cat_f = f2.multiselect("Category", sorted(df["category"].unique()))
        search = f3.text_input("Search")
        view = df
        if status_f:
            view = view[view["status"].isin(status_f)]
        if cat_f:
            view = view[view["category"].isin(cat_f)]
        if search:
            view = view[view.apply(lambda r: search.lower() in " ".join(r.astype(str)).lower(), axis=1)]
        st.dataframe(view.iloc[::-1], hide_index=True)

        d1, d2 = st.columns(2)
        d1.download_button("⬇️ Download CSV", df.to_csv(index=False).encode("utf-8-sig"),
                           "tickets.csv", "text/csv")
        buf = io.BytesIO()
        df.to_excel(buf, index=False, engine="openpyxl")
        d2.download_button("⬇️ Download Excel", buf.getvalue(), "tickets.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

# ----------------------------------------------------------------------------
# Tab 3: dashboard
# ----------------------------------------------------------------------------
with tab_dash:
    st.header("Dashboard")
    if df.empty:
        st.info("The dashboard fills up once tickets are logged.")
    else:
        resolved = int((df["status"] == "Resolved").sum())
        escalated = int((df["status"] == "Escalated").sum())
        avg_urg = df["urgency"].map({"Low": 1, "Medium": 2, "High": 3}).mean()
        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Total tickets", len(df))
        m2.metric("Resolved by AI", resolved)
        m3.metric("Escalated", escalated, f"{escalated / len(df):.0%}", delta_color="off")
        m4.metric("Avg urgency (1-3)", f"{avg_urg:.2f}")
        left, right = st.columns(2)
        left.subheader("Tickets per category")
        left.bar_chart(pd.crosstab(df["category"], df["status"]))
        right.subheader("Resolved vs escalated")
        right.bar_chart(df["status"].value_counts())

# ----------------------------------------------------------------------------
# Tab 4: human review (human-in-the-loop for escalated tickets)
# ----------------------------------------------------------------------------
with tab_human:
    st.header("Escalated tickets waiting for a human")
    st.caption("A support person reviews the AI draft, edits it if needed and approves it. The approval "
               "runs through AutoGen's UserProxyAgent, and the approved reply and reviewer name are saved in the log.")
    if st.session_state.get("review_msg"):
        st.success(st.session_state.pop("review_msg"))
    if df.empty:
        st.info("No tickets yet.")
    else:
        esc = df[df["status"] == "Escalated"]
        pending = esc[esc["reviewed_by"] == ""]
        done = esc[esc["reviewed_by"] != ""]
        st.write(f"**{len(pending)}** waiting · **{len(done)}** handled")
        if pending.empty:
            st.success("Nothing waiting. All escalated tickets have been handled.")
        for _, t in pending.iloc[::-1].iterrows():
            tid = t["ticket_id"]
            with st.expander(f"{tid} · {t['category']} · {t['escalation_reason']}"):
                st.markdown(f"**Customer:** {t['customer_name'] or '-'} · **Order:** {t['order_id'] or '-'}")
                st.markdown(f"**Complaint:** {t['complaint']}")
                reply = st.text_area("Reply to send", t["response"], height=220, key=f"reply_{tid}")
                who = st.text_input("Your name", key=f"who_{tid}")
                if st.button("✅ Approve & mark handled", key=f"approve_{tid}", disabled=not who.strip()):
                    try:
                        approved, via_agent = approve_reply(
                            tid, t["complaint"], t["escalation_reason"], t["response"], reply)
                        ticket_log.update_review(tid, approved, who.strip())
                        st.session_state.review_msg = (
                            f"{tid} approved by {who.strip()} through AutoGen's UserProxyAgent."
                            if via_agent else
                            f"{tid} approved by {who.strip()} (saved directly: the UserProxyAgent step failed).")
                        st.rerun()
                    except PermissionError:
                        st.error("Could not save. Close tickets.csv if it is open in Excel and try again.")
        if not done.empty:
            st.subheader("Handled by a human")
            st.dataframe(done[["ticket_id", "category", "escalation_reason", "reviewed_by", "reviewed_at"]]
                         .iloc[::-1], hide_index=True)