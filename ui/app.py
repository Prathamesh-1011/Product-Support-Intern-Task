"""Streamlit demo UI for TAM and support teams."""

from __future__ import annotations

import json

import streamlit as st

st.set_page_config(page_title="Support & TAM AI Tools", page_icon="🎫", layout="wide")

st.title("Support & TAM AI Tools")
st.caption("Intelligent ticket triage and account health briefs for internal teams")

tab_triage, tab_tam = st.tabs(["Ticket Triage", "TAM Account Brief"])

with tab_triage:
    st.subheader("Task 1 · Intelligent Ticket Triage")
    col1, col2 = st.columns(2)
    with col1:
        subject = st.text_input("Subject", value="URGENT: DataBridge Connectors pipeline down")
    with col2:
        stream_mode = st.checkbox("Stream draft response", value=False)

    body = st.text_area(
        "Ticket body",
        height=200,
        value=(
            "Our Connectors pipeline has been failing since this morning. "
            "Error: ERR_CONNECTION_TIMEOUT after 30s. "
            "Impacting 47 users in production. No workaround available."
        ),
    )

    if st.button("Triage ticket", type="primary", key="triage_btn"):
        if stream_mode:
            from src.triage.models import TriageInput
            from src.triage.agent import triage_ticket_stream_text

            st.write("**Draft response (streaming):**")
            placeholder = st.empty()
            accumulated = ""
            try:
                for token in triage_ticket_stream_text(TriageInput(subject=subject, body=body)):
                    accumulated += token
                    placeholder.markdown(accumulated)
            except RuntimeError as exc:
                st.error(str(exc))
        else:
            from src.triage.models import TriageInput
            from src.triage.agent import triage_ticket

            try:
                with st.spinner("Triaging..."):
                    result = triage_ticket(TriageInput(subject=subject, body=body))
                st.success(f"Urgency: **{result.urgency}** · Team: **{result.recommended_team}**")
                c1, c2, c3 = st.columns(3)
                c1.metric("Product area", result.product_area)
                c2.metric("Category", result.issue_category)
                c3.metric("Prompt", result.prompt_version)
                st.markdown("**Reasoning**")
                st.info(result.reasoning)
                if result.kb_match:
                    st.markdown("**Knowledge-base match**")
                    st.json(result.kb_match.model_dump())
                st.markdown("**Draft first response**")
                st.write(result.draft_response)
                with st.expander("Full JSON"):
                    st.code(json.dumps(result.model_dump(mode="json"), indent=2))
            except RuntimeError as exc:
                st.error(str(exc))

with tab_tam:
    st.subheader("Task 2 · TAM Account Health Brief")
    from src.data.loader import build_account_map

    accounts = build_account_map()
    account_ids = sorted(accounts.keys())
    account_id = st.selectbox("Account ID", account_ids, index=0)
    stream_tam = st.checkbox("Stream executive summary", value=False, key="tam_stream")

    if st.button("Generate brief", type="primary", key="tam_btn"):
        if stream_tam:
            from src.tam.summarizer import tam_brief_stream_text

            st.write("**Executive summary (streaming):**")
            placeholder = st.empty()
            accumulated = ""
            try:
                for token in tam_brief_stream_text(account_id):
                    accumulated += token
                    placeholder.markdown(accumulated)
            except RuntimeError as exc:
                st.error(str(exc))
        else:
            from src.tam.summarizer import generate_tam_brief

            try:
                with st.spinner("Generating brief..."):
                    brief = generate_tam_brief(account_id)
                st.success(f"**{brief.company}** ({brief.account_id}) · {brief.ticket_count_90d} tickets (90d)")
                st.markdown("### Executive Summary")
                st.write(brief.executive_summary)
                st.markdown("### Open Risks")
                for risk in brief.open_risks:
                    st.warning(f"[{risk.severity.upper()}] {risk.risk} _(source: {risk.source})_")
                st.markdown("### Flagged Tickets")
                if brief.flagged_tickets:
                    for ft in brief.flagged_tickets:
                        st.error(f"**{ft.ticket_id}** ({ft.signal}): _\"{ft.quote}\"_ — {ft.justification}")
                else:
                    st.info("No churn/escalation flags detected.")
                st.markdown("### QBR Talking Points")
                for i, tp in enumerate(brief.talking_points, 1):
                    st.write(f"{i}. {tp}")
                st.caption(f"Content hash: `{brief.content_hash[:16]}...` · {brief.prompt_version}")
            except RuntimeError as exc:
                st.error(str(exc))

st.sidebar.markdown("### Quick links")
st.sidebar.markdown("- `python main.py serve` — API")
st.sidebar.markdown("- `python main.py eval --offline` — CI eval")
st.sidebar.markdown("- Set `GROQ_API_KEY` in `.env` for the default backend")
