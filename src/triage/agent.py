from __future__ import annotations

from src.llm.client import LLMClient, get_llm
from src.prompts.registry import TRIAGE_PROMPT
from src.rag.retriever import get_retriever
from src.triage.models import TriageInput, TriageLLMOutput, TriageOutput


def triage_ticket(
    ticket: TriageInput,
    llm: LLMClient | None = None,
) -> TriageOutput:
    """
    Classify, route, and draft a first response for an incoming support ticket.

    Uses RAG over the knowledge base and structured LLM output.
    """
    llm = llm or get_llm()
    retriever = get_retriever()
    query = f"{ticket.subject}\n{ticket.body}"
    kb_context = retriever.format_context(query)
    retrieved = [c.doc_path for c in retriever.retrieve(query)]

    user_prompt = TRIAGE_PROMPT.user_template.format(
        subject=ticket.subject,
        body=ticket.body,
        kb_context=kb_context,
    )

    result = llm.complete_json(
        system=TRIAGE_PROMPT.system,
        user=user_prompt,
        schema=TriageLLMOutput,
    )

    return TriageOutput(
        product_area=result.product_area,
        issue_category=result.issue_category,
        urgency=result.urgency,
        reasoning=result.reasoning,
        kb_match=result.kb_match,
        recommended_team=result.recommended_team,
        draft_response=result.draft_response,
        retrieved_docs=sorted(set(retrieved)),
        prompt_version=f"{TRIAGE_PROMPT.id}@{TRIAGE_PROMPT.version}",
    )


def triage_ticket_stream_text(ticket: TriageInput, llm: LLMClient | None = None):
    """Stream the draft response section for demo purposes."""
    llm = llm or get_llm()
    retriever = get_retriever()
    query = f"{ticket.subject}\n{ticket.body}"
    kb_context = retriever.format_context(query)

    system = (
        TRIAGE_PROMPT.system
        + "\n\nWrite ONLY the draft first-response message for the support agent. "
        "Do not include JSON or labels."
    )
    user = TRIAGE_PROMPT.user_template.format(
        subject=ticket.subject,
        body=ticket.body,
        kb_context=kb_context,
    )
    yield from llm.stream_text(system=system, user=user)
