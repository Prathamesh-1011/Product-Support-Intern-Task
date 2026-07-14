from __future__ import annotations

import json

from src.data.loader import build_account_map, get_account, get_account_tickets, load_tickets
from src.llm.client import LLMClient, get_llm
from src.prompts.registry import TAM_BRIEF_PROMPT
from src.tam.models import TAMBriefLLMOutput, TAMBriefOutput, canonical_brief_hash


def _compact_ticket(t: dict) -> dict:
    return {
        "ticket_id": t["ticket_id"],
        "subject": t["subject"],
        "body": t["body"],
        "product": t.get("product"),
        "category": t.get("category"),
        "urgency": t.get("urgency"),
        "status": t.get("status"),
        "created_at": t.get("created_at"),
    }


def generate_tam_brief(
    account_id: str,
    llm: LLMClient | None = None,
    days: int = 90,
) -> TAMBriefOutput:
    """
    Generate a deterministic TAM account brief from account data and recent tickets.

    Temperature is fixed at 0 and seed is set via config for reproducibility.
    """
    llm = llm or get_llm()
    account_map = build_account_map()
    account = get_account(account_id, account_map)
    tickets = get_account_tickets(account_id, load_tickets(), days=days)

    account_payload = account or {
        "account_id": account_id,
        "note": "No matching account record in accounts.json — synthesize from tickets only.",
    }

    user_prompt = TAM_BRIEF_PROMPT.user_template.format(
        account_json=json.dumps(account_payload, indent=2, sort_keys=True),
        tickets_json=json.dumps([_compact_ticket(t) for t in tickets], indent=2, sort_keys=True),
        ticket_count=len(tickets),
    )

    result = llm.complete_json(
        system=TAM_BRIEF_PROMPT.system,
        user=user_prompt,
        schema=TAMBriefLLMOutput,
    )

    return TAMBriefOutput(
        account_id=account_id,
        company=account.get("company") if account else None,
        executive_summary=result.executive_summary,
        open_risks=result.open_risks,
        flagged_tickets=result.flagged_tickets,
        talking_points=result.talking_points,
        ticket_count_90d=len(tickets),
        prompt_version=f"{TAM_BRIEF_PROMPT.id}@{TAM_BRIEF_PROMPT.version}",
        content_hash=canonical_brief_hash(result),
    )


def tam_brief_stream_text(account_id: str, llm: LLMClient | None = None, days: int = 90):
    """Stream the executive summary for demo purposes."""
    llm = llm or get_llm()
    account_map = build_account_map()
    account = get_account(account_id, account_map)
    tickets = get_account_tickets(account_id, load_tickets(), days=days)

    account_payload = account or {"account_id": account_id, "note": "Account not found in dataset."}

    system = (
        TAM_BRIEF_PROMPT.system
        + "\n\nWrite ONLY the executive summary section (3-5 sentences). Plain text, no JSON."
    )
    user = TAM_BRIEF_PROMPT.user_template.format(
        account_json=json.dumps(account_payload, indent=2, sort_keys=True),
        tickets_json=json.dumps([_compact_ticket(t) for t in tickets], indent=2, sort_keys=True),
        ticket_count=len(tickets),
    )
    yield from llm.stream_text(system=system, user=user)
