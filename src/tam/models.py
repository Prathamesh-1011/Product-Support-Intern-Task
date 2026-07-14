from __future__ import annotations

import hashlib
import json
from typing import Literal

from pydantic import BaseModel, Field


RiskSeverity = Literal["high", "medium", "low"]
SignalType = Literal["churn", "escalation", "dissatisfaction", "critical_incident"]


class OpenRisk(BaseModel):
    risk: str
    severity: RiskSeverity
    source: Literal["account", "ticket", "escalation_note"]


class FlaggedTicket(BaseModel):
    ticket_id: str
    signal: SignalType
    quote: str
    justification: str


class TAMBriefLLMOutput(BaseModel):
    executive_summary: str
    open_risks: list[OpenRisk]
    flagged_tickets: list[FlaggedTicket]
    talking_points: list[str]


class TAMBriefOutput(BaseModel):
    account_id: str
    company: str | None
    executive_summary: str
    open_risks: list[OpenRisk]
    flagged_tickets: list[FlaggedTicket]
    talking_points: list[str]
    ticket_count_90d: int
    prompt_version: str
    content_hash: str = Field(
        description="SHA-256 of canonical output for determinism verification"
    )


def canonical_brief_hash(brief: TAMBriefLLMOutput) -> str:
    """Stable hash for determinism checks (excludes metadata)."""
    payload = brief.model_dump(mode="json")
    canonical = json.dumps(payload, sort_keys=True, ensure_ascii=True)
    return hashlib.sha256(canonical.encode()).hexdigest()
