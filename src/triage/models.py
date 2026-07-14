from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


IssueCategory = Literal[
    "Bug",
    "Feature Request",
    "How-To",
    "Performance",
    "Billing",
    "Integration",
    "Onboarding",
    "Data Loss",
]

UrgencyTier = Literal["P1", "P2", "P3", "P4"]

ResponderTeam = Literal[
    "tier-1-support",
    "tier-2-engineering",
    "integrations-team",
    "onboarding-team",
    "billing-team",
]


class KBMatch(BaseModel):
    doc_path: str
    title: str
    relevance: str


class TriageLLMOutput(BaseModel):
    product_area: str
    issue_category: IssueCategory
    urgency: UrgencyTier
    reasoning: str
    kb_match: KBMatch | None = None
    recommended_team: ResponderTeam
    draft_response: str


class TriageInput(BaseModel):
    subject: str = Field(..., min_length=1)
    body: str = Field(..., min_length=1)


class TriageOutput(BaseModel):
    product_area: str
    issue_category: IssueCategory
    urgency: UrgencyTier
    reasoning: str
    kb_match: KBMatch | None = None
    recommended_team: ResponderTeam
    draft_response: str
    retrieved_docs: list[str] = Field(default_factory=list)
    prompt_version: str


class RawTicketInput(BaseModel):
    """Accept raw text or structured ticket."""

    text: str | None = None
    subject: str | None = None
    body: str | None = None

    def to_triage_input(self) -> TriageInput:
        if self.subject and self.body:
            return TriageInput(subject=self.subject.strip(), body=self.body.strip())
        if self.text:
            lines = self.text.strip().split("\n", 1)
            subject = lines[0][:200]
            body = lines[1] if len(lines) > 1 else self.text
            return TriageInput(subject=subject, body=body.strip())
        raise ValueError("Provide either 'text' or both 'subject' and 'body'")
