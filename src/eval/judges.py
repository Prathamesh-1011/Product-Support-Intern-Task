from __future__ import annotations

import json
from typing import Any

from src.llm.client import LLMClient
from src.triage.models import TriageInput


class LLMJudge:
    """Optional LLM-as-judge for qualitative acceptance."""

    def __init__(self, llm: LLMClient | None = None):
        self.llm = llm

    def score(self, prompt: str, output: dict[str, Any]) -> tuple[float, str]:
        if self.llm is None:
            return 1.0, "LLM judge skipped (no client)"

        system = (
            "You are an evaluation judge for an AI support tooling pipeline. "
            "Score the output quality from 0.0 to 1.0 based on the criteria. "
            'Respond with JSON: {"score": float, "reasoning": string}'
        )
        user = f"Criteria:\n{prompt}\n\nOutput:\n{json.dumps(output, indent=2)}"

        try:
            response = self.llm.client.chat.completions.create(
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
                response_format={"type": "json_object"},
                temperature=0,
                seed=42,
                model=self.llm.model,
            )
            raw = json.loads(response.choices[0].message.content or "{}")
            return float(raw.get("score", 0.5)), str(raw.get("reasoning", ""))
        except Exception as exc:
            return 0.5, f"LLM judge error: {exc}"


def run_triage_for_eval(subject: str, body: str, llm: LLMClient) -> dict[str, Any]:
    from src.triage.agent import triage_ticket

    result = triage_ticket(TriageInput(subject=subject, body=body), llm=llm)
    return result.model_dump(mode="json")


def run_tam_for_eval(account_id: str, llm: LLMClient) -> dict[str, Any]:
    from src.tam.summarizer import generate_tam_brief

    result = generate_tam_brief(account_id, llm=llm)
    return result.model_dump(mode="json")
