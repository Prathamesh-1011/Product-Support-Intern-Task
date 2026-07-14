from __future__ import annotations

import json
from collections.abc import AsyncIterator

from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from src.tam.summarizer import generate_tam_brief, tam_brief_stream_text
from src.triage.agent import triage_ticket, triage_ticket_stream_text
from src.triage.models import RawTicketInput, TriageInput, TriageOutput
from src.tam.models import TAMBriefOutput

app = FastAPI(
    title="Support & TAM AI Tools",
    description="Intelligent ticket triage and TAM account health summariser",
    version="1.0.0",
)


class HealthResponse(BaseModel):
    status: str
    prompts: dict[str, str]


@app.get("/health", response_model=HealthResponse)
def health() -> HealthResponse:
    from src.prompts.registry import PROMPT_REGISTRY

    return HealthResponse(
        status="ok",
        prompts={k: f"{v.id}@{v.version}" for k, v in PROMPT_REGISTRY.items()},
    )


@app.post("/triage", response_model=TriageOutput)
def triage_endpoint(payload: RawTicketInput) -> TriageOutput:
    try:
        ticket = payload.to_triage_input()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc
    try:
        return triage_ticket(ticket)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post("/triage/stream")
def triage_stream_endpoint(payload: RawTicketInput):
    try:
        ticket = payload.to_triage_input()
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    def event_stream():
        try:
            for token in triage_ticket_stream_text(ticket):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except RuntimeError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/tam/{account_id}", response_model=TAMBriefOutput)
def tam_brief_endpoint(account_id: str) -> TAMBriefOutput:
    try:
        return generate_tam_brief(account_id)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.get("/tam/{account_id}/stream")
def tam_stream_endpoint(account_id: str):
    def event_stream():
        try:
            for token in tam_brief_stream_text(account_id):
                yield f"data: {json.dumps({'token': token})}\n\n"
            yield "data: [DONE]\n\n"
        except RuntimeError as exc:
            yield f"data: {json.dumps({'error': str(exc)})}\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")
