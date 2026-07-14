from __future__ import annotations

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from src.config import settings


def load_tickets(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or settings.data_dir / "tickets.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_accounts(path: Path | None = None) -> list[dict[str, Any]]:
    path = path or settings.data_dir / "accounts.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def build_account_map(accounts: list[dict[str, Any]] | None = None) -> dict[str, dict[str, Any]]:
    accounts = accounts if accounts is not None else load_accounts()
    return {a["account_id"]: a for a in accounts}


def get_account(account_id: str, account_map: dict[str, dict[str, Any]] | None = None) -> dict[str, Any] | None:
    account_map = account_map or build_account_map()
    return account_map.get(account_id)


def parse_ticket_timestamp(value: str) -> datetime:
    return datetime.fromisoformat(value.replace("Z", "+00:00"))


def get_account_tickets(
    account_id: str,
    tickets: list[dict[str, Any]] | None = None,
    days: int = 90,
    reference: datetime | None = None,
) -> list[dict[str, Any]]:
    """Return tickets for an account within the last N days."""
    tickets = tickets if tickets is not None else load_tickets()
    if reference is None:
        reference = datetime.now(timezone.utc)
        if tickets:
            max_ticket_time = max(parse_ticket_timestamp(t["created_at"]) for t in tickets)
            if reference > max_ticket_time + timedelta(days=days):
                reference = max_ticket_time
    cutoff = reference - timedelta(days=days)
    return [
        t
        for t in tickets
        if t.get("account_id") == account_id
        and parse_ticket_timestamp(t["created_at"]) >= cutoff
    ]


def get_sample_ticket(ticket_id: str) -> dict[str, Any] | None:
    for ticket in load_tickets():
        if ticket["ticket_id"] == ticket_id:
            return ticket
    return None
