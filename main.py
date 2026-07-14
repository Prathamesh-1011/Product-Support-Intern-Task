#!/usr/bin/env python3
"""Single entry point for all tasks."""

from __future__ import annotations

import argparse
import json
import sys


def cmd_serve(args: argparse.Namespace) -> int:
    import uvicorn

    uvicorn.run("src.api.main:app", host=args.host, port=args.port, reload=args.reload)
    return 0


def cmd_triage(args: argparse.Namespace) -> int:
    from src.triage.agent import triage_ticket
    from src.triage.models import TriageInput

    result = triage_ticket(TriageInput(subject=args.subject, body=args.body))
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


def cmd_tam(args: argparse.Namespace) -> int:
    from src.tam.summarizer import generate_tam_brief

    result = generate_tam_brief(args.account_id)
    print(json.dumps(result.model_dump(mode="json"), indent=2))
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    from src.eval.harness import run_eval_harness, write_report

    report = run_eval_harness(use_llm=not args.offline)
    json_path, md_path = write_report(report)
    print(json.dumps(report.to_dict(), indent=2))
    print(f"\nReports written to {json_path} and {md_path}", file=sys.stderr)
    return 0 if report.passed_cases == report.total_cases else 1


def cmd_ui(_: argparse.Namespace) -> int:
    import subprocess

    subprocess.run(
        [sys.executable, "-m", "streamlit", "run", "ui/app.py", "--server.headless", "true"],
        check=True,
    )
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(
        description="US Delivery Internship — Support & TAM AI Tools",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    serve = sub.add_parser("serve", help="Start FastAPI server")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8000)
    serve.add_argument("--reload", action="store_true")
    serve.set_defaults(func=cmd_serve)

    triage = sub.add_parser("triage", help="Run ticket triage (Task 1)")
    triage.add_argument("--subject", required=True)
    triage.add_argument("--body", required=True)
    triage.set_defaults(func=cmd_triage)

    tam = sub.add_parser("tam", help="Generate TAM account brief (Task 2)")
    tam.add_argument("--account-id", required=True)
    tam.set_defaults(func=cmd_tam)

    ev = sub.add_parser("eval", help="Run evaluation harness (Task 3)")
    ev.add_argument(
        "--offline",
        action="store_true",
        help="Run structural checks only (no API key required)",
    )
    ev.set_defaults(func=cmd_eval)

    ui = sub.add_parser("ui", help="Launch Streamlit demo UI (bonus)")
    ui.set_defaults(func=cmd_ui)

    args = parser.parse_args()
    try:
        return args.func(args)
    except RuntimeError as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
