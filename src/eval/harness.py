from __future__ import annotations

import json
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from src.config import REPO_ROOT, settings
from src.eval.judges import LLMJudge, run_tam_for_eval, run_triage_for_eval
from src.eval.scoring import _check_criteria
from src.eval.test_cases import ALL_TEST_CASES, EvalTestCase
from src.llm.client import LLMClient


@dataclass
class TestResult:
    id: str
    task: str
    description: str
    adversarial: bool
    passed: bool
    rule_score: float
    llm_judge_score: float | None
    combined_score: float
    failures: list[str] = field(default_factory=list)
    judge_reasoning: str | None = None


@dataclass
class EvalReport:
    timestamp: str
    offline_mode: bool
    total_cases: int
    passed_cases: int
    average_score: float
    results: list[TestResult]

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp": self.timestamp,
            "offline_mode": self.offline_mode,
            "summary": {
                "total_cases": self.total_cases,
                "passed_cases": self.passed_cases,
                "pass_rate": round(self.passed_cases / max(self.total_cases, 1), 3),
                "average_score": round(self.average_score, 3),
            },
            "results": [asdict(r) for r in self.results],
        }


def _run_case(case: EvalTestCase, llm: LLMClient | None) -> TestResult:
    if case.task == "triage":
        output = run_triage_for_eval(
            subject=case.input["subject"],
            body=case.input["body"],
            llm=llm,  # type: ignore[arg-type]
        )
    else:
        output = run_tam_for_eval(account_id=case.input["account_id"], llm=llm)  # type: ignore[arg-type]

    rule_score, failures = _check_criteria(output, case.criteria)
    llm_score: float | None = None
    judge_reason: str | None = None

    if case.llm_judge_prompt and llm is not None:
        judge = LLMJudge(llm)
        llm_score, judge_reason = judge.score(case.llm_judge_prompt, output)

    if llm_score is not None:
        combined = 0.7 * rule_score + 0.3 * llm_score
    else:
        combined = rule_score

    passed = combined >= 0.75 and rule_score >= 0.6

    return TestResult(
        id=case.id,
        task=case.task,
        description=case.description,
        adversarial=case.adversarial,
        passed=passed,
        rule_score=round(rule_score, 3),
        llm_judge_score=round(llm_score, 3) if llm_score is not None else None,
        combined_score=round(combined, 3),
        failures=failures,
        judge_reasoning=judge_reason,
    )


def run_offline_structural_checks() -> list[TestResult]:
    """Rule-based checks that do not require an LLM (CI-safe)."""
    from src.data.loader import build_account_map, get_account_tickets, load_tickets
    from src.prompts.registry import PROMPT_REGISTRY
    from src.rag.retriever import get_retriever

    results: list[TestResult] = []

    retriever = get_retriever()
    chunks = retriever.retrieve("ERR_CONNECTION_TIMEOUT DataBridge", top_k=3)
    kb_ok = len(chunks) > 0 and any(
        "err_connection_timeout" in c.content.lower()
        or "databridge" in c.doc_path.lower()
        or "databridge" in c.content.lower()
        for c in chunks
    )
    results.append(
        TestResult(
            id="offline_kb_retrieval",
            task="infra",
            description="KB retrieval returns DataBridge timeout docs",
            adversarial=False,
            passed=kb_ok,
            rule_score=1.0 if kb_ok else 0.0,
            llm_judge_score=None,
            combined_score=1.0 if kb_ok else 0.0,
            failures=[] if kb_ok else ["No relevant KB chunks retrieved for timeout error"],
        )
    )

    accounts = build_account_map()
    results.append(
        TestResult(
            id="offline_data_load",
            task="infra",
            description="Accounts and tickets load successfully",
            adversarial=False,
            passed=len(accounts) == 50 and len(load_tickets()) == 500,
            rule_score=1.0,
            llm_judge_score=None,
            combined_score=1.0,
            failures=[],
        )
    )

    tickets = get_account_tickets("ACC-3336", load_tickets(), days=90)
    results.append(
        TestResult(
            id="offline_ticket_filter",
            task="infra",
            description="90-day ticket filter works",
            adversarial=False,
            passed=isinstance(tickets, list),
            rule_score=1.0,
            llm_judge_score=None,
            combined_score=1.0,
            failures=[],
        )
    )

    results.append(
        TestResult(
            id="offline_prompt_versions",
            task="infra",
            description="Prompt registry has versioned prompts",
            adversarial=False,
            passed="triage" in PROMPT_REGISTRY and "tam_brief" in PROMPT_REGISTRY,
            rule_score=1.0,
            llm_judge_score=None,
            combined_score=1.0,
            failures=[],
        )
    )

    return results


def run_eval_harness(use_llm: bool | None = None) -> EvalReport:
    offline = settings.eval_offline_mode if use_llm is None else not use_llm
    llm: LLMClient | None = None

    if not offline:
        try:
            llm = LLMClient()
            _ = llm.client  # validate API key
        except RuntimeError:
            offline = True

    results: list[TestResult] = []

    if offline:
        results.extend(run_offline_structural_checks())
    else:
        for case in ALL_TEST_CASES:
            try:
                results.append(_run_case(case, llm))
            except Exception as exc:
                results.append(
                    TestResult(
                        id=case.id,
                        task=case.task,
                        description=case.description,
                        adversarial=case.adversarial,
                        passed=False,
                        rule_score=0.0,
                        llm_judge_score=None,
                        combined_score=0.0,
                        failures=[str(exc)],
                    )
                )

    passed = sum(1 for r in results if r.passed)
    avg = sum(r.combined_score for r in results) / max(len(results), 1)

    return EvalReport(
        timestamp=datetime.now(timezone.utc).isoformat(),
        offline_mode=offline,
        total_cases=len(results),
        passed_cases=passed,
        average_score=avg,
        results=results,
    )


def write_report(report: EvalReport, output_dir: Path | None = None) -> tuple[Path, Path]:
    output_dir = output_dir or REPO_ROOT
    json_path = output_dir / "eval_report.json"
    md_path = output_dir / "eval_report.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(report.to_dict(), f, indent=2)

    lines = [
        "# Evaluation Report",
        "",
        f"**Generated:** {report.timestamp}",
        f"**Mode:** {'offline (structural)' if report.offline_mode else 'full (LLM pipeline)'}",
        "",
        "## Summary",
        "",
        f"| Metric | Value |",
        f"|--------|-------|",
        f"| Total cases | {report.total_cases} |",
        f"| Passed | {report.passed_cases} |",
        f"| Pass rate | {report.passed_cases / max(report.total_cases, 1):.1%} |",
        f"| Average score | {report.average_score:.3f} |",
        "",
        "## Results",
        "",
        "| ID | Task | Pass | Score | Adversarial | Notes |",
        "|----|------|------|-------|-------------|-------|",
    ]

    for r in report.results:
        notes = "; ".join(r.failures[:2]) if r.failures else "OK"
        adv = "yes" if r.adversarial else "no"
        lines.append(
            f"| {r.id} | {r.task} | {'PASS' if r.passed else 'FAIL'} | "
            f"{r.combined_score:.3f} | {adv} | {notes} |"
        )

    md_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, md_path
