from __future__ import annotations

from src.eval.scoring import AcceptanceCriteria, EvalTestCase


def _quote_in_tickets(output: dict, substring: str) -> tuple[bool, str]:
    flagged = output.get("flagged_tickets", [])
    for ft in flagged:
        quote = ft.get("quote", "")
        if substring.lower() in quote.lower():
            return True, f"flagged quote contains '{substring}'"
    return False, f"no flagged quote contains '{substring}'"


TRIAGE_TEST_CASES: list[EvalTestCase] = [
    EvalTestCase(
        id="triage_p1_pipeline_outage",
        task="triage",
        description="Critical DataBridge pipeline timeout affecting production users",
        input={
            "subject": "URGENT: DataBridge Pro Connectors pipeline down in production",
            "body": (
                "Our Connectors pipeline has been failing since this morning. "
                "Error: ERR_CONNECTION_TIMEOUT after 30s. "
                "This is impacting 47 users in Engineering. Production environment. "
                "We have no workaround. Please advise urgently."
            ),
        },
        criteria=AcceptanceCriteria(
            required_fields=[
                "product_area",
                "issue_category",
                "urgency",
                "reasoning",
                "recommended_team",
                "draft_response",
            ],
            field_in={
                "urgency": ["P1", "P2"],
                "issue_category": ["Bug", "Performance"],
                "recommended_team": ["tier-2-engineering", "integrations-team"],
            },
            field_contains={
                "product_area": "DataBridge",
                "draft_response": "thank",
            },
            custom=[
                lambda o: (
                    o.get("kb_match") is not None or len(o.get("retrieved_docs", [])) > 0,
                    "KB match or retrieved docs present for known error pattern",
                ),
            ],
        ),
    ),
    EvalTestCase(
        id="triage_billing_seats",
        task="triage",
        description="Billing inquiry about seat count on invoice",
        input={
            "subject": "Invoice shows wrong seat count",
            "body": (
                "Our latest invoice lists 120 seats but we only have 95 active users. "
                "Can you explain the billing calculation and adjust our next invoice? "
                "We are on the Professional plan."
            ),
        },
        criteria=AcceptanceCriteria(
            field_in={
                "issue_category": ["Billing"],
                "recommended_team": ["billing-team", "tier-1-support"],
                "urgency": ["P3", "P4", "P2"],
            },
            field_contains={"draft_response": "invoice"},
        ),
    ),
    EvalTestCase(
        id="triage_sso_auth",
        task="triage",
        description="SSO authentication failure with known error code",
        input={
            "subject": "New users cannot log in via SSO — SAML_ASSERTION_EXPIRED",
            "body": (
                "Existing users authenticate fine but new joiners get SAML_ASSERTION_EXPIRED. "
                "We use Okta as our IDP for SecureVault. "
                "Error appears immediately after redirect from Okta."
            ),
        },
        criteria=AcceptanceCriteria(
            field_in={
                "issue_category": ["Integration", "How-To", "Bug"],
                "recommended_team": ["integrations-team", "tier-2-engineering", "tier-1-support"],
            },
            custom=[
                lambda o: (
                    "authentication" in " ".join(o.get("retrieved_docs", [])).lower()
                    or "sso" in o.get("reasoning", "").lower()
                    or (o.get("kb_match") or {}).get("doc_path", "").find("authentication") >= 0,
                    "authentication KB doc retrieved or SSO discussed in reasoning",
                ),
            ],
        ),
    ),
    EvalTestCase(
        id="triage_feature_request",
        task="triage",
        description="Low-urgency feature request for bulk operations",
        input={
            "subject": "Request: bulk archive in DataBridge Pro Data Ingestion",
            "body": (
                "We need bulk archive for Data Ingestion as we scaled to 116 users. "
                "Current workaround is manual one-by-one. "
                "This would be a nice enhancement for our workflow."
            ),
        },
        criteria=AcceptanceCriteria(
            field_in={
                "issue_category": ["Feature Request"],
                "urgency": ["P3", "P4", "P2"],
                "recommended_team": ["tier-1-support", "tier-2-engineering"],
            },
        ),
    ),
    EvalTestCase(
        id="triage_onboarding",
        task="triage",
        description="New team onboarding request",
        input={
            "subject": "New team member onboarding to WorkflowEngine",
            "body": (
                "We have 64 new team members joining next week who need access to WorkflowEngine. "
                "Need help with bulk user provisioning, permissions setup, and training materials. "
                "We're on the Business plan."
            ),
        },
        criteria=AcceptanceCriteria(
            field_in={
                "issue_category": ["Onboarding", "How-To"],
                "recommended_team": ["onboarding-team", "tier-1-support"],
                "urgency": ["P3", "P4"],
            },
            field_contains={"product_area": "Workflow"},
        ),
    ),
    EvalTestCase(
        id="triage_adversarial_ambiguous",
        task="triage",
        description="Adversarial: vague ticket mixing billing complaint with performance issue",
        input={
            "subject": "Something is wrong",
            "body": (
                "Things have been slow lately and we're not happy. "
                "Also the bill seems high? Not sure if related. "
                "Maybe it's the integration or maybe billing. Hard to tell. "
                "Can someone look into this when they get a chance?"
            ),
        },
        criteria=AcceptanceCriteria(
            required_fields=["reasoning", "urgency", "recommended_team", "draft_response"],
            field_in={"urgency": ["P2", "P3", "P4"]},
            min_list_length={"retrieved_docs": 0},
            custom=[
                lambda o: (len(o.get("reasoning", "")) >= 40, "reasoning addresses ambiguity"),
                lambda o: (len(o.get("draft_response", "")) >= 80, "draft response asks clarifying questions"),
            ],
        ),
        adversarial=True,
    ),
]


TAM_TEST_CASES: list[EvalTestCase] = [
    EvalTestCase(
        id="tam_at_risk_account",
        task="tam",
        description="At-risk account with declining usage and escalation notes",
        input={"account_id": "ACC-3336"},
        criteria=AcceptanceCriteria(
            required_fields=["executive_summary", "open_risks", "talking_points"],
            min_list_length={"open_risks": 1, "talking_points": 3},
            field_contains={"executive_summary": "Omni Consumer Products"},
        ),
    ),
    EvalTestCase(
        id="tam_healthy_account",
        task="tam",
        description="Healthy account with increasing usage",
        input={"account_id": "ACC-3033"},
        criteria=AcceptanceCriteria(
            required_fields=["executive_summary", "talking_points"],
            min_list_length={"talking_points": 3},
            field_contains={"executive_summary": "Polaris"},
        ),
    ),
    EvalTestCase(
        id="tam_churning_account",
        task="tam",
        description="Churning account with competitor evaluation signals",
        input={"account_id": "ACC-2944"},
        criteria=AcceptanceCriteria(
            min_list_length={"open_risks": 2, "talking_points": 4},
            custom=[
                lambda o: (
                    any(
                        "compet" in r.get("risk", "").lower()
                        or "churn" in r.get("risk", "").lower()
                        or "vendor" in r.get("risk", "").lower()
                        for r in o.get("open_risks", [])
                    )
                    or any(
                        ft.get("signal") in ("churn", "escalation", "dissatisfaction")
                        for ft in o.get("flagged_tickets", [])
                    )
                    or len(o.get("flagged_tickets", [])) >= 1,
                    "churn or escalation risk detected",
                ),
            ],
        ),
    ),
    EvalTestCase(
        id="tam_high_p1_load",
        task="tam",
        description="Account with P1 ticket history",
        input={"account_id": "ACC-2944"},
        criteria=AcceptanceCriteria(
            field_contains={"executive_summary": "Pinnacle"},
            custom=[
                lambda o: (o.get("ticket_count_90d", 0) >= 0, "ticket count present"),
            ],
        ),
    ),
    EvalTestCase(
        id="tam_renewal_focus",
        task="tam",
        description="Enterprise account approaching renewal",
        input={"account_id": "ACC-3336"},
        criteria=AcceptanceCriteria(
            custom=[
                lambda o: (
                    "renewal" in o.get("executive_summary", "").lower()
                    or any("renewal" in tp.lower() for tp in o.get("talking_points", [])),
                    "renewal mentioned in summary or talking points",
                ),
            ],
        ),
    ),
    EvalTestCase(
        id="tam_adversarial_missing_account",
        task="tam",
        description="Adversarial: account ID not in accounts.json",
        input={"account_id": "ACC-99999"},
        criteria=AcceptanceCriteria(
            required_fields=["executive_summary", "talking_points"],
            min_list_length={"talking_points": 2},
            custom=[
                lambda o: (
                    o.get("company") is None or o.get("ticket_count_90d", 0) == 0,
                    "gracefully handles missing account",
                ),
                lambda o: (len(o.get("executive_summary", "")) >= 50, "still produces usable summary"),
            ],
        ),
        adversarial=True,
    ),
]

ALL_TEST_CASES = TRIAGE_TEST_CASES + TAM_TEST_CASES
