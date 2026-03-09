# Governance and Human Approval Policy

## Policy Inputs

Governance policy is loaded from `config/defaults.yaml` and environment variables:

- overall risk threshold
- recommendation confidence threshold
- evidence completeness threshold
- contradiction severity threshold
- unresolved question count threshold
- high-stakes task categories
- manual approval policy by task type

## Evaluation Inputs

`services/governance.py` evaluates:

- structured recommendation output
- coverage record
- evidence assessments
- conflict records
- task type and user request context

## Governance Output

The governance result includes:

- `requires_human_review`
- `triggered_policies`
- `risk_summary`
- `evidence_gaps`
- `contradiction_summary`
- `recommendation_confidence`
- `required_human_action`
- `overall_risk_score`

## Decision Path Integration

- Analyst writes recommendation and governance output to shared state.
- Reviewer consumes governance output and confidence thresholds.
- Human approval interrupt payload includes governance trigger details.
- Supervisor routing can dispatch to `human_review` when governance requires manual approval.
