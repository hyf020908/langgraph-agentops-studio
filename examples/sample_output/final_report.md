# LangGraph AgentOps Studio Report

## Task
Evaluate two orchestration options for a regulated automation workflow and produce a governance-ready recommendation with evidence references.

## Acceptance Criteria
- Recommendation references ranked evidence records.
- Governance section includes policy trigger details.
- Report captures unresolved questions and residual risks.

## Recommendation
Type: `conditional`

Summary: Proceed with a phased decision path while collecting additional validation evidence on operational risk and rollback guarantees.

Rationale: The top evidence set supports progress, but conflict severity and coverage gaps indicate additional validation is required before a final single-option commitment.

Confidence: 0.61
Supporting Evidence IDs: EVD-01, EVD-02, EVD-03
Unresolved Questions: Which rollback signal should block rollout?, Which data retention control must be validated first?
Residual Risks: risk:compliance-interpretation, risk:operational-rollout

## Governance Evaluation
Requires Human Review: True
Overall Risk Score: 0.69
Triggered Policies: overall_risk_threshold, recommendation_confidence_threshold, contradiction_severity_threshold
Risk Summary: overall_risk_score=0.69; recommendation_confidence=0.61; open_questions=2
Evidence Gaps: acceptance criteria coverage is below target
Contradiction Summary: max_conflict_severity=0.67, conflict_count=2
Required Human Action: Review governance triggers, validate unresolved risks, and provide explicit approval decision.

## Findings
### Governance Readiness
- Insight: Policy checks should gate final rollout approval.
- Rationale: Coverage and contradiction signals indicate decision pressure.
- Evidence IDs: EVD-01, EVD-03
- Risk level: medium
### Delivery Sequencing
- Insight: A phased release plan reduces exposure while unresolved questions are closed.
- Rationale: Evidence supports progress but not immediate full-scale rollout.
- Evidence IDs: EVD-02, EVD-03
- Risk level: medium

## Evidence Coverage
Query Coverage: 0.66
Criteria Coverage: 0.49
Evidence Count: 4
Coverage Notes: acceptance criteria coverage is below target

## Evidence Relations
Support Links: 3
Conflict Links: 2
Conflict Summary: SRC-02->SRC-03(0.67), SRC-03->SRC-02(0.67)

## Evidence Ledger
- `EVD-01`: Policy design guidance for regulated rollout decisions (confidence=0.74, sources=SRC-01)
  - overall_score: 0.74
  - score_breakdown: relevance=0.80, source_credibility=0.78, recency=0.72, completeness=0.70, corroboration=0.61, contradiction_penalty=0.22, extraction_quality=0.83, actionability=0.76
  - strengths: high query relevance, credible source profile, decision-useful content
  - weaknesses: none
  - flags: none
  - conflicts_with: SRC-03
  - supports: SRC-02
- `EVD-02`: Delivery sequencing analysis for controlled rollouts (confidence=0.69, sources=SRC-02)
  - overall_score: 0.69
  - score_breakdown: relevance=0.75, source_credibility=0.72, recency=0.68, completeness=0.66, corroboration=0.59, contradiction_penalty=0.18, extraction_quality=0.80, actionability=0.71
  - strengths: high query relevance, decision-useful content
  - weaknesses: limited recency
  - flags: none
  - conflicts_with: none
  - supports: SRC-01, SRC-04

## Source Register
- [Policy design guidance](https://example.com/policy-guide) | provider=tavily | source_type=webpage | score=0.77
- [Delivery sequencing analysis](https://example.com/rollout-sequencing) | provider=exa | source_type=webpage | score=0.73
- [Operational risk debate](https://example.com/risk-debate) | provider=tavily | source_type=web_search | score=0.66
- [Control validation checklist](https://example.com/control-checklist) | provider=qdrant | source_type=vector | score=0.71

## Reviewer Notes
Verdict: `escalate`

Summary: Manual review is required because governance thresholds are triggered.

Questions:
- Which validation criteria must be completed before approval?

Revision Requests:
- none

Major Risks:
- policy:overall_risk_threshold
- policy:contradiction_severity_threshold

## Workflow Outcome
Status: `approved_for_export`
