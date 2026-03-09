# Evidence Evaluation

## Assessment Dimensions

`services/evidence.py` evaluates each source across:

1. relevance
2. source credibility
3. recency
4. completeness/depth
5. corroboration/agreement
6. contradiction/conflict penalty
7. extraction quality
8. actionability

## Structured Outputs

The pipeline produces:

- `EvidenceAssessment`
- `EvidenceScoreBreakdown`
- `ConflictRecord`
- `SupportRecord`
- `CoverageRecord`

Each evidence item can include:

- overall score
- score breakdown
- strengths and weaknesses
- flags
- conflict links
- support links

## Coverage and Relations

Coverage scoring tracks:

- query coverage
- acceptance criteria coverage
- evidence count
- coverage notes

Relation scoring tracks support and conflict links between sources and feeds those signals into recommendation and governance stages.
