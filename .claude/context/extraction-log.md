# extraction-log.md

> **Purpose:** track extraction prompt version history, model comparison results, extraction statistics, and any deviations or incidents during graph extraction. This is the canonical record for Phase 1.

---

## prompt version history

| version | file | date | change summary | rationale |
|---|---|---|---|---|
| v1 | `extraction/prompts/v1.txt` | 2026-06-07 | Initial extraction prompt | Based on ENGINEERING.md §5.2. Four entity types (Construct, Value, Stance, CognitiveStyleMarker), four relation types (SERVES, EXPRESSED_VIA, MODULATED_BY, CONFLICTS_WITH). Human-turn-only node extraction. Maximum 2 CSM per transcript. |

---

## dataset verification

| date | finding |
|---|---|
| 2026-06-07 | Confirmed dataset uses `Assistant:` (opening turn), `AI:` (subsequent AI turns), `User:` (human turns) — NOT `Human:` as ENGINEERING.md §5.1 originally stated. ENGINEERING.md to be corrected. |
| 2026-06-07 | 1,250 transcripts confirmed: workforce 1,000, creatives 125, scientists 125. Turns range: 15–55 per transcript. |

---

## model comparison

> *To be populated after the comparison experiment (`graph-modality-7xc`).*

| model | R1 (size consistency) | R2 (ontology) | R3 (bipolarity) | R4 (hallucination) | R5 (relation typing) | total |
|---|---|---|---|---|---|---|
| — | — | — | — | — | — | — |

**Winner:** TBD  
**Tiebreaker used:** TBD (R3 — bipolarity capture — is the tiebreaker per CHARTER.md §4)

**Sample IDs:** `extraction/model_comparison/sample_ids.txt` (4 workforce, 3 creatives, 3 scientists)

---

## scale extraction stats

> *To be populated after scale extraction (`graph-modality-5jq`).*

| metric | value |
|---|---|
| transcripts processed | — |
| successful extractions | — |
| failed extractions | — |
| invalid graphs (validator violations) | — |
| mean nodes per graph | — |
| mean edges per graph | — |
| mean bipolarity score | — |
| CONFLICTS_WITH edges found | — |

---

## incidents and deviations

> *Record any unexpected behaviour, API issues, or decisions that depart from the plan.*

None yet.
