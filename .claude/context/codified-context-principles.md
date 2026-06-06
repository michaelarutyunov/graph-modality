# Codified Context — Governance Principles

This document governs how the project's knowledge infrastructure is created, maintained, and evolved. It is loaded by any agent tasked with modifying CLAUDE.md, creating specialist agents, or writing context documents.

---

## Architecture

Three tiers, separated by loading frequency:

| Tier | Artifact | Loaded | Purpose |
|------|----------|--------|---------|
| 1 | `CLAUDE.md` | Every session | Conventions, routing, boundaries |
| 2 | `.claude/agents/{id}/AGENT.md` | Per task (via trigger table) | Domain-expert personas |
| 3 | `.claude/context/{topic}.md` | On demand | Detailed specifications |

**Invariant:** Tier 1 is the entry point. An agent that loads only CLAUDE.md must know enough to (a) follow all project conventions, (b) route itself to the correct Tier 2 agent, and (c) identify which Tier 3 docs to consult. If it cannot do all three, the constitution is incomplete.

---

## Tier 1: Constitution (CLAUDE.md)

### What belongs here

- Project identity (one paragraph: what this is, tech stack, repo structure)
- Architecture principles that are **non-negotiable and universal** — every session needs them
- Conventions: naming, file organization, build/run/test commands, branch strategy
- Agent trigger table: maps file patterns → specialist agents
- Key file map: 10–15 most important files with one-line descriptions
- Edge/type vocabularies or enums that multiple agents reference
- Cross-references to Tier 2 agents and Tier 3 docs (by path)

### What does NOT belong here

- Detailed specifications (move to Tier 3)
- Implementation guides or tutorials (move to Tier 3)
- Domain knowledge specific to one subsystem (move to Tier 2 agent)
- Rationale or history (move to ADRs or wiki)
- Anything that changes more than once per phase

### Size constraint

Target 400–700 lines. If it exceeds 800 lines, audit and extract content to Tier 2/3. The constitution must fit comfortably in a single context load alongside a meaningful task prompt.

### Update protocol

1. **Add a convention** only when an agent has violated it at least once, or when a new subsystem is introduced that all agents must know about.
2. **Remove a convention** when it no longer applies or has been superseded.
3. **Never duplicate** content that exists in a Tier 2 agent or Tier 3 doc. Reference it by path instead.
4. After any update, run `.claude/scripts/drift_check.py` to verify cross-references.

---

## Tier 2: Specialist Agents

### When to create a new agent

Create a specialist agent when ALL of these are true:

1. A distinct area of the codebase requires **domain knowledge that general-purpose prompting gets wrong** — not just file familiarity, but knowledge of constraints, failure modes, or design patterns specific to that subsystem.
2. The knowledge needed is **too large or too specific** to fit in CLAUDE.md without bloating it.
3. You can define a **clear trigger condition** — a file pattern, task type, or keyword that reliably identifies when this agent should be invoked.

Do NOT create an agent for:
- A one-off task (use an inline prompt instead)
- A subsystem that is simple enough to cover with a paragraph in CLAUDE.md
- A domain where you don't yet know the failure modes (wait until you do)

### Agent specification structure

Every agent MUST follow this structure:

```markdown
# {Agent Name}

## Role
One sentence: what this agent is responsible for.

## Trigger Conditions
When this agent is invoked (must match an entry in CLAUDE.md trigger table).

## Domain Knowledge
The core knowledge this agent needs to operate correctly.
This section is often >50% of the agent spec.
Embed key facts, formulas, patterns, and constraints directly —
do not rely solely on Tier 3 retrieval for critical knowledge.

## Key Constraints
Behavioral rules this agent must follow. Phrased as imperatives.

## Anti-patterns
Specific mistakes this agent must flag or avoid.
Each anti-pattern should reference a real failure that motivated it.

## Context Documents
List of Tier 3 docs this agent should consult, by path.
```

### Naming convention

- Directory: `.claude/agents/{kebab-case-id}/AGENT.md`
- The ID must be short and descriptive: `engine-specialist`, `block-developer`, `api-specialist`
- The trigger table in CLAUDE.md references agents by this ID

### Knowledge embedding principle

Agents should embed substantial domain knowledge directly in their spec — often over half the content. This creates intentional overlap with Tier 3 docs. The rationale:

- Agents operating in complex domains produce significantly more errors without pre-loaded context
- Retrieving from Tier 3 adds a step that may be skipped under context pressure
- Critical knowledge should be available without any retrieval step

The overlap is deliberate, not redundant. Tier 2 embeds the **operational subset** (what the agent needs to act correctly). Tier 3 contains the **full specification** (what the agent needs for edge cases and deep reasoning).

### Lifecycle

- **Creation**: Triggered by observed failure patterns, not upfront planning
- **Growth**: Add anti-patterns when new failure modes are discovered
- **Compaction**: If an agent spec exceeds ~1200 lines, split into a focused agent + a Tier 3 reference doc
- **Retirement**: Delete when the subsystem it covers is removed or merged

---

## Tier 3: Knowledge Base (Context Documents)

### When to create a context document

Create a context document when:

1. An agent has made a mistake on a topic that a specification would have prevented
2. A subsystem's behavior is complex enough that multiple agents need to reference it
3. An external integration or protocol requires documentation that doesn't exist elsewhere in the repo

The trigger is always **an observed problem**, not a planning exercise.

### Document structure

Optimized for AI consumption — structured, not prose:

```markdown
# {Topic} Specification

## Current Version: {version}

## {Primary Schema/Structure}
Table format with Field | Type | Required | Description columns.

## Rules
Numbered list. Each rule is testable and unambiguous.

## Examples
Concrete input/output pairs or code snippets.

## Known Failure Modes
Specific mistakes agents have made, with the correct behavior.
Each entry should be phrased as: "{wrong thing} → {consequence} → {correct approach}"
```

### Formatting rules

- **Tables over prose** for structured data (schemas, enums, field definitions)
- **Numbered rules over paragraphs** for behavioral constraints
- **Code blocks** for any syntax, schema, or pattern that must be exact
- **No narrative** — context docs are reference material, not tutorials
- Keep each doc under 500 lines. If it grows beyond that, split by subtopic.

### Naming convention

- Path: `.claude/context/{kebab-case-topic}.md`
- Name should be the subsystem or concept, not the problem that motivated it
- Good: execution-engine, edge-type-system, hitl-state-machine
- Bad: fix-executor-bug, how-edges-work

### Cross-referencing

- Context docs MUST reference the source files they describe (by path)
- Context docs SHOULD be referenced from at least one Tier 2 agent spec
- If a context doc is not referenced by any agent, it may be dead weight — verify or remove

---

## Trigger Table Maintenance

The trigger table in CLAUDE.md is the routing layer. It must satisfy these properties:

1. **Complete coverage**: Every directory in the codebase that an agent might modify should match at least one trigger pattern. Gaps mean agents operate without specialist guidance.
2. **No ambiguity**: Each file pattern should map to exactly one agent. If two agents could be invoked, either merge them or refine the patterns.
3. **Observable signals**: Triggers must be based on what's visible at task start — primarily which files are being modified. Not on intent, which is ambiguous.

When adding a new agent, always add its trigger pattern to the table simultaneously.

---

## Drift Detection

### What to validate

Run on every session start or as a pre-commit hook:

- [ ] Every file path in CLAUDE.md key file map exists on disk
- [ ] Every agent ID in the trigger table has a corresponding AGENT.md
- [ ] Every context doc referenced in an agent spec exists
- [ ] Every source file referenced in a context doc exists
- [ ] No context doc is orphaned (not referenced by any agent or CLAUDE.md)

### When drift is detected

1. If a referenced file has moved → update the reference
2. If a referenced file has been deleted → remove the reference and flag for review
3. If a new subsystem directory has no trigger match → flag as coverage gap
4. Log all drift findings at session start so the agent can self-correct before proceeding

---

## Growth Benchmarks

These are guidelines, not hard limits:

| Project Stage | Expected Tier 1 | Expected Tier 2 | Expected Tier 3 |
|---------------|-----------------|-----------------|-----------------|
| Phase 1 (skeleton) | ~400 lines | 2 agents | 2 docs |
| Phase 2 (execution engine) | ~500 lines | 4 agents | 5–8 docs |
| Phase 3+ (expansion) | ~600 lines | 5–7 agents | 8–15 docs |
| Mature (10k+ LOC) | ~700 lines | 6–10 agents | 10–20 docs |

Knowledge-to-code ratio benchmark: ~1 line of documentation per 4 lines of code (from Vasilopoulos 2026). This includes all three tiers combined.

If CLAUDE.md is growing but Tier 2/3 are not, you're putting too much in the constitution. Extract.

If Tier 3 is growing but agents aren't referencing the new docs, you're writing specs that won't be loaded. Connect them to agents or remove them.

---

## Self-Modification Protocol

When an agent modifies any part of the codified context infrastructure:

1. **State what changed and why** in the commit message
2. **Run drift detection** after the change
3. **Verify the constitution still fits** within the size constraint
4. **Do not modify Tier 1 and Tier 2/3 in the same action** — constitution changes should be reviewed separately from knowledge base changes, because constitution changes affect every future session

When an agent encounters a failure it could have avoided with better context:

1. Identify which tier should capture the lesson (convention → Tier 1, domain knowledge → Tier 2, specification → Tier 3)
2. Write the minimum addition that would have prevented the failure
3. Add a "Known Failure Modes" entry if modifying a Tier 3 doc
4. Add an "Anti-patterns" entry if modifying a Tier 2 agent
5. Add a convention only if the rule applies universally (Tier 1)
