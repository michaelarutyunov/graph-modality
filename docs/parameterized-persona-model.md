# Parameterized Persona Model for Multi-Modality Synthetic Data

> **Status:** Design document — not implemented
> **Date:** 2026-07-04  ·  revised 2026-07-04 (critique incorporation)
> **Related:** `s7_synthetic/synthetic_demo.py` (DPI logic demonstration); ADR-0006/0007 (real-data verdict: distributional, not relational); ADR-0008 (delegation-boundary ontology, never extracted)

---

## Purpose

This document defines a **parameterized persona model** capable of driving two independent
generative processes — a text generator (narrative interview transcript) and a graph generator
(delegation-boundary structure) — from a shared latent parameter vector. The two generators
share only the persona parameters; neither sees the other's output. This preserves the
data-processing inequality (DPI) guarantee: the graph is not derived from the text, so
`I(Graph; Label) ≤ I(Text; Label)` does not constrain the result. Any complementary signal the
graph adds over text comes from the shared personality latent, accessed through different
expressive registers.

**Design principle (critique-incorporated):** No parameter that directly shapes graph output
is hidden from the text generator. Where the channels differ, the difference is either:
(a) the same psychological variable expressed through different representational mechanisms
(e.g., `internal_boundary_clarity` → fewer hedges in text, sharper partition in graph), or
(b) a property of the elicitation format itself, not hidden psychology (e.g., a card-sort task
inherently forces decomposition that free narrative does not). This prevents the "latent
leakage" trap: graph advantage, if observed, emerges from representational differences, not
from information deliberately withheld from one channel.

**This is a controlled fiction.** No claim is made that generated transcripts resemble real
interviews or that generated graphs resemble real human elicitations. The model is a
falsifiable framework: it defines *which* psychological mechanisms would drive complementarity,
parameterizing the space of possible findings so that claims can be tested against human data
if such data is ever collected.

---

## 1. The persona parameter schema

The persona is a vector of 32 latent variables organized in four layers.

### Layer 0 — Role context (shared by both generators)

These parameters determine *what work the person does*. They constrain the task repertoire
before values or dispositions have any say. This reflects the reality that occupation, not
personality, is the primary determinant of which tasks a person faces.

#### Profession

A categorical variable drawn from a small taxonomy of professional archetypes:

| Profession | Typical task areas | Description |
|---|---|---|
| `clinician` | clinical, analytic, interpersonal | Doctors, nurses, therapists |
| `creative` | creative, interpersonal, learning | Writers, designers, artists |
| `analyst` | analytic, administrative, learning | Data scientists, researchers, finance |
| `manager` | interpersonal, administrative, analytic | Team leads, project managers, executives |
| `administrator` | administrative, interpersonal | Office managers, coordinators, support staff |
| `technician` | analytic, clinical, administrative | Lab techs, IT support, engineers |
| `educator` | interpersonal, learning, creative | Teachers, trainers, academics |

#### Seniority and organizational context (each 0–1)

| Parameter | Range | Effect |
|---|---|---|
| `seniority` | 0–1 (junior → senior) | More senior → more autonomy over task allocation; broader task repertoire; more likely to have delegation authority |
| `org_context` | 0–1 (independent → institutional) | Low: freelancer, small firm, self-employed — high autonomy, narrow repertoire. High: large organization, public sector — more administrative tasks, more constraint on delegation choices |

**Task repertoire logic:** profession determines which task areas are in play (and their base
probabilities); seniority broadens the repertoire within those areas; org_context adds
administrative overhead at the high end and narrows the repertoire at the low end. Values
and dispositions then determine *which specific tasks are retained vs. ceded*, not which
tasks exist. This breaks the false `values → task selection` chain of the original design.

```
profession  ──→ task areas in play
seniority   ──→ breadth within areas
org_context ──→ administrative load + delegation autonomy

values + dispositions ──→ boundary decisions on that repertoire
```

### Layer A — Dispositional core (shared by both generators)

#### Values (Schwartz, 10 dimensions, each 0–1, normalized to sum to 1)

Sampled from the Schwartz circumplex, not independent uniforms. Adjacent values are
psychologically compatible; opposite values are in tension. The circumplex projection ensures
sampled personas are psychologically plausible.

| Parameter | What it captures |
|---|---|
| `v_self_direction` | Independence of thought and action; creativity; exploration |
| `v_stimulation` | Excitement, novelty, challenge |
| `v_hedonism` | Pleasure, sensuous gratification |
| `v_achievement` | Personal success through demonstrated competence |
| `v_power` | Social status, prestige, control over people and resources |
| `v_security` | Safety, harmony, stability of society/relationships/self |
| `v_conformity` | Restraint of impulses likely to harm others or violate norms |
| `v_tradition` | Respect, commitment to customs and ideas of one's culture |
| `v_benevolence` | Preserving and enhancing the welfare of close others |
| `v_universalism` | Understanding, appreciation, tolerance, protection of all people and nature |

> **Psychological grounding:** The Schwartz theory of basic values is one of the most
> extensively validated frameworks in cross-cultural psychology, with support from 80+ countries.
> The circumplex structure and the 10 basic values are documented in Schwartz (1992, 2012); the
> refined 19-value PVQ-RR model (Schwartz et al., 2012) provides finer granularity but the
> 10-value version is chosen here for tractability. See Appendix B for full references.

Schwartz's circular structure constrains what personas are plausible. The circumplex order
(approximate): self-direction — stimulation — hedonism — achievement — power — security —
conformity — tradition — benevolence — universalism — (back to self-direction). A persona
sampled from a narrow arc of the circle is internally consistent; a persona sampled uniformly
across the circle risks combining incompatible values (e.g., universalism + power).

#### Domain-specific dispositions (7 parameters, each 0–1)

These bridge the abstract values to the concrete domain of AI delegation. They are partially
determined by the value profile but add independent variance. **Every disposition listed here
is wired into both generators** — none are decorative.

| Parameter | Definition | Text effect | Graph effect |
|---|---|---|---|
| `delegation_comfort` | Baseline willingness to give tasks to AI [B1] | Fewer anxious/defensive statements about AI | Lower retention probability |
| `competence_attachment` | How much identity is bound to skilled performance [B2] | More emphasis on skill, craft, judgment in narrative | Higher retention of tasks whose `value_relevance` includes `v_achievement` |
| `identity_attachment` | How much authorship/ownership of work defines the self [B3] | More emphasis on "my work," "my voice," ownership language | Higher retention of tasks tagged with authorship/creative control |
| `trust_in_ai` | Belief that AI outputs are reliable and correct [B4] | More positive AI language, fewer cautionary anecdotes | Reduces the effective `ai_suitability` penalty in the retention equation |
| `coupling_awareness` | How explicitly the person links tasks to personal values [B5] | More explicit value-language: "I do this because..." | Proportion of retained tasks receiving SERVES edges |
| `internal_boundary_clarity` | How clearly the person distinguishes retained from delegated tasks [B6] | Fewer hedges and contradictions in delegation statements | Lower proportion of SHARED edges; sharper retain/cede partition |
| `self_awareness` | Congruence between implicit dispositions and explicit self-report [B7] | Higher: narrative is consistent with actual boundary decisions. Lower: stated attitudes diverge from graph structure (implicit/explicit gap) | Higher: boundary decisions align with value profile. Lower: more random/deviant boundary decisions (noise injection into the retention equation) |

**Approximate dependencies on values:**

- `delegation_comfort` ↑ with `v_security`, `v_hedonism`; ↓ with `v_self_direction`, `v_achievement`
- `competence_attachment` ↑ with `v_achievement`, `v_self_direction`; ↓ with `v_tradition`
- `identity_attachment` ↑ with `v_self_direction`, `v_power`; ↓ with `v_conformity`
- `trust_in_ai` ↑ with `v_security`, `v_conformity`; ↓ with `v_self_direction`
- `coupling_awareness` ↑ with `v_self_direction`, `v_universalism` (reflective people articulate more)
- `internal_boundary_clarity` ↑ with `v_achievement`, `v_power`; ↓ with `v_benevolence` (decisiveness vs. accommodation)
- `self_awareness` ↑ with `v_self_direction`, `v_universalism`; ↓ with `v_conformity`

These are soft correlations, not deterministic functions. Each disposition retains residual
variance to allow personas that violate the typical pattern.

#### Cognitive style (2 parameters, each 0–1)

These capture *how* the person thinks, orthogonal to *what* they value. Both channels see them.

| Parameter | Definition | Text effect | Graph effect |
|---|---|---|---|
| `systematic_vs_intuitive` | 0 = intuitive/heuristic, 1 = systematic/analytic [B8] | Systematic: more structured narratives, explicit reasoning chains. Intuitive: more anecdotal, associative storytelling | Systematic: more complete coupling (fewer retained-but-uncoupled tasks). Intuitive: sparser, less justified graph |
| `abstract_vs_concrete` | 0 = concrete/example-driven, 1 = abstract/principle-driven [B9] | Abstract: more conceptual language, references to principles. Concrete: more specific examples, fewer generalizations | Abstract: SERVES edges link to higher-level values (universalism, self-direction). Concrete: SERVES edges link to concrete values (security, achievement) |

### Layer B — Channel-specific stylistics

#### Text-only parameters (5 dimensions, each 0–1)

Control *how* the person narrates, not *what* they believe. Read by the text generator only.
These are properties of the narrative register, not hidden psychological information.

| Parameter | Effect on generated text |
|---|---|
| `verbosity` [B10] | Transcript length; number of distinct points made; turn length |
| `hedging_tendency` [B11] | Frequency of qualifiers ("maybe", "I think", "sometimes", "it depends") |
| `emotional_expressiveness` [B12] | Density of affective language; degree of personal disclosure |
| `narrative_coherence` [B13] | How well-structured the story is; topic drift; digression frequency |
| `self_presentation_bias` [B14] | Social desirability filtering; how polished the self-image presented |

Key insight: `hedging_tendency` is where ambivalence lives in the text register. A high-hedging
person sounds uncertain in prose, but the shared `internal_boundary_clarity` parameter may
produce a clean, resolved boundary in the graph modality. The divergence between these two
channels on the *same* underlying disposition is where modality complementarity lives.

#### Graph-format parameters (3 dimensions, each 0–1)

These model **properties of the elicitation procedure, not properties of the person.**
A card-sort or diagram task differs from free narrative in three ways that are inherent to
the format:

1. **Forced decomposition** — the format demands discrete itemization; narrative does not.
2. **Value surfacing** — the format makes value distinctions explicit that narrative may
   leave implicit.
3. **Response noise** — any structured elicitation introduces motor errors, attention lapses,
   order effects, and mid-task mind-changing that are properties of the measurement
   instrument, not the measured psychology.

These parameters are read by the graph generator only. They are **not** latent psychological
variables — a skeptic's question "why does the person have graph-specific decomposition
ability?" has a simple answer: they don't. The *instrument* forces decomposition; the
parameter controls how much of that forced structure appears. The text generator doesn't see
these parameters because a free narrative interview has no card-sort.

| Parameter | Effect on generated graph | Type |
|---|---|---|
| `decomposition_granularity` [B15] | How many distinct tasks they identify (3–15) | Format property: forced itemization intensity |
| `value_differentiation` [B16] | How many distinct values they distinguish (1–8) | Format property: value-surfacing intensity |
| `situational_noise` | Random boundary-decision flips independent of any disposition (0–0.15) | Format property: motor error, inattention, order effects |

`situational_noise` is the probability that a task's boundary decision is flipped to a random
alternative *after* all disposition-driven computation, independently per task. It introduces
the kind of inconsistency real elicitations produce — misclicks, fatigue effects, second-guessing
mid-task — without attributing it to a psychological trait. It is bounded at 0.15 (15% of
tasks affected, on average) so the graph remains predominantly signal-driven.

### Complete persona vector

```
persona = {
    # Layer 0 — Role context (shared)
    "role": {
        profession,                    # categorical: 7 archetypes
        seniority,                     # 0–1
        org_context                    # 0–1 (independent → institutional)
    },

    # Layer A — Dispositional core (shared)
    "values": {
        v_self_direction, v_stimulation, v_hedonism, v_achievement, v_power,
        v_security, v_conformity, v_tradition, v_benevolence, v_universalism
    },                                                         # 10 dims, sum=1
    "dispositions": {
        delegation_comfort, competence_attachment, identity_attachment,
        trust_in_ai, coupling_awareness, internal_boundary_clarity,
        self_awareness
    },                                                         # 7 dims, each 0–1
    "cognitive_style": {
        systematic_vs_intuitive, abstract_vs_concrete
    },                                                         # 2 dims, each 0–1

    # Layer B — Channel-specific
    "text_style": {
        verbosity, hedging_tendency, emotional_expressiveness,
        narrative_coherence, self_presentation_bias
    },                                                         # 5 dims, each 0–1
    "graph_format": {
        decomposition_granularity, value_differentiation,
        situational_noise
    }                                                          # 3 dims, each 0–1
}
```

**Total: 30 free parameters** (profession is categorical, 7 values; 29 continuous dims
plus the profession category). Compact enough for systematic sampling; rich enough to drive
meaningfully different outputs across both modalities.

---

## 2. Shared data model: task ontology, value taxonomy, and profession mapping

Both generators draw from shared pools of tasks and values. The persona's **role context**
determines which tasks are available; the persona's **values and dispositions** determine
how those tasks are treated.

### Profession → task area mapping

| Profession | Primary areas (weight 1.0) | Secondary areas (weight 0.5) |
|---|---|---|
| `clinician` | clinical, analytic | interpersonal, learning |
| `creative` | creative, interpersonal | learning, analytic |
| `analyst` | analytic, administrative | learning, interpersonal |
| `manager` | interpersonal, administrative | analytic, learning |
| `administrator` | administrative | interpersonal |
| `technician` | analytic, clinical | administrative, learning |
| `educator` | interpersonal, learning | creative, analytic |

`seniority` multiplies all base weights by `0.7 + 0.6 × seniority` (range 0.7–1.3), broadening
or narrowing the effective repertoire. `org_context` adds `0.3 × org_context` weight to the
administrative area (institutional roles have more admin overhead).

### Task ontology (domain: AI in professional work)

A fixed pool of ~30 task types organized by functional area. Each task type has three
fixed properties:

- `ai_suitability` (0–1): how objectively amenable this task is to AI automation (derived from
  the automation literature, not from the persona)
- `value_relevance`: which Schwartz values the task typically engages
- `authorship_weight` (0–1): how much the task involves personal authorship/creative control
  (engaged by `identity_attachment`)
- `skill_signaling_weight` (0–1): how much the task signals competence (engaged by
  `competence_attachment`)

| Area | Tasks | Typical `ai_suitability` | Typical `authorship` | Typical `skill_signaling` |
|---|---|---|---|---|
| Clinical | diagnose, treat_plan, monitor_patient, document_encounter | 0.4–0.7 | 0.3–0.7 | 0.6–0.9 |
| Creative | generate_ideas, draft_content, refine_output, final_approval | 0.3–0.6 | 0.7–1.0 | 0.4–0.8 |
| Analytic | gather_data, analyze_data, interpret_results, report_findings | 0.5–0.8 | 0.2–0.5 | 0.5–0.8 |
| Administrative | schedule, route_info, track_status, file_compliance | 0.8–1.0 | 0.0–0.1 | 0.0–0.2 |
| Interpersonal | advise_colleague, mentor_junior, negotiate_stakeholder, present_to_client | 0.1–0.3 | 0.3–0.6 | 0.3–0.7 |
| Learning | read_literature, synthesize_knowledge, practice_skill, teach_others | 0.3–0.5 | 0.1–0.4 | 0.5–0.9 |

### Value taxonomy

A fixed pool of ~15 terminal values relevant to professional work, each mapped to one or more
Schwartz values. The mapping is directional: a persona high in `v_self_direction` is more likely
to select `mastery_craft` and `autonomy` as personally relevant.

Each value also carries an `abstraction_level` (0–1) used by the `abstract_vs_concrete`
cognitive style parameter:
- High abstraction (0.7–1.0): universalism-linked values like `intellectual_growth`, `helping_others`, `autonomy`
- Low abstraction (0.0–0.3): concrete values like `financial_stability`, `efficiency`, `comfort`

| Terminal value | Primary Schwartz mapping | `abstraction_level` |
|---|---|---|
| `mastery_craft` | v_achievement, v_self_direction | 0.4 |
| `intellectual_growth` | v_self_direction, v_universalism | 0.8 |
| `professional_standing` | v_achievement, v_power | 0.5 |
| `creative_authorship` | v_self_direction, v_stimulation | 0.7 |
| `helping_others` | v_benevolence | 0.7 |
| `social_harmony` | v_conformity, v_security | 0.4 |
| `financial_stability` | v_security | 0.1 |
| `autonomy` | v_self_direction | 0.8 |
| `recognition` | v_power | 0.5 |
| `belonging` | v_benevolence, v_conformity | 0.5 |
| `tradition_honoring` | v_tradition | 0.3 |
| `efficiency` | v_achievement | 0.2 |
| `certainty` | v_security | 0.2 |
| `novelty` | v_stimulation | 0.6 |
| `comfort` | v_hedonism | 0.1 |

---

## 3. The text generator

### What it produces

An interview transcript: a sequence of exchanges between an AI interviewer and a human
respondent on the topic "how AI is affecting your professional work."

```
{
  "transcript_id": "synth_0042",
  "topic": "AI in professional work",
  "turns": [
    {"speaker": "ai",  "text": "..."},
    {"speaker": "human", "text": "..."},
    ...
  ]
}
```

### Generation mechanism

The text is generated by an LLM prompted with a **prose persona description** derived from the
persona parameters. The LLM sees the shared latent (role context + values + dispositions +
cognitive style) and the text-only parameters (`text_style`), but does **not** see the
graph-format parameters (`graph_format`) or any graph output.

The persona description is a structured prose sketch:

> You are a [seniority-level] [profession] working in [org_context_description].
> 
> Your personality:
> - You strongly value independence of thought and creative expression and care less
>   about tradition and conformity.
> - You're somewhat comfortable delegating routine tasks to AI but deeply attached to
>   tasks requiring skilled judgment — your competence matters to your sense of self.
> - You tend to think in [systematic/intuitive] and [abstract/concrete] ways.
> - You have [high/moderate/low] self-awareness about your own motivations.
> 
> In conversation:
> - You tend to be moderately verbose, frequently hedge your statements, and are
>   somewhat emotionally expressive.
> - Your narrative is loosely structured — you sometimes drift between topics.
> - You present yourself cautiously.
>
> You're being interviewed about how AI is affecting your professional work.
> Respond naturally.

The topic is fixed. The output is free narrative — no structural constraint beyond the persona
description. The persona prompt describes the person in behavioral and trait terms rather than
stating parameter values directly, to avoid the LLM simply reciting the latent vector in prose.

### LLM realism risk and mitigation

Modern LLMs are very good at reconstructing latent structure from rich prompts. A persona
description that includes the full value profile may produce a transcript where the entire
psychology is recoverable from prose alone, leaving no headroom for the graph. This is the
opposite of the "latent leakage" concern — text becomes *too* informative, and the synthetic
experiment understates graph complementarity because the text channel is unrealistically
clean.

**Mitigation strategies (implementation-level, not spec-level):**

1. **Behavioral framing.** The prompt describes the persona in behavioral terms (what they do,
   how they react) rather than trait terms (what they score on). "You tend to double-check
   AI outputs before using them" rather than "your trust_in_ai is 0.3."
2. **Noise injection.** Instruct the LLM to include realistic imperfections: memory gaps ("I
   can't recall a specific example"), inconsistencies ("actually, now that I think about
   it..."), topic drift, and incomplete thoughts.
3. **Implicit limits.** Do not list all seven dispositions in the prompt. Describe the 2–3
   most salient ones, leaving the others to be inferred (or missed) by the LLM. A real
   interview does not give the listener a parameter vector.
4. **Structured annotation as a ground-truth check.** The post-hoc parse (see below) measures
   how much latent structure actually leaked into the text, enabling post-hoc calibration
   of text informativeness.

### Variation control: structured annotation

To make the text output systematically analyzable, the generator also produces a post-hoc
parse of the transcript into the shared task/value vocabulary. This is **not** the graph — it
is a coding of the text content, used for comparison:

```
{
  "text_mentions": {
    "tasks_discussed": ["draft_content", "final_approval", "schedule"],
    "values_expressed": ["creative_authorship", "efficiency", "autonomy"],
    "delegation_stance": {
      "draft_content": "shared",      # "I let it draft but I always rewrite"
      "final_approval": "retained",   # "that has to be me"
      "schedule": "ceded"             # "AI handles that, it's just logistics"
    }
  }
}
```

This annotation measures how much of the graph's information is *implicitly present* in the
text, even though the text doesn't structure it as a graph. It also serves as a calibration
check: if the text annotation recovers near-perfectly the persona's task repertoire and
boundary decisions, then the LLM prompt was too revealing and the mitigation strategies
need tightening.

### Information boundary

The text generator has access to:
- `role` (profession, seniority, org_context)
- `values` (10 dims)
- `dispositions` (7 dims)
- `cognitive_style` (2 dims)
- `text_style` (5 dims)
- Topic (fixed)
- Task and value pools (shared vocabulary)

The text generator does **NOT** have access to:
- `graph_format` parameters
- The generated graph
- Any structured decomposition of the topic into tasks/values beyond the shared pools

---

## 4. The graph generator

### What it produces

A delegation-boundary graph in a clean, minimal schema. The generator is **fully deterministic**
given its parameters — no LLM is involved. This is essential for the DPI-respecting claim: if
an LLM generated the graph, the same circularity concern that affected the real-data project
(the extractor agreeing with itself) would apply.

```
{
  "graph_id": "synth_0042_graph",
  "source": "independent_elicitation",
  "parameters_used": ["role", "values", "dispositions", "cognitive_style", "graph_format"],
  "parameters_excluded": ["text_style"],
  "nodes": {
    "self": {"type": "Self"},
    "ai":   {"type": "AI"},
    "tasks": [
      {"id": "t1", "label": "draft_content"},
      {"id": "t2", "label": "final_approval"},
      {"id": "t3", "label": "schedule"}
    ],
    "values": [
      {"id": "v1", "label": "creative_authorship"},
      {"id": "v2", "label": "efficiency"}
    ]
  },
  "edges": {
    "boundary": [
      {"source": "t1", "target": "ai",   "relation": "SHARED_WITH"},
      {"source": "t2", "target": "self", "relation": "RETAINED_BY"},
      {"source": "t3", "target": "ai",   "relation": "CEDED_TO"}
    ],
    "coupling": [
      {"source": "t2", "target": "v1", "relation": "SERVES"}
    ]
  },
  "deterministic_readouts": {
    "delegation_breadth": 0.50,
    "breadth_class": "med",
    "alignment_phi": 1.0,
    "retained_coupled_ratio": 1.0,
    "ceded_coupled_ratio": 0.0
  }
}
```

### Generation algorithm

The graph generator runs deterministically from its parameters. No randomness beyond the
parameter vector itself (fixed seed per persona).

#### Step 1: Sample the task repertoire (driven by role, not values)

```
n_tasks = 3 + floor(decomposition_granularity × 12)    # range: 3–15

# Role-weighted task pool:
For each task t in task_pool:
    area = area_of(t)
    base_weight = profession_area_weight(profession, area)  # from profession→area mapping
    seniority_multiplier = 0.7 + 0.6 × seniority
    admin_overhead = 0.3 × org_context if area == "administrative" else 0.0
    
    p_select(t) = max(0.05, (base_weight + admin_overhead) × seniority_multiplier)

tasks = weighted_sample(task_pool, n_tasks, weights=p_select, without_replacement)
```

The task repertoire is determined by profession, seniority, and organizational context. Values
do not influence which tasks appear — they influence what happens to those tasks in Step 2.

#### Step 2: For each task, sample the boundary decision

The retention logit now uses **all seven dispositions** plus cognitive style and self-awareness
modulation. Every disposition is wired.

```
For each task t in tasks:

    # --- Value pull (unchanged) ---
    value_pull = sum(
        relevance_weight(t, v) × persona.values[v]
        for v in schwartz_values
    )
    
    # --- Disposition-driven modifiers ---
    
    # competence_attachment: increases retention of skill-signaling tasks
    comp_mod = competence_attachment × skill_signaling_weight(t) × 1.5
    
    # identity_attachment: increases retention of authorship tasks
    ident_mod = identity_attachment × authorship_weight(t) × 1.5
    
    # trust_in_ai: reduces the effective AI-suitability penalty
    # (a trusting person treats AI-fit tasks as less threatening)
    effective_ai_suit = ai_suitability(t) × (1.0 − 0.5 × trust_in_ai)
    
    # coupling_awareness + systematic_vs_intuitive: more aware/systematic people
    # are more deliberate about retention — they retain tasks they can justify
    awareness_mod = coupling_awareness × 0.3 + systematic_vs_intuitive × 0.2
    
    # self_awareness: modulates noise in the decision
    # Low self-awareness → boundary decisions deviate from what values would predict
    noise_scale = (1.0 − self_awareness) × 0.5
    ε = rng.normal(0, noise_scale)  # mean-zero noise, scaled by unawareness
    
    # --- Retention logit ---
    logit_retain = (
        −2.0 × effective_ai_suit         # AI-fit pushes toward ceding
        − 1.5 × delegation_comfort        # comfort pushes toward ceding
        + 2.0 × value_pull                # personal relevance pulls toward retaining
        + comp_mod                        # competence attachment pulls toward retaining
        + ident_mod                       # identity attachment pulls toward retaining
        + awareness_mod                   # awareness/systematicity → more deliberate retention
        + 0.5                             # mild baseline toward retain
        + ε                               # self-awareness noise
    )
    
    p_retain = sigmoid(logit_retain)
    
    # --- Cede-vs-shared split ---
    # internal_boundary_clarity: shared parameter, both channels see it
    # High clarity → fewer SHARED edges (sharp partition)
    p_cede_given_not_retain = internal_boundary_clarity
    
    decision = categorical_sample({
        "retain": p_retain,
        "cede":   (1 − p_retain) × p_cede_given_not_retain,
        "shared": (1 − p_retain) × (1 − p_cede_given_not_retain)
    })
    
    # --- Situational noise (format property, not psychology) ---
    # Independently per task, a small chance of random override — models
    # motor errors, attention lapses, order effects inherent to any
    # structured elicitation instrument.
    if rng.uniform(0, 1) < situational_noise:
        decision = rng.choice([RETAIN, CEDE, SHARE])  # equiprobable override
```

**Every disposition has a specific, psychologically-motivated effect:**

| Disposition | Mechanism |
|---|---|
| `delegation_comfort` | Baseline shift in the retention intercept |
| `competence_attachment` | Increases retention of tasks high in `skill_signaling_weight` |
| `identity_attachment` | Increases retention of tasks high in `authorship_weight` |
| `trust_in_ai` | Shrinks the `ai_suitability` penalty (trusting people see AI-fit tasks as less threatening) |
| `coupling_awareness` | Modest positive shift — aware people are more deliberate about retention |
| `internal_boundary_clarity` | Controls the cede-vs-shared split (replaces the old `boundary_rigidity` graph-only parameter) |
| `self_awareness` | Injects noise into the retention logit — low awareness → decisions deviate from value/disposition predictions |

**Behavioral realism:** The `competence_attachment` and `identity_attachment` mechanisms create
differential retention: a person high in `competence_attachment` particularly retains tasks
that signal skill (diagnose, analyze_data, mentor_junior), while a person high in
`identity_attachment` particularly retains tasks involving authorship (final_approval,
generate_ideas). Two personas with the same `delegation_comfort` and same task repertoire
produce different boundary patterns because *different kinds of tasks matter to them*.

#### Step 3: Select personally relevant values

```
n_values = 1 + floor(value_differentiation × 7)    # range: 1–8

# Each terminal value in the pool maps to Schwartz values + has abstraction_level.
# Weight selection by persona's Schwartz profile, modulated by cognitive style.
For each value v in value_pool:
    # Base weight from Schwartz mapping
    schwartz_weight = max(
        persona.values[sv] 
        for sv in v.schwartz_mapping
    )
    
    # Cognitive style modulation:
    # abstract thinkers favor high-abstraction values
    # concrete thinkers favor low-abstraction values
    abstraction_match = 1.0 − abs(v.abstraction_level − abstract_vs_concrete)
    style_bonus = 0.2 × abstraction_match
    
    weight = schwartz_weight + style_bonus

selected_values = weighted_sample(value_pool, n_values, weights=weight, without_replacement)
```

#### Step 4: For each retained task, sample SERVES edges

```
For each task t where decision == "retain":
    # coupling_awareness is a SHARED disposition — both channels see it.
    # In text: more explicit value-language. In graph: more SERVES edges.
    if random() < coupling_awareness:
        for each value v in selected_values:
            coupling_strength = (
                relevance_weight(t, v) × 
                persona.values[mapped_schwartz(v)]
            )
            
            # Systematic thinkers produce more complete coupling graphs
            systematic_bonus = systematic_vs_intuitive × 0.15
            
            p_edge = clip(coupling_strength × 0.5 + systematic_bonus, 0, 1)
            
            if random() < p_edge:
                add_edge(t → v, "SERVES")

    # Retained tasks with no SERVES edge are "retained but not justified"
    # — intentional structure. Low coupling_awareness produces this.
```

**Design property:** `coupling_awareness` is a shared parameter — both generators see it.
In text, it drives explicit value-language ("I do this because it's who I am"). In graph, it
drives SERVES edge density. Graph advantage, if observed, comes from the *representational
efficiency* of explicit edges over implicit narrative, not from hidden information.

#### Step 5: Compute deterministic readouts

The readouts are functions of the generated structure — they are what downstream classifiers
would try to predict or recover from the graph.

```
# Delegation breadth: how much is given to AI
n_tasks = count(tasks)
n_ceded = count(tasks where decision == "cede")
n_shared = count(tasks where decision == "shared")
n_retained = count(tasks where decision == "retain")

delegation_breadth = (n_ceded + 0.5 × n_shared) / n_tasks
breadth_class = (
    "low"  if delegation_breadth < 0.33 else
    "med"  if delegation_breadth < 0.66 else
    "high"
)

# Boundary–coupling alignment: are retained tasks the coupled ones?
# 2×2 table: {retained, not_retained} × {has_SERVES, no_SERVES}
# Positive φ means retained tasks carry value coupling
alignment_phi = phi_coefficient(
    row = [is_retained(t) for t in tasks],
    col = [has_SERVES_edge(t) for t in tasks]
)

# Supporting metrics
retained_coupled_ratio = (
    count(tasks where retained AND has_SERVES) / max(n_retained, 1)
)
ceded_coupled_ratio = (
    count(tasks where NOT retained AND has_SERVES) / max(n_ceded + n_shared, 1)
)
```

### Information boundary

The graph generator has access to:
- `role` (profession, seniority, org_context)
- `values` (10 dims)
- `dispositions` (7 dims)
- `cognitive_style` (2 dims)
- `graph_format` (3 dims — properties of the elicitation instrument, not the person)
- Task and value pools (shared vocabulary)

The graph generator does **NOT** have access to:
- `text_style` parameters
- The generated transcript
- Any LLM

**On `graph_format` exclusivity:** The `graph_format` parameters are visible only to the
graph generator because they model properties of the elicitation instrument, not the
person. A free narrative interview has no forced decomposition, no value-surfacing prompts,
and different noise characteristics than a card-sort. Including these in the text generator
would be asking "why doesn't the interview transcript reflect the card-sort's format?" —
the answer is that the formats are different by design. This is the one asymmetry that the
DPI-respecting architecture permits: the *elicitation method itself* can add structure that
a different method lacks, and that structure is part of the modality difference being tested.

---

## 5. Independence architecture

```
                    ┌─────────────────────────────────────┐
                    │         PERSONA PARAMETERS           │
                    │                                      │
                    │  role ──────────── shared ──────────┐│
                    │  values (10) ───── shared ─────────┐││
                    │  dispositions (7) ─ shared ───────┐│││
                    │  cognitive_style (2) ─ shared ───┐││││
                    │  text_style (5) ─── text only ─┐ │││││
                    │  graph_format (2) ─ graph only  │ │││││
                    └─────────────────────────────────┘ ││││││
                                                        ││││││
              ┌─────────────────────────────────────────┘│││││
              │        ┌─────────────────────────────────┘││││
              │        │        ┌─────────────────────────┘│││
              │        │        │        ┌─────────────────┘││
              │        │        │        │        ┌─────────┘│
              │        │        │        │        │          │
              ▼        │        │        │        │          │
    ┌──────────────┐   │        │        │        │          │
    │ TEXT GENERATOR│   │        │        │        │          │
    │ (LLM, sees:  │   │        │        │        │          │
    │  role, values,│   │        │        │        │          │
    │  dispositions,│   │        │        │        │          │
    │  cognitive,   │   │        │        │        │          │
    │  text_style,  │   │        │        │        │          │
    │  topic)       │   │        │        │        │          │
    └──────┬───────┘   │        │        │        │          │
           │           │        │        │        │          │
           ▼           │        │        │        │          │
    ┌──────────┐       │        │        │        │          │
    │ TRANSCRIPT│      │        │        │        │          │
    │ (narrative│      │        │        │        │          │
    │  prose)   │      │        │        │        │          │
    └──────────┘       │        │        │        │          │
                       │        │        │        │          │
              ┌────────┴────────┴────────┴────────┴──────────┴──┐
              │             GRAPH GENERATOR                      │
              │  (deterministic, sees:                           │
              │   role, values, dispositions, cognitive_style,   │
              │   graph_format, task_pool, value_pool)           │
              └──────────────────────┬──────────────────────────┘
                                     │
                                     ▼
                              ┌──────────────┐
                              │    GRAPH      │
                              │  (Task/Value  │
                              │   nodes,      │
                              │   boundary +  │
                              │   coupling    │
                              │   edges)      │
                              └──────────────┘
```

**The independence guarantee:** The graph generator never sees the transcript. The text
generator never sees the graph or the `graph_format` parameters. They share only `role`,
`values`, `dispositions`, `cognitive_style`, and `topic`. All parameters that shape graph
output are either shared (so text has the same information, expressed differently) or are
properties of the elicitation format (`graph_format`), not hidden psychological variables.

Any shared predictive content comes from the shared persona latent, not from one channel
being derived from the other. DPI does not constrain `I(Graph; Label)` relative to
`I(Text; Label)` because Graph is not a function of Text. Complementarity, if observed,
emerges from representational differences between the channels, not from information
asymmetry in the parameter vector.

---

## 6. Measurable experiments

The synthetic experiment mirrors Panel B of `s7_synthetic/synthetic_demo.py` but with
psychologically-real structure instead of abstract noise.

### Experiment 1: Complementarity baseline

For N sampled personas:
1. Generate text (transcript) + graph from each persona
2. Embed text via SBERT or similar frozen encoder
3. Extract graph features: deterministic readouts, graph statistics, or learned GNN embedding
4. Train classifiers to predict a target (e.g., `breadth_class`, or directly a persona
   parameter like `delegation_comfort`)
5. Compare: text-only, graph-only, text+graph

**If graph adds signal over text**, it is because the structural register captured something
the narrative register expressed less efficiently — and the ground-truth persona parameters
reveal exactly what.

**Control check:** Since all psychologically substantive parameters are shared, any graph
advantage must come from representational efficiency (the graph makes explicit what text
leaves implicit) or elicitation-format effects (the forced decomposition surfaces information
free narrative doesn't), not from hidden variables.

### Experiment 2: The hedging–clarity gap

`hedging_tendency` (text-only) and `internal_boundary_clarity` (shared) interact. A persona
with `hedging=0.8` and `internal_boundary_clarity=0.8` produces text that *sounds* uncertain
("well, it depends...") but a graph with a sharp, clean partition. The graph modality
surfaces *resolved disposition* that the narrative register obscures.

**Testable prediction:** For personas with high `hedging_tendency` and high
`internal_boundary_clarity`, graph adds more predictive value over text for targets like
`breadth_class`. For low-hedging, low-clarity personas, text outperforms graph. This is a
falsifiable version of the "different register" hypothesis: the gap is not about hidden
information but about *expression asymmetry*.

### Experiment 3: The coupling awareness test

`coupling_awareness` is a shared disposition. In text, it produces explicit value-language
("I do this because..."). In graph, it produces SERVES edges. The same psychological
variable, two representational forms.

**Testable prediction:** Graph advantage over text on predicting `coupling_awareness`
itself (or the alignment φ it produces) is *not* guaranteed — it depends on whether
explicit edges are more learnable than explicit prose statements. The experiment tests
representational efficiency, not information asymmetry. A null result ("text recovers
coupling awareness as well as graph does") is a valid and informative finding.

### Experiment 4: Value-profile recovery

Both generators see the same `values` vector. A classifier trained on text-only features
should recover the persona's value profile to some degree. A classifier trained on graph-only
features should also recover it. The difference in recovery accuracy per value dimension
reveals *which* values are better expressed in which register.

**Testable prediction:** `v_achievement` and `v_power` (hierarchy/status values) are
expressed through task boundary decisions → graph recovers them better. `v_benevolence` and
`v_universalism` (relational/universal values) are expressed through narrative → text recovers
them better. `v_self_direction` should be expressed in both channels (via autonomy-relevant
tasks + narrative emphasis on independence).

### Experiment 5: Self-awareness and channel divergence

`self_awareness` is a shared disposition that injects noise into the graph generator's
boundary decisions (low awareness → more random retention). It also affects text: low
awareness → narrative self-description diverges from actual dispositions.

**Testable prediction:** The correlation between text-derived persona predictions and
graph-derived persona predictions decreases as `self_awareness` decreases. For high
self-awareness personas, the two channels converge (both reflect the same underlying
psychology). For low self-awareness personas, they diverge — and the pattern of divergence
is psychologically interpretable (the graph shows implicit attitudes; the text shows
explicit self-report). This models the well-established implicit/explicit attitude
distinction.

---

## 7. Implementation sketch

```
s8_parameterized/
├── persona_schema.py        # Parameter definitions, Schwartz circumplex sampling, profession taxonomy
├── task_ontology.py         # Task pool, value pool, profession→area mappings, relevance weights
├── text_generator.py        # LLM-based text generation from persona description
├── graph_generator.py       # Deterministic graph generation algorithm
├── generate_dataset.py      # Sample N personas, generate both outputs for each
├── encode_and_classify.py   # Embed, graph-featurize, classify, measure complementarity
└── README.md                # "This is a controlled fiction..." disclaimer
```

**Dependencies:** The text generator requires an LLM (controlled via persona description
prompt, with behavioral framing and noise injection). The graph generator is pure Python —
deterministic functions, no ML. The Schwartz circumplex sampler uses standard angular
coordinates in 2D space, projected to the 10 value dimensions via the established circumplex
loadings.

**Persona sampling:** Rather than sampling 10 independent uniforms (which would produce
psychologically implausible value combinations), the sampler draws a point in the Schwartz
2D circular space — an angle θ (which value region is emphasized) and a radius r (how strongly
differentiated the values are). The angle and radius are projected to the 10 value scores
via the circumplex structure documented in Schwartz (1992, 2012). Profession is sampled
uniformly from the 7 archetypes. Seniority and org_context are sampled from Beta(2,2)
(centered at 0.5). Dispositions, cognitive style, text_style, and graph_format parameters
are sampled independently from Beta distributions with soft correlations to the value profile
as specified in §1.

---

## 8. Limitations and honest framing

1. **This is a controlled fiction.** Generated transcripts do not resemble real interviews;
   generated graphs do not resemble real human elicitations. The model illustrates logic,
   not reality.

2. **The text generator uses an LLM.** This introduces a subtle circularity concern: the LLM
   may encode its own implicit model of how values map to narrative, which could create
   artificial correlation between text and graph channels. The structured annotation and
   the graph generator's determinism partially control for this, but do not eliminate it. A
   human study with independent elicitation would not have this concern. See §3 "LLM realism
   risk and mitigation" for implementation-level strategies to reduce this risk.

3. **The task and value pools are fixed and domain-specific.** They cover AI adoption in
   professional work only. Extending to other domains requires new pools but not a new
   architecture.

4. **The persona model is a simplification.** Real people have within-person variability across
   contexts, developmental trajectories, and non-linear interactions between values that this
   ~30-parameter vector does not capture. Known gaps: (a) the 7-archetype profession taxonomy
   is coarse — "analyst" spans quantitative traders to policy researchers — and a hierarchical
   domain→occupation→specialization structure would be more realistic; (b) values remain the
   dominant driver of retention decisions, while real delegation also responds to situational
   constraints (incentives, workload, deadlines, organizational norms, risk exposure) that
   the model captures only indirectly through `org_context`; (c) the `situational_noise`
   parameter adds random inconsistency but does not model *systematic* contextual effects
   (e.g., the same person making different delegation choices under time pressure vs. leisure).
   These are acknowledged simplifications; the model is sufficient for demonstrating the
   *logic* of multi-modality generation, not for simulating actual individuals.

5. **The design prevents latent leakage but does not guarantee complementarity.** All
   substantive psychological parameters are shared between generators. Graph advantage, if
   observed, comes from representational efficiency or elicitation-format effects, not from
   hidden information. A null result — text recovers everything, graph adds nothing — is
   a valid and expected possible outcome. The model parameterizes *when* graph should help
   (high hedging + high clarity, high coupling awareness, low self-awareness divergence)
   rather than *assuming* it will.

6. **The real test remains a human study.** This model says: *if* you ran a study with
   independent elicitation (e.g., narrative interview + card-sort boundary task), *here is
   the structure of what you would find — and here is the space of possible findings,
   parameterized by the psychological mechanisms at work.* The model provides falsifiable
   predictions; the human study provides the evidence.

---

## Appendix A: Psychological grounding references

Each parameter marked with a `[Bn]` tag in §1 maps to the literature below. The grounding
is not uniform: some dimensions correspond directly to well-validated constructs with
established measurement instruments; others are adaptions or composites of related constructs.

### B1 — Delegation comfort

Grounded in the **technology acceptance and trust-in-automation** literature. The construct draws
on Davis's Technology Acceptance Model (TAM; Davis, 1989) — perceived ease of use and perceived
usefulness as predictors of adoption intent — and on Parasuraman & Riley's (1997) framework of
automation use, misuse, disuse, and abuse. Lee & See (2004) provide the process model of
trust in automation linking dispositional trust to reliance behaviour. Hoff & Bashir (2015)
meta-analyze factors influencing trust in automation, including individual differences.

- Davis, F. D. (1989). Perceived usefulness, perceived ease of use, and user acceptance of information technology. *MIS Quarterly*, 13(3), 319–340.
- Parasuraman, R., & Riley, V. (1997). Humans and automation: Use, misuse, disuse, abuse. *Human Factors*, 39(2), 230–253.
- Lee, J. D., & See, K. A. (2004). Trust in automation: Designing for appropriate reliance. *Human Factors*, 46(1), 50–80.
- Hoff, K. A., & Bashir, M. (2015). Trust in automation: Integrating empirical evidence on factors that influence trust. *Human Factors*, 57(3), 407–434.

### B2 — Competence attachment

Grounded in **Self-Determination Theory** (Deci & Ryan, 1985, 2000), where competence is one
of three basic psychological needs. The attachment dimension — competence as identity-constitutive
rather than merely satisfying — draws on the **implicit theories of ability** literature
(Dweck, 2006; Dweck & Leggett, 1988): entity theorists (fixed mindset) treat competence as
a stable attribute of the self, making its demonstration identity-relevant. Bandura's
**self-efficacy** theory (1977, 1997) provides the mechanism: people high in competence
attachment experience self-efficacy threat when tasks they excel at are delegated, motivating
retention.

- Deci, E. L., & Ryan, R. M. (1985). *Intrinsic motivation and self-determination in human behavior.* Plenum Press.
- Deci, E. L., & Ryan, R. M. (2000). The "what" and "why" of goal pursuits: Human needs and the self-determination of behavior. *Psychological Inquiry*, 11(4), 227–268.
- Dweck, C. S. (2006). *Mindset: The new psychology of success.* Random House.
- Dweck, C. S., & Leggett, E. L. (1988). A social-cognitive approach to motivation and personality. *Psychological Review*, 95(2), 256–273.
- Bandura, A. (1977). Self-efficacy: Toward a unifying theory of behavioral change. *Psychological Review*, 84(2), 191–215.
- Bandura, A. (1997). *Self-efficacy: The exercise of control.* W. H. Freeman.

### B3 — Identity attachment

Grounded in **social identity theory** (Tajfel & Turner, 1979) and its extension to
organizational/work identity (Ashforth & Mael, 1989). The construct captures the degree to
which work tasks are incorporated into the self-concept — what Ashforth et al. (2008) call
*identification*. Wrzesniewski et al.'s (1997) distinction between job, career, and **calling**
orientations is directly relevant: people with a calling orientation experience their work
as inseparable from who they are, and should show the strongest identity-driven task retention.
The construct also draws on Schwartz's self-direction value as its dispositional anchor.

- Tajfel, H., & Turner, J. C. (1979). An integrative theory of intergroup conflict. In W. G. Austin & S. Worchel (Eds.), *The social psychology of intergroup relations* (pp. 33–47). Brooks/Cole.
- Ashforth, B. E., & Mael, F. (1989). Social identity theory and the organization. *Academy of Management Review*, 14(1), 20–39.
- Ashforth, B. E., Harrison, S. H., & Corley, K. G. (2008). Identification in organizations: An examination of four fundamental questions. *Journal of Management*, 34(3), 325–374.
- Wrzesniewski, A., McCauley, C., Rozin, P., & Schwartz, B. (1997). Jobs, careers, and callings: People's relations to their work. *Journal of Research in Personality*, 31(1), 21–33.

### B4 — Trust in AI

Grounded in **interpersonal trust theory** adapted to automation. Mayer et al.'s (1995)
integrative model defines trust as willingness to be vulnerable to another party based on
their perceived ability, benevolence, and integrity — directly mappable to AI (ability = model
capability; benevolence = alignment; integrity = reliability). Lee & See (2004) and Hoff
& Bashir (2015) extend this to automation. Hancock et al.'s (2011) meta-analysis of factors
influencing trust in human–robot interaction provides the most comprehensive empirical
mapping. McKnight et al. (2011) specifically address trust in technology contexts including
recommendation agents.

- Mayer, R. C., Davis, J. H., & Schoorman, F. D. (1995). An integrative model of organizational trust. *Academy of Management Review*, 20(3), 709–734.
- Hancock, P. A., Billings, D. R., Schaefer, K. E., Chen, J. Y. C., de Visser, E. J., & Parasuraman, R. (2011). A meta-analysis of factors affecting trust in human-robot interaction. *Human Factors*, 53(5), 517–527.
- McKnight, D. H., Carter, M., Thatcher, J. B., & Clay, P. F. (2011). Trust in a specific technology: An investigation of its components and measures. *ACM Transactions on Management Information Systems*, 2(2), 1–25.

### B5 — Coupling awareness

Grounded in the **self-reflection and insight** literature. Grant et al.'s (2002) Self-Reflection
and Insight Scale distinguishes between *reflection* (willingness to examine one's thoughts)
and *insight* (clarity about those thoughts). Coupling awareness corresponds to the insight
component — the degree to which a person can articulate why they do what they do in terms of
personal values. This maps to **emotional intelligence** (Salovey & Mayer, 1990; Goleman,
1995), specifically the self-awareness branch. Within the Schwartz framework, coupling
autonomy correlates with self-direction (independent thought) and universalism (reflective
breadth). In the means-end chain tradition (Gutman, 1982), coupling awareness is the capacity
for value-laddering.

- Grant, A. M., Franklin, J., & Langford, P. (2002). The Self-Reflection and Insight Scale: A new measure of private self-consciousness. *Social Behavior and Personality*, 30(8), 821–835.
- Salovey, P., & Mayer, J. D. (1990). Emotional intelligence. *Imagination, Cognition and Personality*, 9(3), 185–211.
- Gutman, J. (1982). A means-end chain model based on consumer categorization processes. *Journal of Marketing*, 46(2), 60–72.

### B6 — Internal boundary clarity

Grounded in the **personal need for structure** (Neuberg & Newsom, 1993; Thompson et al., 2001)
and **preference for consistency** (Cialdini et al., 1995) literatures. Personal Need for
Structure (PNS) captures individual differences in the desire for clear, unambiguous structure.
Preference for Consistency (PFC) captures the motivation to maintain congruence across
attitudes and behaviours. Both predict sharper, more decisive classifications. The construct
also relates to **decisiveness** as a decision-making style (Germeijs & De Boeck, 2002;
Leykin & DeRubeis, 2010 — Decisional Balance Scale) and to the **need for closure** (Webster
& Kruglanski, 1994), which captures discomfort with ambiguity.

- Neuberg, S. L., & Newsom, J. T. (1993). Personal need for structure: Individual differences in the desire for simple structure. *Journal of Personality and Social Psychology*, 65(1), 113–131.
- Cialdini, R. B., Trost, M. R., & Newsom, J. T. (1995). Preference for consistency: The development of a valid measure and the discovery of surprising behavioral implications. *Journal of Personality and Social Psychology*, 69(2), 318–328.
- Webster, D. M., & Kruglanski, A. W. (1994). Individual differences in need for cognitive closure. *Journal of Personality and Social Psychology*, 67(6), 1049–1062.
- Germeijs, V., & De Boeck, P. (2002). A measurement scale for indecisiveness and its relationship to career indecision and other types of indecision. *European Journal of Psychological Assessment*, 18(2), 113–122.

### B7 — Self-awareness

Grounded in **objective self-awareness** theory (Duval & Wicklund, 1972) and the **self-consciousness
scale** (Fenigstein et al., 1975) which distinguishes private self-consciousness (attention
to internal states) from public self-consciousness (awareness of self as a social object).
The construct also draws on the **implicit/explicit attitude** distinction (Greenwald & Banaji,
1995; Greenwald et al., 1998) — our design's most important psychological mechanism. Greenwald
et al. (2009) provide the meta-analytic evidence that implicit and explicit measures converge
at varying degrees depending on the domain. **Mindfulness** (Brown & Ryan, 2003) provides the
attention-to-present-experience component.

- Duval, S., & Wicklund, R. A. (1972). *A theory of objective self awareness.* Academic Press.
- Fenigstein, A., Scheier, M. F., & Buss, A. H. (1975). Public and private self-consciousness: Assessment and theory. *Journal of Consulting and Clinical Psychology*, 43(4), 522–527.
- Greenwald, A. G., & Banaji, M. R. (1995). Implicit social cognition: Attitudes, self-esteem, and stereotypes. *Psychological Review*, 102(1), 4–27.
- Greenwald, A. G., McGhee, D. E., & Schwartz, J. L. K. (1998). Measuring individual differences in implicit cognition: The Implicit Association Test. *Journal of Personality and Social Psychology*, 74(6), 1464–1480.
- Greenwald, A. G., Poehlman, T. A., Uhlmann, E. L., & Banaji, M. R. (2009). Understanding and using the Implicit Association Test: III. Meta-analysis of predictive validity. *Journal of Personality and Social Psychology*, 97(1), 17–41.
- Brown, K. W., & Ryan, R. M. (2003). The benefits of being present: Mindfulness and its role in psychological well-being. *Journal of Personality and Social Psychology*, 84(4), 822–848.

### B8 — Systematic vs. intuitive cognitive style

Grounded in **dual-process theories** of cognition. Kahneman's (2011) System 1 (fast, intuitive,
automatic) / System 2 (slow, deliberate, analytical) distinction is the most prominent
formulation. The Cognitive-Experiential Self-Theory (Epstein et al., 1996) provides the most
direct operationalization via the Rational-Experiential Inventory (Pacini & Epstein, 1999),
which measures Need for Cognition (rational/systematic) and Faith in Intuition (experiential/intuitive)
as independent dimensions. Frederick's (2005) Cognitive Reflection Test provides a
performance-based measure.

- Kahneman, D. (2011). *Thinking, fast and slow.* Farrar, Straus and Giroux.
- Epstein, S., Pacini, R., Denes-Raj, V., & Heier, H. (1996). Individual differences in intuitive-experiential and analytical-rational thinking styles. *Journal of Personality and Social Psychology*, 71(2), 390–405.
- Pacini, R., & Epstein, S. (1999). The relation of rational and experiential information processing styles to personality, basic beliefs, and the ratio-bias phenomenon. *Journal of Personality and Social Psychology*, 76(6), 972–987.
- Frederick, S. (2005). Cognitive reflection and decision making. *Journal of Economic Perspectives*, 19(4), 25–42.

### B9 — Abstract vs. concrete cognitive style

Grounded in **construal level theory** (Trope & Liberman, 2010) and **action identification
theory** (Vallacher & Wegner, 1989). Construal level theory shows that people vary in the
abstractness of their mental representations, with higher-level construals capturing superordinate,
decontextualized meaning and lower-level construals capturing subordinate, contextualized
details. The Behavior Identification Form (Vallacher & Wegner, 1989) directly measures this
as an individual-difference dimension. Liberman & Trope (1998) and Fujita et al. (2006)
establish the connection between abstract construal and value-consistent behaviour — the
mechanism by which abstract thinkers link tasks to higher-order values in our model.

- Trope, Y., & Liberman, N. (2010). Construal-level theory of psychological distance. *Psychological Review*, 117(2), 440–463.
- Vallacher, R. R., & Wegner, D. M. (1989). Levels of personal agency: Individual variation in action identification. *Journal of Personality and Social Psychology*, 57(4), 660–671.
- Liberman, N., & Trope, Y. (1998). The role of feasibility and desirability considerations in near and distant future decisions: A test of temporal construal theory. *Journal of Personality and Social Psychology*, 75(1), 5–18.

### B10 — Verbosity

Grounded in **personality–language** research. Extraversion robustly predicts talkativeness
and word count (Mehl et al., 2006; Pennebaker & King, 1999). The Big Five taxonomy (Costa
& McCrae, 1992) provides the trait framework. Pennebaker's LIWC program (Pennebaker et al.,
2015) establishes word count as a psychologically meaningful variable.

- Mehl, M. R., Gosling, S. D., & Pennebaker, J. W. (2006). Personality in its natural habitat: Manifestations and implicit folk theories of personality in daily life. *Journal of Personality and Social Psychology*, 90(5), 862–877.
- Pennebaker, J. W., & King, L. A. (1999). Linguistic styles: Language use as an individual difference. *Journal of Personality and Social Psychology*, 77(6), 1296–1312.
- Costa, P. T., & McCrae, R. R. (1992). *Revised NEO Personality Inventory (NEO-PI-R) and NEO Five-Factor Inventory (NEO-FFI) professional manual.* Psychological Assessment Resources.

### B11 — Hedging tendency

Grounded in **politeness and uncertainty-marking** in discourse. Brown & Levinson's (1987)
politeness theory identifies hedging as a face-saving strategy. The construct maps to
**neuroticism** (anxiety-driven hedging; Pennebaker & King, 1999) and **need for closure**
(Webster & Kruglanski, 1994), which predicts reluctance to commit to definitive statements.
In the clinical domain, hedging is a marker of **ambivalence** in motivational interviewing
(Miller & Rollnick, 2013) — sustain talk vs. change talk with qualifiers. This is the
construct the real-data Phase 6 experiment attempted to target.

- Brown, P., & Levinson, S. C. (1987). *Politeness: Some universals in language usage.* Cambridge University Press.
- Miller, W. R., & Rollnick, S. (2013). *Motivational interviewing: Helping people change* (3rd ed.). Guilford Press.

### B12 — Emotional expressiveness

Grounded in the **emotional expressivity** literature. Kring et al. (1994) developed the
Emotional Expressivity Scale (EES), capturing individual differences in the outward display
of emotion. Gross & John's (1995, 1997) work on emotion regulation distinguishes expressive
suppression from cognitive reappraisal, establishing expressiveness as a stable disposition.
Extraversion predicts positive emotional expression; neuroticism predicts negative emotional
expression (Costa & McCrae, 1992).

- Kring, A. M., Smith, D. A., & Neale, J. M. (1994). Individual differences in dispositional expressiveness: Development and validation of the Emotional Expressivity Scale. *Journal of Personality and Social Psychology*, 66(5), 934–949.
- Gross, J. J., & John, O. P. (1997). Revealing feelings: Facets of emotional expressivity in self-reports, peer ratings, and behavior. *Journal of Personality and Social Psychology*, 72(2), 435–448.

### B13 — Narrative coherence

Grounded in **narrative psychology**. McAdams (2001, 2008) establishes life story coherence
as a central construct in personality: people differ in how well they construct causally and
thematically coherent narratives of their experience. Baerger & McAdams (1999) found that
life story coherence predicts psychological well-being. The construct maps to the episodic
memory and meaning-making literatures (Singer, 2004).

- McAdams, D. P. (2001). The psychology of life stories. *Review of General Psychology*, 5(2), 100–122.
- McAdams, D. P. (2008). Personal narratives and the life story. In O. P. John, R. W. Robins, & L. A. Pervin (Eds.), *Handbook of personality: Theory and research* (3rd ed., pp. 242–262). Guilford Press.
- Baerger, D. R., & McAdams, D. P. (1999). Life story coherence and its relation to psychological well-being. *Narrative Inquiry*, 9(1), 69–96.

### B14 — Self-presentation bias

Grounded in **impression management** (Leary & Kowalski, 1990) and **social desirability**
(Crowne & Marlowe, 1960). Snyder's (1974) **self-monitoring** construct captures the degree
to which people regulate their self-presentation to match situational demands — high
self-monitors produce more polished, context-appropriate narratives. Paulhus's (1984)
distinction between self-deception (honestly held but inaccurate self-views) and impression
management (deliberate tailoring) maps onto the self-awareness/self-presentation distinction
in our model.

- Leary, M. R., & Kowalski, R. M. (1990). Impression management: A literature review and two-component model. *Psychological Bulletin*, 107(1), 34–47.
- Crowne, D. P., & Marlowe, D. (1960). A new scale of social desirability independent of psychopathology. *Journal of Consulting Psychology*, 24(4), 349–354.
- Snyder, M. (1974). Self-monitoring of expressive behavior. *Journal of Personality and Social Psychology*, 30(4), 526–537.
- Paulhus, D. L. (1984). Two-component models of socially desirable responding. *Journal of Personality and Social Psychology*, 46(3), 598–609.

### B15 — Decomposition granularity

Grounded in **personal construct theory** (Kelly, 1955) and **cognitive complexity** (Bieri,
1955). Kelly's repertory grid methodology directly measures how many distinct constructs a
person uses to make sense of a domain — the direct analogue of our task decomposition
granularity. Bieri's (1955) cognitive complexity construct captures differentiation in
interpersonal construal. **Category width** (Pettigrew, 1958) captures individual differences
in the breadth vs. narrowness of categories.

- Kelly, G. A. (1955). *The psychology of personal constructs.* W. W. Norton.
- Bieri, J. (1955). Cognitive complexity-simplicity and predictive behavior. *Journal of Abnormal and Social Psychology*, 51(2), 263–268.
- Pettigrew, T. F. (1958). The measurement and correlates of category width as a cognitive variable. *Journal of Personality*, 26(4), 532–544.

### B16 — Value differentiation

Grounded in **self-concept differentiation** (Showers, 1992; Linville, 1987) — the degree to
which the self-concept is partitioned into distinct, non-overlapping domains. Donahue et al.
(1993) found that higher self-concept differentiation (compartmentalization) is associated
with greater emotional variability. Showers (1992) distinguishes evaluatively integrated
from evaluatively compartmentalized self-organization. **Values clarity**, a related construct
in the acceptance and commitment therapy tradition (Hayes et al., 1999), captures how clearly
an individual can articulate their personal values.

- Linville, P. W. (1987). Self-complexity as a cognitive buffer against stress-related illness and depression. *Journal of Personality and Social Psychology*, 52(4), 663–676.
- Showers, C. J. (1992). Compartmentalization of positive and negative self-knowledge: Keeping bad apples out of the bunch. *Journal of Personality and Social Psychology*, 62(6), 1036–1049.
- Donahue, E. M., Robins, R. W., Roberts, B. W., & John, O. P. (1993). The divided self: Concurrent and longitudinal effects of psychological adjustment and social roles on self-concept differentiation. *Journal of Personality and Social Psychology*, 64(5), 834–846.

### B17 — Schwartz values (v_self_direction through v_universalism)

The Schwartz theory of basic values (Schwartz, 1992; Schwartz et al., 2012) identifies 10
(or 19 in the refined PVQ-RR) motivationally distinct values organized in a circular structure.
The theory has been validated in 80+ countries with 100,000+ respondents. The Portrait Values
Questionnaire (PVQ) and Schwartz Value Survey (SVS) are the standard instruments. The
Self-Direction / Universalism / Benevolence / Tradition / Conformity / Security / Power /
Achievement / Hedonism / Stimulation structure has held across cultures. Schwartz (2012)
provides the definitive overview.

- Schwartz, S. H. (1992). Universals in the content and structure of values: Theoretical advances and empirical tests in 20 countries. In M. P. Zanna (Ed.), *Advances in experimental social psychology* (Vol. 25, pp. 1–65). Academic Press.
- Schwartz, S. H. (2012). An overview of the Schwartz theory of basic values. *Online Readings in Psychology and Culture*, 2(1).
- Schwartz, S. H., Cieciuch, J., Vecchione, M., Davidov, E., Fischer, R., Beierlein, C., ... & Konty, M. (2012). Refining the theory of basic individual values. *Journal of Personality and Social Psychology*, 103(4), 663–688.

---

## Appendix B: Change log from original design

| Change | Rationale |
|---|---|
| Added Layer 0 (role context): profession, seniority, org_context | Fixes "task repertoire should be driven by occupation, not values" gap |
| `coupling_explicitness` → `coupling_awareness` (graph-only → shared disposition) | Fixes latent leakage: both channels see it, expressed differently |
| `boundary_rigidity` → `internal_boundary_clarity` (graph-only → shared disposition) | Fixes latent leakage: text gets hedging/clarity expression, graph gets partition sharpness |
| Added `self_awareness` (shared disposition) | Enables implicit/explicit attitude divergence between channels |
| Added cognitive style: `systematic_vs_intuitive`, `abstract_vs_concrete` (shared) | Adds "how" dimension orthogonal to "what" (values) |
| Wired all 7 dispositions into retention equation | Fixes "most dispositions are decorative" gap |
| `graph_style` → `graph_format` (renamed, reduced to 2 params) | Renamed to clarify these model elicitation-format effects, not hidden psychology |
| `decomposition_granularity` and `value_differentiation` retained as channel-specific | Justified as format demand characteristics, not psychological leakage |
| Task selection now role-driven, not value-driven | Fixes `values → task selection` chain |
| Added profession→task area mapping and task properties (`authorship_weight`, `skill_signaling_weight`) | Enables `competence_attachment` and `identity_attachment` to have differential effects across tasks |
| Added `abstraction_level` to value taxonomy | Enables `abstract_vs_concrete` to influence value selection |
| Added LLM realism risk note + mitigation strategies (§3) | Addresses "LLM text may be unrealistically informative" concern |
| Parameter count: 24 → 30 (free continuous: 29 + 1 categorical) | Reflects added dimensions |
| Added `situational_noise` to `graph_format` (3rd format parameter) | Addresses "graph too clean" — models motor errors, attention lapses, order effects as instrument properties, not psychology |
| Strengthened `graph_format` justification language | Makes explicit that these are elicitation-procedure properties, not latent psychological variables — preempts the skeptical-reviewer question |
| Added realism limitations note (profession granularity, values dominance, situational constraints) | Acknowledges known simplifications without adding premature complexity |
