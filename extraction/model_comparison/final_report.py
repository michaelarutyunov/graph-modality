"""Generate the final 3-model validation report."""

import json

OUT_DIR = "extraction/model_comparison/validation_results"
TIDS = ["work_0657", "creativity_0014", "science_0003"]

# Load all results
results = {}
for model in ["Claude", "Agnes"]:
    # From the earlier run (raw_results.json)
    raw = json.load(open(f"{OUT_DIR}/raw_results.json"))
    results[model] = raw[model]

# DeepSeek via OpenAI endpoint
results["DeepSeek"] = {}
for tid in TIDS:
    path = f"{OUT_DIR}/DeepSeek_openai_{tid}.json"
    g = json.load(open(path))
    results["DeepSeek"][tid] = {"status": "OK", "graph": g}

# Load transcripts
tagged = {}
for split in ["workforce", "creatives", "scientists"]:
    for line in open(f"data/tagged/{split}.jsonl"):
        r = json.loads(line)
        if r["transcript_id"] in TIDS:
            tagged[r["transcript_id"]] = r

# ── Build report ─────────────────────────────────────────────────────

L = []
L.append("# Model Comparison — Final Validation Report")
L.append("")
L.append(f"**3 transcripts × 3 models.** All using prompt v3 (two-shot examples).")
L.append("")
L.append("| Model | Endpoint | Key Config |")
L.append("|---|---|---|")
L.append("| Claude | Anthropic API | claude-sonnet-4-6, max_tokens=4096 |")
L.append("| DeepSeek | OpenAI-compatible API | deepseek-chat, JSON mode, max_tokens=8192 |")
L.append("| Agnes | OpenAI-compatible API | agnes-2.0-flash, max_tokens=4096 |")
L.append("")
L.append("---")
L.append("")

# Source transcripts
L.append("## Source Transcripts")
L.append("")
for tid in TIDS:
    rec = tagged[tid]
    L.append(
        f"### {tid} ({rec['split']}, {rec['n_human_turns']} human turns, "
        f"{len(rec['formatted'])} chars)"
    )
    L.append("")
    L.append("<details><summary>Full formatted transcript</summary>")
    L.append("")
    L.append("```text")
    L.append(rec["formatted"])
    L.append("```")
    L.append("</details>")
    L.append("")
L.append("---")
L.append("")

# Summary table
L.append("## Comparison Summary")
L.append("")
L.append("| Model | Transcript | Nodes | Edges | Bipolarity | Violations | Types |")
L.append("|---|---|---|---|---|---|---|")
for model_name in ["Claude", "DeepSeek", "Agnes"]:
    for tid in TIDS:
        r = results[model_name][tid]
        if r["graph"]:
            g = r["graph"]
            types = {}
            for n in g["nodes"]:
                types[n["type"]] = types.get(n["type"], 0) + 1
            bip = sum(
                1.0 if n.get("bipolarity_complete") else 0.5
                for n in g["nodes"]
                if n["type"] == "Construct"
            )
            n_cons = max(sum(1 for n in g["nodes"] if n["type"] == "Construct"), 1)
            bip /= n_cons
            ts = (
                f"C{types.get('Construct', 0)} V{types.get('Value', 0)} "
                f"S{types.get('Stance', 0)} CSM{types.get('CognitiveStyleMarker', 0)}"
            )
            L.append(
                f"| {model_name} | {tid} | {len(g['nodes'])} | {len(g['edges'])} | "
                f"{bip:.2f} | {len(g['validation_violations'])} | {ts} |"
            )
        else:
            L.append(f"| {model_name} | {tid} | — | — | — | FAIL | — |")
L.append("")
L.append("---")
L.append("")

# Per-model details
for model_name in ["Claude", "DeepSeek", "Agnes"]:
    L.append(f"## {model_name}")
    L.append("")

    for tid in TIDS:
        r = results[model_name][tid]
        L.append(f"### {tid}")
        L.append("")

        if r["graph"] is None:
            L.append(f"**FAILED**")
            L.append("")
            continue

        g = r["graph"]
        L.append(
            f"**{len(g['nodes'])} nodes, {len(g['edges'])} edges, "
            f"{len(g['validation_violations'])} violations**"
        )
        L.append("")

        # Nodes table
        L.append("| ID | Type | Label | Details |")
        L.append("|---|---|---|---|")
        for n in g["nodes"]:
            details = []
            if n["type"] == "Construct":
                details.append(f"↔ {n.get('label_negative', '?')}")
                details.append(f"bip={n.get('bipolarity_complete', '?')}")
            elif n["type"] == "Stance":
                details.append(f"valence={n.get('valence', '?')}")
            span = n.get("grounding_span", "")
            if len(span) > 100:
                span = span[:97] + "..."
            details.append(f'"{span}"')
            L.append(f"| {n['id']} | {n['type']} | {n['label'][:60]} | {'<br>'.join(details)} |")

        L.append("")
        L.append("**Edges:**")
        L.append("")
        for e in g["edges"]:
            L.append(f"- {e['source']} --[{e['relation']}]--> {e['target']}")
        L.append("")

        if g.get("validation_violations"):
            L.append("**Violations:**")
            for v in g["validation_violations"]:
                L.append(f"- {v}")
            L.append("")

        L.append("<details><summary>Raw JSON</summary>")
        L.append("")
        L.append("```json")
        L.append(json.dumps(g, indent=2, ensure_ascii=False))
        L.append("```")
        L.append("</details>")
        L.append("")

    L.append("---")
    L.append("")

report = "\n".join(L)
path = f"{OUT_DIR}/final_validation_report.md"
with open(path, "w", encoding="utf-8") as f:
    f.write(report)
print(f"Report: {len(report)} chars → {path}")
