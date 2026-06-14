"""Tests for s2_extraction.ambivalence_adjudicator."""

from __future__ import annotations

import json
import sys
from typing import TYPE_CHECKING

import pytest

from s2_extraction.ambivalence_adjudicator import (
    _annotate_label,
    _build_judge_prompt,
    _call_kimi,
    _compute_disagreements,
    _load_existing_adjudications,
    _load_existing_details,
    _parse_judge_output,
)

if TYPE_CHECKING:
    from pathlib import Path


def _make_annotation(label: str, model: str) -> dict:
    return {
        "transcript_id": "t1",
        "stance_ambivalence": {
            "label": label,
            "reasoning": f"reason for {label}",
            "quotes": [f"quote {label}"],
        },
        "_model": model,
    }


def test_compute_disagreements():
    a = {
        "t1": _make_annotation("low", "agnes"),
        "t2": _make_annotation("med", "agnes"),
        "t3": _make_annotation("high", "agnes"),
    }
    b = {
        "t1": _make_annotation("low", "haiku"),
        "t2": _make_annotation("high", "haiku"),
        "t4": _make_annotation("med", "haiku"),
    }
    result = _compute_disagreements(a, b)
    assert result == [("t2", "med", "high")]


def test_annotate_label_formats_all_fields():
    ann = _make_annotation("med", "agnes")
    text = _annotate_label(ann)
    assert "Label: med" in text
    assert "reason for med" in text
    assert "quote med" in text


def test_annotate_label_handles_missing_quotes():
    ann = _make_annotation("uncertain", "agnes")
    ann["stance_ambivalence"]["quotes"] = []
    text = _annotate_label(ann)
    assert "(none provided)" in text


def test_build_judge_prompt_contains_anonymized_annotations():
    record = {"transcript_id": "t1", "turns": [{"speaker": "Human", "text": "I like AI."}]}
    ann_a = _make_annotation("low", "agnes")
    ann_b = _make_annotation("high", "haiku")
    system, user, order = _build_judge_prompt("rubric text", record, ann_a, ann_b)

    assert "rubric text" in system
    assert "Annotator A" in user
    assert "Annotator B" in user
    assert "Label: low" in user or "Label: high" in user
    assert "reason for low" in user
    assert "reason for high" in user
    assert "I like AI." in user
    assert set(order) == {"agnes", "haiku"}


def test_build_judge_prompt_randomizes_order():
    """Different transcript IDs should occasionally produce different A/B orders."""
    ann_a = _make_annotation("low", "agnes")
    ann_b = _make_annotation("high", "haiku")

    orders = set()
    for i in range(20):
        record = {"transcript_id": f"t{i}", "turns": [{"speaker": "Human", "text": "x"}]}
        order = _build_judge_prompt("rubric", record, ann_a, ann_b)[2]
        orders.add(order)
    assert len(orders) > 1


def test_parse_judge_output_maps_annotator_to_model():
    detail = _parse_judge_output(
        {
            "chosen_label": "med",
            "chosen_annotator": "A",
            "reasoning": "r",
            "supporting_quotes": ["q"],
        },
        "t1",
        ("agnes", "haiku"),
    )
    assert detail["chosen_label"] == "med"
    assert detail["chosen_annotator"] == "A"
    assert detail["chosen_model"] == "agnes"
    assert detail["supporting_quotes"] == ["q"]


def test_parse_judge_output_invalid_label_becomes_manual_review():
    detail = _parse_judge_output(
        {"chosen_label": "extreme", "chosen_annotator": "B", "reasoning": "r"},
        "t1",
        ("agnes", "haiku"),
    )
    assert detail["chosen_label"] == "manual_review"
    assert detail["chosen_model"] == "haiku"


def test_parse_judge_output_uncertain_maps_to_none():
    detail = _parse_judge_output(
        {"chosen_label": "uncertain", "chosen_annotator": "none", "reasoning": "r"},
        "t1",
        ("agnes", "haiku"),
    )
    assert detail["chosen_label"] == "uncertain"
    assert detail["chosen_model"] is None


def test_load_existing_details(tmp_path: Path):
    details_path = tmp_path / "details.jsonl"
    details_path.write_text(
        json.dumps({"transcript_id": "t1", "chosen_label": "low"}) + "\n",
        encoding="utf-8",
    )
    result = _load_existing_details(details_path)
    assert result == {"t1": {"transcript_id": "t1", "chosen_label": "low"}}


def test_load_existing_adjudications(tmp_path: Path):
    path = tmp_path / "adj.json"
    path.write_text(json.dumps({"t1": "low", "t2": "high"}), encoding="utf-8")
    assert _load_existing_adjudications(path) == {"t1": "low", "t2": "high"}


def test_load_existing_adjudications_missing_returns_empty(tmp_path: Path):
    assert _load_existing_adjudications(tmp_path / "does_not_exist.json") == {}


def test_call_kimi_dry_run_returns_placeholder(monkeypatch):
    # Ensure env key is not required in dry-run mode.
    monkeypatch.delenv("KIMI_API_KEY", raising=False)
    result = _call_kimi("sys", "user", dry_run=True)
    assert result["chosen_label"] == "manual_review"
    assert result["chosen_annotator"] == "none"
    assert "dry-run" in result["reasoning"]


@pytest.fixture
def mock_disagreement_data(monkeypatch, tmp_path: Path):
    """Set up in-memory caches and a temporary output directory."""
    from s2_extraction import ambivalence_adjudicator as adj

    a = {
        "t1": _make_annotation("low", "agnes"),
        "t2": _make_annotation("med", "agnes"),
    }
    b = {
        "t1": _make_annotation("high", "haiku"),
        "t2": _make_annotation("med", "haiku"),
    }
    records = {
        "t1": {"transcript_id": "t1", "turns": [{"speaker": "Human", "text": "x"}]},
        "t2": {"transcript_id": "t2", "turns": [{"speaker": "Human", "text": "y"}]},
    }

    monkeypatch.setattr(adj, "CACHE_DIR", tmp_path)
    monkeypatch.setattr(adj, "ADJUDICATION_PATH", tmp_path / "adj.json")
    monkeypatch.setattr(adj, "DETAILS_PATH", tmp_path / "details.jsonl")
    monkeypatch.setattr(adj, "DISAGREEMENT_PATH", tmp_path / "disagreements.json")
    monkeypatch.setattr(adj, "_load_jsonl", lambda _p: a if "agnes" in str(_p) else b)
    monkeypatch.setattr(adj, "_load_tagged_records", lambda: records)

    calls: list[tuple[str, str, bool]] = []

    def _fake_call_kimi(system: str, user: str, dry_run: bool = False) -> dict:
        calls.append((system, user, dry_run))
        # Always side with Annotator A in the prompt.
        return {
            "chosen_label": "med",
            "chosen_annotator": "A",
            "reasoning": "picked A",
            "supporting_quotes": ["q"],
        }

    monkeypatch.setattr(adj, "_call_kimi", _fake_call_kimi)
    monkeypatch.setattr(adj, "RATE_LIMIT", 0.0)
    return calls


def test_main_adjudicates_disagreements(mock_disagreement_data, tmp_path: Path, monkeypatch):
    from s2_extraction import ambivalence_adjudicator as adj

    monkeypatch.setattr(sys, "argv", ["prog"])
    adj.main()

    calls = mock_disagreement_data
    assert len(calls) == 1  # only t1 disagrees

    adjudications = json.loads((tmp_path / "adj.json").read_text(encoding="utf-8"))
    assert adjudications == {"t1": "med"}

    details = [
        json.loads(line)
        for line in (tmp_path / "details.jsonl").read_text(encoding="utf-8").strip().split("\n")
    ]
    assert len(details) == 1
    assert details[0]["transcript_id"] == "t1"
    assert details[0]["chosen_label"] == "med"


def test_main_resumes_from_existing_details(mock_disagreement_data, tmp_path: Path, monkeypatch):
    from s2_extraction import ambivalence_adjudicator as adj

    monkeypatch.setattr(sys, "argv", ["prog"])

    # Pre-populate one adjudication detail so t1 is skipped.
    (tmp_path / "details.jsonl").write_text(
        json.dumps({"transcript_id": "t1", "chosen_label": "low"}) + "\n",
        encoding="utf-8",
    )

    adj.main()
    calls = mock_disagreement_data
    assert len(calls) == 0


def test_main_limit(mock_disagreement_data, monkeypatch):
    from s2_extraction import ambivalence_adjudicator as adj

    # Add a second disagreement so we can test --limit.
    a = {
        "t1": _make_annotation("low", "agnes"),
        "t2": _make_annotation("low", "agnes"),
    }
    b = {
        "t1": _make_annotation("high", "haiku"),
        "t2": _make_annotation("high", "haiku"),
    }
    monkeypatch.setattr(
        adj, "_load_jsonl", lambda _p: a if "agnes" in str(_p) else b
    )

    monkeypatch.setattr(sys, "argv", ["prog", "--limit", "1"])
    adj.main()

    calls = mock_disagreement_data
    assert len(calls) == 1
