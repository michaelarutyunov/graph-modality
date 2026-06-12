from s2_extraction.ambivalence_labeler import (
    _build_human_text,
    _strip_fences,
    _validate_ambivalence,
)


def test_strip_fences_removes_markdown():
    assert _strip_fences('```json\n{"a": 1}\n```') == '{"a": 1}'
    assert _strip_fences('{"a": 1}') == '{"a": 1}'


def test_build_human_text_keeps_only_human_turns():
    record = {
        "turns": [
            {"speaker": "AI", "text": "as a creative professional, tell me..."},
            {"speaker": "Human", "text": "I use AI daily."},
            {"speaker": "Human", "text": "But I worry about it."},
        ]
    }
    out = _build_human_text(record)
    assert "as a creative professional" not in out
    assert "[Human]: I use AI daily." in out
    assert "[Human]: But I worry about it." in out


def test_validate_flags_invalid_label_and_missing_quotes():
    bad = {
        "transcript_id": "t1",
        "stance_ambivalence": {"label": "extreme", "quotes": [], "reasoning": "x"},
    }
    warns = _validate_ambivalence(bad, "t1")
    assert any("invalid" in w for w in warns)

    no_quotes = {
        "transcript_id": "t1",
        "stance_ambivalence": {"label": "high", "quotes": [], "reasoning": "x"},
    }
    warns = _validate_ambivalence(no_quotes, "t1")
    assert any("no quotes" in w for w in warns)


def test_validate_passes_uncertain_without_quotes():
    ok = {
        "transcript_id": "t1",
        "stance_ambivalence": {"label": "uncertain", "quotes": [], "reasoning": "thin"},
    }
    assert _validate_ambivalence(ok, "t1") == []
