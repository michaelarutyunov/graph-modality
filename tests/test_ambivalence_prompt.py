from pathlib import Path

PROMPT = Path("s2_extraction/prompts/ambivalence_v1.txt")


def test_prompt_exists_and_defines_ordinal_scheme():
    text = PROMPT.read_text(encoding="utf-8")
    # three ordinal levels present
    for level in ('"low"', '"med"', '"high"'):
        assert level in text, f"missing level {level}"
    # the JSON output key the labeler/loader depend on
    assert "stance_ambivalence" in text
    # holistic-judgment guardrail: must forbid counting graph features
    assert "do not count" in text.lower() or "not by counting" in text.lower()
    # evidence requirement
    assert "quote" in text.lower()
