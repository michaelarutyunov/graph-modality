"""Tests for extraction.tagger."""

from s2_extraction.tagger import Turn, format_for_extraction, parse_transcript


def test_parse_simple_transcript():
    text = "Assistant: Hello!\nUser: Hi there!\nAI: How are you?\nUser: I'm good."
    turns = parse_transcript(text)
    assert len(turns) == 4
    assert turns[0] == Turn(speaker="AI", text="Hello!", turn_index=0)
    assert turns[1] == Turn(speaker="Human", text="Hi there!", turn_index=1)
    assert turns[2] == Turn(speaker="AI", text="How are you?", turn_index=2)
    assert turns[3] == Turn(speaker="Human", text="I'm good.", turn_index=3)


def test_parse_multiline_turns():
    text = "AI: Line one.\nLine two.\nUser: Response line.\nAlso response line."
    turns = parse_transcript(text)
    assert len(turns) == 2
    assert turns[0].speaker == "AI"
    assert "Line one.\nLine two." in turns[0].text


def test_format_for_extraction():
    turns = [
        Turn(speaker="AI", text="Hello!", turn_index=0),
        Turn(speaker="Human", text="I think AI is useful.", turn_index=1),
    ]
    formatted = format_for_extraction(turns)
    assert "[AI]:" in formatted
    assert "[Human]:" in formatted
    assert "Hello!" in formatted
    assert "I think AI is useful." in formatted


def test_parse_empty_transcript():
    turns = parse_transcript("")
    assert turns == []


def test_parse_user_only():
    text = "User: I use AI daily."
    turns = parse_transcript(text)
    assert len(turns) == 1
    assert turns[0].speaker == "Human"
    assert turns[0].text == "I use AI daily."


def test_parse_ai_only():
    text = "Assistant: Welcome to the interview.\nAI: Let's begin."
    turns = parse_transcript(text)
    assert len(turns) == 2
    assert all(t.speaker == "AI" for t in turns)


def test_mid_text_assistant_not_split():
    """Regression: 'assistant' in User text must not trigger a false split."""
    text = (
        "AI: What do you do?\n"
        "User: I work as a virtual assistant: I handle scheduling and email.\n"
        "AI: Interesting."
    )
    turns = parse_transcript(text)
    assert len(turns) == 3
    human_turn = turns[1]
    assert human_turn.speaker == "Human"
    assert "virtual assistant: I handle scheduling" in human_turn.text


def test_mid_text_user_not_split():
    """Regression: 'User' in AI text must not trigger a false split."""
    text = "AI: User experience is a key consideration in our design process.\nUser: I agree."
    turns = parse_transcript(text)
    assert len(turns) == 2


def test_whitespace_only_turn():
    """Turn with only whitespace is included with stripped (empty) text."""
    text = "Assistant: Hello!\nUser:   \nAI: Continue."
    turns = parse_transcript(text)
    assert len(turns) == 3
    assert turns[1].speaker == "Human"
    assert turns[1].text == ""


def test_consecutive_same_speaker():
    """Consecutive turns by the same speaker are preserved as distinct turns."""
    text = "AI: First question.\nAI: Follow-up.\nUser: Answer."
    turns = parse_transcript(text)
    assert len(turns) == 3
    assert turns[0].speaker == "AI"
    assert turns[1].speaker == "AI"
    assert turns[0].text == "First question."
    assert turns[1].text == "Follow-up."
