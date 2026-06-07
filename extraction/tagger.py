"""Speaker-tag transcript turns for graph extraction.

Transcripts use three speaker prefixes, confirmed by dataset inspection:
- ``Assistant:`` — first AI turn (opening)
- ``AI:``         — subsequent AI turns
- ``User:``       — all human turns

This module parses those turns and writes a tagged .jsonl file for each CSV split.

Usage:
    uv run python extraction/tagger.py
"""

import json
import re
from dataclasses import dataclass
from pathlib import Path

import polars as pl

RAW_DIR = Path("data/raw/interview_transcripts")
TAGGED_DIR = Path("data/tagged")

SPLITS = ["workforce", "creatives", "scientists"]

# Speaker prefixes found in the dataset.  ``Assistant`` appears only as the
# opening turn; subsequent AI turns use ``AI``.  Human turns use ``User``.
_SPEAKER_PREFIX = re.compile(r"^(Assistant|AI|User):\s*", re.MULTILINE)


@dataclass
class Turn:
    """A single speaker turn in a transcript."""

    speaker: str  # "AI" | "Human"
    text: str
    turn_index: int


def parse_transcript(raw_text: str) -> list[Turn]:
    """Split a transcript into tagged turns.

    Handles the three-prefix convention:
    ``Assistant:`` (opening AI), ``AI:`` (subsequent AI),
    ``User:`` (human).

    The ``^`` anchor with ``re.MULTILINE`` prevents false splits on
    mid-text mentions like "I work as an assistant."
    """
    parts = _SPEAKER_PREFIX.split(raw_text.strip())
    # parts alternates: [pre-match, speaker, text, speaker, text, ...]
    # pre-match is '' when the transcript starts with a speaker tag
    turns: list[Turn] = []
    i = 1
    while i < len(parts) - 1:
        speaker_raw = parts[i].strip()
        text = parts[i + 1].strip()
        # Map: Assistant / AI → AI, User → Human
        speaker = "Human" if speaker_raw == "User" else "AI"
        turns.append(Turn(speaker=speaker, text=text, turn_index=len(turns)))
        i += 2
    return turns


def format_for_extraction(turns: list[Turn]) -> str:
    """Format tagged turns for insertion into an extraction prompt.

    Returns a string with ``[AI]:`` / ``[Human]:`` prefixes, double-newline
    separated, suitable for pasting into the prompt template.
    """
    lines: list[str] = []
    for t in turns:
        tag = "[AI]" if t.speaker == "AI" else "[Human]"
        lines.append(f"{tag}: {t.text}")
    return "\n\n".join(lines)


def tag_split(split: str) -> list[dict]:
    """Parse one CSV split and return tagged transcript records."""
    path = RAW_DIR / f"{split}_transcripts.csv"
    df = pl.read_csv(path)

    records: list[dict] = []
    for row in df.iter_rows(named=True):
        transcript_id: str = row["transcript_id"]
        raw_text: str = row["text"]
        turns = parse_transcript(raw_text)
        formatted = format_for_extraction(turns)

        records.append(
            {
                "transcript_id": transcript_id,
                "split": split,
                "n_turns": len(turns),
                "n_ai_turns": sum(1 for t in turns if t.speaker == "AI"),
                "n_human_turns": sum(1 for t in turns if t.speaker == "Human"),
                "turns": [
                    {
                        "speaker": t.speaker,
                        "text": t.text,
                        "turn_index": t.turn_index,
                    }
                    for t in turns
                ],
                "formatted": formatted,
            }
        )

    return records


def main() -> None:
    """Tag all splits and write one .jsonl file per split."""
    TAGGED_DIR.mkdir(parents=True, exist_ok=True)

    for split in SPLITS:
        out_path = TAGGED_DIR / f"{split}.jsonl"
        records = tag_split(split)

        with open(out_path, "w", encoding="utf-8") as f:
            for rec in records:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")

        print(f"  {split}: {len(records)} transcripts → {out_path}")


if __name__ == "__main__":
    main()
