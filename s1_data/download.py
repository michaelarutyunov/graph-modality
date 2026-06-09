"""Download the Anthropic Interviewer dataset from HuggingFace.

Idempotent: skips download if all expected files are present.
Run once before any pipeline stage.

Usage:
    uv run python data/download.py
"""

from pathlib import Path

from huggingface_hub import snapshot_download

RAW_DIR = Path("s1_data/raw")
EXPECTED = [
    "interview_transcripts/workforce_transcripts.csv",
    "interview_transcripts/creatives_transcripts.csv",
    "interview_transcripts/scientists_transcripts.csv",
]


def ensure_dataset() -> None:
    """Download dataset files that are not already present."""
    missing = [f for f in EXPECTED if not (RAW_DIR / f).exists()]
    if not missing:
        print("dataset already present — skipping download")
        return

    print(f"downloading {len(missing)} missing file(s)...")
    snapshot_download(
        repo_id="Anthropic/AnthropicInterviewer",
        repo_type="dataset",
        local_dir=RAW_DIR,
        ignore_patterns=["*.md", "*.json"],
    )
    print("download complete")


if __name__ == "__main__":
    ensure_dataset()
