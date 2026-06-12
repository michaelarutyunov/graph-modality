import json

import s4_encoding.build_dataset as bd


def test_load_ambivalence_labels(tmp_path, monkeypatch):
    path = tmp_path / "ambivalence.jsonl"
    records = [
        {"transcript_id": "t1", "stance_ambivalence": {"label": "low", "source": "consensus"}},
        {"transcript_id": "t2", "stance_ambivalence": {"label": "med", "source": "consensus"}},
        {"transcript_id": "t3", "stance_ambivalence": {"label": "high", "source": "adjudicated"}},
    ]
    with open(path, "w", encoding="utf-8") as f:
        for r in records:
            json.dump(r, f)
            f.write("\n")

    monkeypatch.setattr(bd, "AMBIVALENCE_PATH", path)
    labels = bd._load_ambivalence_labels()

    assert labels == {"t1": 0, "t2": 1, "t3": 2}
