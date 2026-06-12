from s2_extraction.ambivalence_consensus import compute_agreement, merge_consensus


def _entry(tid, label):
    return {
        "transcript_id": tid,
        "stance_ambivalence": {"label": label, "quotes": [], "reasoning": ""},
    }


def test_compute_agreement_rate_and_kappa():
    a = {"t1": _entry("t1", "low"), "t2": _entry("t2", "high"), "t3": _entry("t3", "med")}
    b = {"t1": _entry("t1", "low"), "t2": _entry("t2", "med"), "t3": _entry("t3", "med")}
    stats = compute_agreement(a, b)
    assert stats["n_common"] == 3
    assert stats["n_agree"] == 2
    assert abs(stats["agreement_rate"] - 2 / 3) < 1e-9
    assert "kappa" in stats
    assert stats["disagreements"] == ["t2"]


def test_merge_consensus_accepts_agreements_and_applies_adjudications():
    a = {"t1": _entry("t1", "low"), "t2": _entry("t2", "high")}
    b = {"t1": _entry("t1", "low"), "t2": _entry("t2", "med")}
    adjudications = {"t2": "high"}  # user resolved the disagreement
    final = merge_consensus(a, b, adjudications)
    assert final["t1"]["stance_ambivalence"]["label"] == "low"
    assert final["t1"]["stance_ambivalence"]["source"] == "consensus"
    assert final["t2"]["stance_ambivalence"]["label"] == "high"
    assert final["t2"]["stance_ambivalence"]["source"] == "adjudicated"


def test_merge_consensus_skips_unadjudicated_disagreements():
    a = {"t2": _entry("t2", "high")}
    b = {"t2": _entry("t2", "med")}
    final = merge_consensus(a, b, adjudications={})
    assert "t2" not in final  # unresolved disagreements are excluded, not guessed
