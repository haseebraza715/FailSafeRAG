from faar.recovery import ByT5Corrector


def test_propose_correction_skips_non_latin_sources() -> None:
    corrector = ByT5Corrector("google/byt5-small")
    proposal = corrector.propose_correction("安徽省在三资清理中采取了哪些措施？")
    assert proposal["applied"] is False
    assert proposal["reason"] == "non_latin_source"


def test_propose_correction_rejects_numeric_drift(monkeypatch) -> None:
    corrector = ByT5Corrector("google/byt5-small")
    monkeypatch.setattr(
        ByT5Corrector,
        "_generate_correction",
        lambda self, text, max_new_tokens=128: "invoice total $91 due on receipt",
    )
    proposal = corrector.propose_correction("invoice t0tal $98 due on receipt")
    assert proposal["applied"] is False
    assert proposal["reason"] == "numeric_signature_changed"
    assert proposal["text"] == "invoice t0tal $98 due on receipt"


def test_propose_correction_skips_clean_text() -> None:
    corrector = ByT5Corrector("google/byt5-small")
    proposal = corrector.propose_correction("installation requires certified engineers")
    assert proposal["applied"] is False
    assert proposal["reason"] == "source_not_noisy_enough"
