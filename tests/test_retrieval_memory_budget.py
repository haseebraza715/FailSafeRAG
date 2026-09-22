from __future__ import annotations

import numpy as np
import pytest

from faar import recovery, resource_limits, retrieval
from faar.settings import RetrievalSettings
from faar.types import Chunk


def _chunks(count: int = 2) -> list[Chunk]:
    return [
        Chunk(
            chunk_id=f"c{index}",
            example_id="e0",
            doc_name="doc",
            page_id=index + 1,
            text=f"sample chunk text {index}",
        )
        for index in range(count)
    ]


class _FakeEmbedder:
    def encode(self, sentences, **kwargs):
        return np.zeros((len(sentences), 8), dtype=np.float32)


class _FakeReranker:
    def predict(self, pairs, **kwargs):
        return [0.5] * len(pairs)


def test_hybrid_retriever_construction_calls_memory_budget(monkeypatch) -> None:
    calls: list[str] = []

    class FakeTorch:
        pass

    monkeypatch.setattr(
        retrieval,
        "enforce_memory_budget",
        lambda stage, torch_module=None: calls.append(stage),
    )
    monkeypatch.setattr(retrieval, "import_module", lambda name: FakeTorch())
    monkeypatch.setattr(retrieval, "_load_embedding_model", lambda *args: _FakeEmbedder())
    monkeypatch.setattr(retrieval, "_load_reranker", lambda *args: _FakeReranker())

    settings = RetrievalSettings()
    settings.embed_batch_size = 4
    retriever = retrieval.HybridRetriever(_chunks(), settings)

    assert calls == ["text retrieval model load", "text retrieval corpus encode"]
    assert retriever._dense_index.ntotal == 2


def test_byt5_load_calls_memory_budget(monkeypatch) -> None:
    calls: list[str] = []

    class FakeTorch:
        pass

    class _Chain:
        def to(self, device):
            return self

        def eval(self):
            return self

    class FakeTokenizer:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            return object()

    class FakeModel:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            return _Chain()

    monkeypatch.setattr(
        recovery,
        "enforce_memory_budget",
        lambda stage, torch_module=None: calls.append(stage),
    )
    monkeypatch.setattr(recovery, "import_module", lambda name: FakeTorch())
    monkeypatch.setattr("transformers.AutoTokenizer", FakeTokenizer)
    monkeypatch.setattr("transformers.AutoModelForSeq2SeqLM", FakeModel)
    monkeypatch.setattr(recovery, "torch_device", lambda torch: "cpu")
    monkeypatch.setattr(recovery, "select_dtype", lambda device, torch: None)
    monkeypatch.setattr(recovery, "enforce_gpu_memory_fraction", lambda torch: None)

    recovery._load_byt5.cache_clear()
    tokenizer, model = recovery._load_byt5("mock/byt5", None)
    recovery._load_byt5.cache_clear()

    assert calls == ["ByT5 model load"]
    assert tokenizer is not None
    assert model is not None


@pytest.mark.parametrize("value", ["nan", "inf", "-inf", "NaN", "Infinity"])
def test_optional_gb_rejects_non_finite(monkeypatch, value: str) -> None:
    monkeypatch.setenv("FAAR_MAX_RSS_GB", value)
    with pytest.raises(ValueError, match="finite"):
        resource_limits._optional_gb("FAAR_MAX_RSS_GB")
