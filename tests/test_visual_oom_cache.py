from __future__ import annotations

import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from faar import visual_baselines
from faar.benchmarks import BenchmarkRepository
from faar.data import DatasetUnavailableError
from faar.settings import AppSettings


def _image_paths(tmp_path: Path, count: int = 6) -> list[Path]:
    from PIL import Image

    paths = []
    for index in range(count):
        path = tmp_path / f"page_{index}.png"
        Image.new("RGB", (8, 8), color=(index * 40, 0, 0)).save(path)
        paths.append(path)
    return paths


class _FakeBatch(dict):
    def to(self, device):
        return self


class _FakeOutput:
    def __init__(self, embeddings) -> None:
        self.embeddings = embeddings


class _CountingModel:
    def __init__(self, seq: int = 4, dim: int = 8, forward=None) -> None:
        self.seq = seq
        self.dim = dim
        self.forward_calls = 0
        self._forward = forward

    def __call__(self, **inputs):
        self.forward_calls += 1
        if self._forward is not None:
            return self._forward(self, inputs)
        n = inputs.get("_n", 1)
        base = torch.arange(n * self.seq * self.dim, dtype=torch.float32)
        return _FakeOutput(base.reshape(n, self.seq, self.dim))

    def to(self, device):
        return self

    def eval(self):
        return self


class _FakeProcessor:
    def __call__(self, **inputs):
        images = inputs.get("images")
        return _FakeBatch({"_n": len(images) if images is not None else 1})

    def to(self, device):
        return self

    def score_retrieval(self, query_embeddings, passage_embeddings):
        query = query_embeddings[0]
        dots = torch.einsum("qd,nsd->nqs", query, passage_embeddings)
        scores = dots.max(dim=2).values.sum(dim=1)
        return scores.unsqueeze(0)


def _install_colpali_fakes(monkeypatch: pytest.MonkeyPatch, model) -> _FakeProcessor:
    import transformers

    processor = _FakeProcessor()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return model

    class FakeProcessorClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return processor

    resolved_model_class = transformers.ColPaliForRetrieval
    resolved_processor_class = transformers.ColPaliProcessor
    assert resolved_model_class is not None and resolved_processor_class is not None
    monkeypatch.setattr(visual_baselines, "torch_device", lambda torch_module: torch.device("cpu"))
    monkeypatch.setattr(visual_baselines, "select_dtype", lambda device, torch_module: torch.float32)
    monkeypatch.setattr(visual_baselines, "release_cuda_cache", lambda torch_module: None)
    monkeypatch.setattr("transformers.ColPaliForRetrieval", FakeModelClass)
    monkeypatch.setattr("transformers.ColPaliProcessor", FakeProcessorClass)
    return processor


def _cache_dir(tmp_path: Path) -> Path:
    return tmp_path / "cache" / "visual_embeddings"


def test_cache_hit_skips_reencoding_and_loads_identical_embeddings(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    paths = _image_paths(tmp_path, count=4)
    model = _CountingModel()
    _install_colpali_fakes(monkeypatch, model)
    cache_dir = _cache_dir(tmp_path)

    first = visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)
    assert model.forward_calls == 2
    content_hashes = visual_baselines._compute_content_hashes(paths)
    key = visual_baselines._visual_cache_key(
        visual_baselines.VISUAL_CACHE_SCHEMA_VERSION,
        "mock/colpali",
        None,
        "torch.float32",
        paths,
        content_hashes,
    )
    assert (cache_dir / f"{key}.npz").is_file()
    assert (cache_dir / f"{key}.meta.json").is_file()

    calls_before = model.forward_calls
    second = visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)

    assert model.forward_calls == calls_before
    assert torch.equal(first.image_embeddings, second.image_embeddings)
    assert second.image_embeddings.shape == (4, model.seq, model.dim)


@pytest.mark.parametrize("mutation", ["path", "revision"])
def test_cache_misses_on_identity_mismatch(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    from PIL import Image

    paths = _image_paths(tmp_path, count=4)
    model = _CountingModel()
    _install_colpali_fakes(monkeypatch, model)
    cache_dir = _cache_dir(tmp_path)

    visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)
    calls_after_first = model.forward_calls

    if mutation == "path":
        extra = tmp_path / "extra.png"
        Image.new("RGB", (8, 8)).save(extra)
        changed_paths = paths + [extra]
        revision = None
    else:
        changed_paths = paths
        revision = "abc123"

    second = visual_baselines.ColPaliRetriever(
        changed_paths, "mock/colpali", revision, batch_size=2, cache_dir=cache_dir
    )

    assert model.forward_calls > calls_after_first
    assert second.image_embeddings.shape[0] == len(changed_paths)


def test_cache_misses_when_bytes_change_at_same_path(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    paths = _image_paths(tmp_path, count=4)
    model = _CountingModel()
    _install_colpali_fakes(monkeypatch, model)
    cache_dir = _cache_dir(tmp_path)

    visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)
    calls_after_first = model.forward_calls

    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(paths[1])

    second = visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)

    assert model.forward_calls > calls_after_first
    assert second.image_embeddings.shape == (4, model.seq, model.dim)


def test_cache_misses_when_dtype_changes(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    paths = _image_paths(tmp_path, count=4)
    model = _CountingModel()
    _install_colpali_fakes(monkeypatch, model)
    cache_dir = _cache_dir(tmp_path)

    visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)
    calls_after_first = model.forward_calls

    monkeypatch.setattr(visual_baselines, "select_dtype", lambda device, torch_module: torch.float16)
    second = visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)

    assert model.forward_calls > calls_after_first
    assert second.image_embeddings.shape == (4, model.seq, model.dim)


@pytest.mark.parametrize("mutation", ["corrupt", "incomplete"])
def test_cache_recomputes_on_invalid_metadata(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mutation: str
) -> None:
    paths = _image_paths(tmp_path, count=4)
    model = _CountingModel()
    _install_colpali_fakes(monkeypatch, model)
    cache_dir = _cache_dir(tmp_path)

    visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)
    calls_after_first = model.forward_calls

    content_hashes = visual_baselines._compute_content_hashes(paths)
    key = visual_baselines._visual_cache_key(
        visual_baselines.VISUAL_CACHE_SCHEMA_VERSION,
        "mock/colpali",
        None,
        "torch.float32",
        paths,
        content_hashes,
    )
    meta_path = cache_dir / f"{key}.meta.json"
    if mutation == "corrupt":
        meta_path.write_text("{corrupt", encoding="utf-8")
    else:
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        del meta["image_hashes"]
        meta_path.write_text(json.dumps(meta), encoding="utf-8")

    second = visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir)

    assert model.forward_calls > calls_after_first
    assert second.image_embeddings.shape == (4, model.seq, model.dim)


def _install_visrag_fakes(monkeypatch: pytest.MonkeyPatch, encode_calls: list[int]) -> None:
    class FakeTokenizerClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return object()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            class Chain:
                def to(self, device):
                    return self

                def eval(self):
                    return self

            return Chain()

    monkeypatch.setattr(visual_baselines, "torch_device", lambda torch_module: torch.device("cpu"))
    monkeypatch.setattr(visual_baselines, "select_dtype", lambda device, torch_module: torch.float32)
    monkeypatch.setattr(visual_baselines, "release_cuda_cache", lambda torch_module: None)
    monkeypatch.setattr("transformers.AutoTokenizer", FakeTokenizerClass)
    monkeypatch.setattr("transformers.AutoModel", FakeModelClass)

    def fake_encode(self, values):
        encode_calls.append(len(values))
        return np.zeros((len(values), 8), dtype=np.float32)

    monkeypatch.setattr(visual_baselines.VisRAGRetriever, "_encode", fake_encode)


@pytest.mark.parametrize("mode", ["colpali", "visrag"])
def test_both_retrievers_bind_cache_to_image_contents(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    from PIL import Image

    paths = _image_paths(tmp_path, count=4)
    cache_dir = _cache_dir(tmp_path)
    if mode == "colpali":
        model = _CountingModel()
        _install_colpali_fakes(monkeypatch, model)

        def build():
            return visual_baselines.ColPaliRetriever(
                paths, "mock/colpali", None, batch_size=2, cache_dir=cache_dir
            )

        def work_count():
            return model.forward_calls
    else:
        encode_calls: list[int] = []
        _install_visrag_fakes(monkeypatch, encode_calls)

        def build():
            return visual_baselines.VisRAGRetriever(
                paths, "mock/visrag", None, batch_size=2, cache_dir=cache_dir
            )

        def work_count():
            return len(encode_calls)

    build()
    after_first = work_count()

    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(paths[2])

    retriever = build()
    assert work_count() > after_first
    assert retriever.image_embeddings.shape[0] == 4


def test_ordered_content_hashes_prefers_recorded_manifest_hashes(tmp_path: Path) -> None:
    paths = _image_paths(tmp_path, count=2)
    recorded = [(paths[0], "a" * 64), (paths[1], "b" * 64)]
    result = visual_baselines._ordered_content_hashes(paths, recorded)
    assert result == [[str(paths[0]), "a" * 64], [str(paths[1]), "b" * 64]]
    computed = visual_baselines._ordered_content_hashes(paths, None)
    assert computed != result


def test_content_hash_changes_when_bytes_change_at_same_path(tmp_path: Path) -> None:
    from PIL import Image

    paths = _image_paths(tmp_path, count=1)
    before = visual_baselines._compute_content_hashes(paths)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(paths[0])
    after = visual_baselines._compute_content_hashes(paths)
    assert before != after


def _hash_repository(tmp_path: Path) -> BenchmarkRepository:
    """Real BenchmarkRepository whose corpus pages record verified image_sha256 claims."""
    from PIL import Image

    image_dir = tmp_path / "images"
    ocr_dir = tmp_path / "ocr"
    image_dir.mkdir()
    ocr_dir.mkdir()
    corpus_pages = []
    for page_id in (1, 2):
        image_path = image_dir / f"doc_page_{page_id}.png"
        Image.new("RGB", (8, 8), color=(page_id * 40, 0, 0)).save(image_path)
        (ocr_dir / f"doc_page_{page_id}.txt").write_text(f"OCR {page_id}")
        corpus_pages.append(
            {
                "corpus_id": f"doc:p{page_id}",
                "doc_name": "doc",
                "page_id": page_id,
                "ocr_text_path": str(ocr_dir / f"doc_page_{page_id}.txt"),
                "image_path": str(image_path),
                "image_sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            }
        )
    records = [
        {
            "example_id": "e1",
            "doc_name": "doc",
            "question": "Q",
            "correct_answer": "A",
            "page_ids": [1],
            "corpus_ids": ["doc:p1", "doc:p2"],
        }
    ]
    return BenchmarkRepository(
        records,
        corpus_pages,
        tmp_path,
        "ohrbench",
        "test",
        document_inventory={"doc": [1, 2]},
    )


def test_verified_unchanged_corpus_hits_visual_cache(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    repository = _hash_repository(tmp_path)
    model = _CountingModel()
    _install_colpali_fakes(monkeypatch, model)
    _cache_dir(tmp_path)
    settings = AppSettings(project_root=tmp_path)

    first = visual_baselines.build_visual_retriever("colpali", repository, settings)
    assert model.forward_calls > 0
    calls_after_first = model.forward_calls

    second = visual_baselines.build_visual_retriever("colpali", repository, settings)
    assert model.forward_calls == calls_after_first
    assert torch.equal(first.image_embeddings, second.image_embeddings)
    assert second.image_embeddings.shape[0] == 2


def test_hash_verification_fails_before_model_or_cache_use(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    from PIL import Image

    repository = _hash_repository(tmp_path)
    Image.new("RGB", (8, 8), color=(255, 0, 0)).save(repository.corpus_image_paths()[0])
    settings = AppSettings(project_root=tmp_path)

    def fail_construction(*args, **kwargs):
        raise AssertionError("retriever must not be constructed when hash verification fails")

    monkeypatch.setattr(visual_baselines, "ColPaliRetriever", fail_construction)
    monkeypatch.setattr(visual_baselines, "VisRAGRetriever", fail_construction)
    with pytest.raises(DatasetUnavailableError, match="hash mismatch"):
        visual_baselines.build_visual_retriever("colpali", repository, settings)


@pytest.mark.parametrize("mode", ["colpali", "visrag"])
def test_both_retrievers_receive_verified_identities(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, mode: str
) -> None:
    repository = _hash_repository(tmp_path)
    captured: dict[str, object] = {}
    settings = AppSettings(project_root=tmp_path)

    def make_colpali(
        image_paths,
        model_name,
        revision,
        batch_size,
        *,
        score_batch_size,
        cache_dir,
        content_hashes=None,
    ):
        captured["content_hashes"] = content_hashes
        return object()

    def make_visrag(
        image_paths,
        model_name,
        revision,
        batch_size,
        *,
        cache_dir,
        content_hashes=None,
    ):
        captured["content_hashes"] = content_hashes
        return object()

    monkeypatch.setattr(visual_baselines, "ColPaliRetriever", make_colpali)
    monkeypatch.setattr(visual_baselines, "VisRAGRetriever", make_visrag)
    visual_baselines.build_visual_retriever(mode, repository, settings)
    expected = [[str(path), sha] for path, sha in repository.corpus_image_hashes()]
    assert captured["content_hashes"] == expected


def test_cuda_oom_halves_batch_size_and_recovers(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    paths = _image_paths(tmp_path, count=6)
    batch_sizes: list[int] = []
    forward_calls = 0

    def flaky_forward(model: _CountingModel, inputs) -> _FakeOutput:
        nonlocal forward_calls
        forward_calls += 1
        n = inputs.get("_n", 1)
        batch_sizes.append(n)
        if forward_calls == 1:
            raise torch.cuda.OutOfMemoryError("CUDA out of memory. Tried to allocate ...")
        base = torch.arange(n * model.seq * model.dim, dtype=torch.float32)
        return _FakeOutput(base.reshape(n, model.seq, model.dim))

    model = _CountingModel(forward=flaky_forward)
    _install_colpali_fakes(monkeypatch, model)

    retriever = visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=4)

    assert batch_sizes == [4, 2, 2, 2]
    assert retriever.image_embeddings.shape == (6, model.seq, model.dim)
    assert "CUDA OOM during ColPali embedding" in capsys.readouterr().out


def test_build_visual_retriever_points_cache_under_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeRepo:
        def corpus_image_paths(self):
            return [tmp_path / "a.png"]

    class FakeColPali:
        def __init__(
            self,
            image_paths: list[Path],
            model_name: str,
            revision: str | None,
            batch_size: int,
            *,
            score_batch_size: int,
            cache_dir: Path | None,
            content_hashes: list[list[str]] | None = None,
        ) -> None:
            captured["cache_dir"] = cache_dir

    monkeypatch.setattr(visual_baselines, "ColPaliRetriever", FakeColPali)
    settings = AppSettings(project_root=tmp_path)

    visual_baselines.build_visual_retriever("colpali", FakeRepo(), settings)

    assert captured["cache_dir"] == tmp_path / "cache" / "visual_embeddings"


def test_build_visual_retriever_no_cache_when_settings_lacks_project_root(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    captured: dict[str, object] = {}

    class FakeRepo:
        def corpus_image_paths(self):
            return [tmp_path / "a.png"]

    class FakeVisRAG:
        def __init__(
            self,
            image_paths: list[Path],
            model_name: str,
            revision: str | None,
            batch_size: int,
            *,
            cache_dir: Path | None,
            content_hashes: list[list[str]] | None = None,
        ) -> None:
            captured["cache_dir"] = cache_dir

    monkeypatch.setattr(visual_baselines, "VisRAGRetriever", FakeVisRAG)
    settings = SimpleNamespace(
        retrieval=SimpleNamespace(visrag_model="mock/visrag", visrag_revision=None, visual_batch_size=1)
    )

    visual_baselines.build_visual_retriever("visrag", FakeRepo(), settings)

    assert captured["cache_dir"] is None
