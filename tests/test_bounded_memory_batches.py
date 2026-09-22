from __future__ import annotations

from io import StringIO
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from faar import ocr, resource_limits, retrieval, visual_baselines
from faar.settings import AppSettings, RetrievalSettings
from faar.types import Chunk


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


class _FakeEmbeddingModel:
    def __init__(self, seq: int = 4, dim: int = 8) -> None:
        self.seq = seq
        self.dim = dim

    def __call__(self, **inputs):
        n = inputs.get("_n", 1)
        base = torch.arange(n * self.seq * self.dim, dtype=torch.float32)
        return _FakeOutput(base.reshape(n, self.seq, self.dim))

    def to(self, device):
        return self

    def eval(self):
        return self


class _FakeProcessor:
    def __init__(self) -> None:
        self.score_batches: list[int] = []

    def __call__(self, **inputs):
        images = inputs.get("images")
        return _FakeBatch({"_n": len(images) if images is not None else 1})

    def to(self, device):
        return self

    def score_retrieval(self, query_embeddings, passage_embeddings):
        count = passage_embeddings.shape[0]
        self.score_batches.append(count)
        query = query_embeddings[0]
        dots = torch.einsum("qd,nsd->nqs", query, passage_embeddings)
        scores = dots.max(dim=2).values.sum(dim=1)
        return scores.unsqueeze(0)


class _FakeRepository:
    def __init__(self, paths: list[Path]) -> None:
        self.paths = paths

    def corpus_image_paths(self):
        return self.paths


def test_embed_batch_size_env_override(monkeypatch) -> None:
    monkeypatch.setenv("FAAR_EMBED_BATCH_SIZE", "4")
    monkeypatch.setenv("FAAR_VISUAL_SCORE_BATCH_SIZE", "8")
    settings = AppSettings(project_root=Path.cwd())
    assert settings.retrieval.embed_batch_size == 4
    assert settings.retrieval.visual_score_batch_size == 8


def test_batch_size_settings_defaults_conservative(monkeypatch) -> None:
    monkeypatch.delenv("FAAR_EMBED_BATCH_SIZE", raising=False)
    monkeypatch.delenv("FAAR_VISUAL_SCORE_BATCH_SIZE", raising=False)
    settings = AppSettings(project_root=Path.cwd())
    assert settings.retrieval.embed_batch_size == 2
    assert settings.retrieval.visual_score_batch_size == 8


@pytest.mark.parametrize(
    ("name", "value"),
    [
        ("FAAR_EMBED_BATCH_SIZE", "0"),
        ("FAAR_EMBED_BATCH_SIZE", "-3"),
        ("FAAR_EMBED_BATCH_SIZE", "abc"),
        ("FAAR_VISUAL_SCORE_BATCH_SIZE", "0"),
        ("FAAR_VISUAL_SCORE_BATCH_SIZE", "1.5"),
    ],
)
def test_batch_size_settings_reject_invalid(monkeypatch, name: str, value: str) -> None:
    monkeypatch.setenv(name, value)
    with pytest.raises(ValueError, match="positive integer"):
        AppSettings(project_root=Path.cwd()).retrieval.embed_batch_size
    with pytest.raises(ValueError, match="positive integer"):
        AppSettings(project_root=Path.cwd()).retrieval.visual_score_batch_size


def test_rss_bytes_from_proc_status_parses_vmrss() -> None:
    payload = "Name:\tpython\nVmRSS:\t204800 kB\nVmHWM:\t409600 kB\n"
    assert resource_limits._rss_bytes_from_proc_status(StringIO(payload)) == 204800 * 1024


def test_rss_bytes_from_proc_status_missing_field_is_none() -> None:
    assert resource_limits._rss_bytes_from_proc_status(StringIO("Name:\tpython\n")) is None
    assert resource_limits._rss_bytes_from_proc_status(StringIO("VmRSS:\t0 kB\n")) is None


def test_current_rss_falls_back_when_proc_is_unavailable(monkeypatch) -> None:
    def unavailable(*args, **kwargs):
        raise OSError("missing")

    monkeypatch.setattr("builtins.open", unavailable)
    monkeypatch.setattr(resource_limits, "_peak_rss_bytes", lambda: 1234)
    assert resource_limits.current_rss_bytes() == 1234


def test_enforce_memory_budget_reports_current_and_limit(monkeypatch) -> None:
    monkeypatch.setattr(resource_limits, "current_rss_bytes", lambda: 6 * 1024**3)
    monkeypatch.setenv("FAAR_MAX_RSS_GB", "5")
    with pytest.raises(MemoryError) as exc:
        resource_limits.enforce_memory_budget("scoring")
    message = str(exc.value)
    assert "current RSS is 6.00 GiB" in message
    assert "limit is 5.00 GiB" in message


def test_enforce_memory_budget_passes_under_limit(monkeypatch) -> None:
    monkeypatch.setattr(resource_limits, "current_rss_bytes", lambda: 1 * 1024**3)
    monkeypatch.setenv("FAAR_MAX_RSS_GB", "5")
    resource_limits.enforce_memory_budget("scoring")


def test_release_cuda_cache_syncs_then_empties() -> None:
    calls: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            calls.append("synchronize")

        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

    resource_limits.release_cuda_cache(SimpleNamespace(cuda=FakeCuda))
    assert calls == ["synchronize", "empty_cache"]


def test_release_cuda_cache_noop_without_cuda() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def synchronize() -> None:
            raise AssertionError("must not synchronize without CUDA")

        @staticmethod
        def empty_cache() -> None:
            raise AssertionError("must not empty cache without CUDA")

    resource_limits.release_cuda_cache(SimpleNamespace(cuda=FakeCuda))


def test_release_cuda_cache_reports_synchronize_failure_without_active_error() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            raise RuntimeError("driver error")

        @staticmethod
        def empty_cache() -> None:
            raise AssertionError("empty_cache must not run after synchronize failed")

    with pytest.raises(RuntimeError, match="driver error"):
        resource_limits.release_cuda_cache(SimpleNamespace(cuda=FakeCuda))


def test_release_cuda_cache_reports_empty_cache_failure_without_active_error() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            return None

        @staticmethod
        def empty_cache() -> None:
            raise RuntimeError("OOM while emptying cache")

    with pytest.raises(RuntimeError, match="OOM while emptying cache"):
        resource_limits.release_cuda_cache(SimpleNamespace(cuda=FakeCuda))


def test_release_cuda_cache_best_effort_does_not_mask_inference_error() -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            raise RuntimeError("sync failure")

        @staticmethod
        def empty_cache() -> None:
            return None

    with pytest.raises(RuntimeError, match="inference boom"):
        try:
            raise RuntimeError("inference boom")
        except RuntimeError:
            resource_limits.release_cuda_cache(SimpleNamespace(cuda=FakeCuda))
            raise


def test_gpu_reserve_releases_cache_before_mem_get_info(monkeypatch) -> None:
    calls: list[str] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def synchronize() -> None:
            calls.append("synchronize")

        @staticmethod
        def empty_cache() -> None:
            calls.append("empty_cache")

        @staticmethod
        def mem_get_info():
            calls.append("mem_get_info")
            return (10 * 1024**3, 20 * 1024**3)

    monkeypatch.setenv("FAAR_MIN_GPU_FREE_GB", "5")
    resource_limits.enforce_memory_budget("scoring", SimpleNamespace(cuda=FakeCuda))
    assert calls == ["synchronize", "empty_cache", "mem_get_info"]


class _FakeDtypes:
    bfloat16 = "bf16"
    float16 = "fp16"
    float32 = "fp32"


class _FakeCuda:
    def __init__(self, supports_bf16: bool) -> None:
        self._supports_bf16 = supports_bf16

    def is_bf16_supported(self) -> bool:
        return self._supports_bf16


def _fake_torch(cuda_available: bool = False, mps_available: bool = False) -> SimpleNamespace:
    class Backends:
        class mps:
            @staticmethod
            def is_available() -> bool:
                return mps_available

    return SimpleNamespace(cuda=SimpleNamespace(is_available=lambda: cuda_available), backends=Backends)


@pytest.mark.parametrize(
    ("device_type", "cuda_bf16", "expected"),
    [
        ("cuda", True, "bf16"),
        ("cuda", False, "fp16"),
        ("mps", False, "fp16"),
        ("cpu", False, "fp32"),
    ],
)
def test_select_dtype_policy(device_type: str, cuda_bf16: bool, expected: str) -> None:
    torch_module = _fake_torch()
    torch_module.cuda = _FakeCuda(cuda_bf16)
    torch_module._dtypes = _FakeDtypes()
    torch_module.bfloat16 = "bf16"
    torch_module.float16 = "fp16"
    torch_module.float32 = "fp32"
    device = SimpleNamespace(type=device_type)
    assert resource_limits.select_dtype(device, torch_module) == expected


def test_torch_device_prefers_cuda_then_mps_then_cpu() -> None:
    class Device:
        def __init__(self, name: str) -> None:
            self.name = name

        def __eq__(self, other):
            return isinstance(other, Device) and self.name == other.name

    def torch_with(cuda: bool, mps: bool) -> SimpleNamespace:
        class Backends:
            class mps:
                @staticmethod
                def is_available() -> bool:
                    return mps

        return SimpleNamespace(
            cuda=SimpleNamespace(is_available=lambda: cuda),
            backends=Backends,
            device=Device,
        )

    assert resource_limits.torch_device(torch_with(True, True)) == Device("cuda:0")
    assert resource_limits.torch_device(torch_with(False, True)) == Device("mps")
    assert resource_limits.torch_device(torch_with(False, False)) == Device("cpu")


def test_got_ocr_loader_uses_single_device_and_no_device_map(monkeypatch) -> None:
    calls: list[dict] = []

    class FakeLoader:
        @classmethod
        def from_pretrained(cls, model, **kwargs):
            calls.append({"model": model, **kwargs})

            class Chain:
                def to(self, device):
                    self.device = device
                    return self

                def eval(self):
                    return self

            return Chain()

    monkeypatch.setattr(ocr, "AutoProcessor", FakeLoader)
    monkeypatch.setattr(ocr, "AutoModelForImageTextToText", FakeLoader)
    monkeypatch.setattr(ocr, "torch_device", lambda torch_module: SimpleNamespace(type="cpu"))
    monkeypatch.setattr(ocr, "select_dtype", lambda device, torch_module: torch.float32)
    ocr._load_got_ocr.cache_clear()

    ocr._load_got_ocr("stepfun-ai/GOT-OCR-2.0-hf", None)

    model_call = next(call for call in calls if "low_cpu_mem_usage" in call)
    assert model_call["low_cpu_mem_usage"] is True
    assert model_call["torch_dtype"] == torch.float32
    assert "device_map" not in model_call


def test_got_ocr_extract_releases_temporary_tensors(monkeypatch, tmp_path: Path) -> None:
    import torch

    image = tmp_path / "page.png"
    image.write_bytes(b"image")
    released: list[str] = []

    class FakeProcessor:
        tokenizer = object()

        def __call__(self, path, return_tensors=None):
            return _FakeBatch({"_n": 1, "input_ids": torch.ones(1, 2, dtype=torch.long)})

        def decode(self, token_ids, skip_special_tokens=False):
            return " OCR TEXT "

    class FakeModel:
        device = torch.device("cpu")

        def generate(self, **kwargs):
            return torch.ones(1, 7, dtype=torch.long)

    monkeypatch.setattr(ocr, "_load_got_ocr", lambda model, revision: (FakeModel(), FakeProcessor()))
    monkeypatch.setattr(ocr, "release_cuda_cache", lambda torch_module: released.append("cache"))

    result = ocr.extract_got_ocr(image)

    assert result == "OCR TEXT"
    assert released == ["cache"]


def test_colpali_scores_corpus_in_bounded_batches(monkeypatch, tmp_path: Path) -> None:
    paths = _image_paths(tmp_path, count=6)
    processor = _FakeProcessor()
    model = _FakeEmbeddingModel()
    released: list[str] = []

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return model

    class FakeProcessorClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return processor

    monkeypatch.setattr(visual_baselines, "torch_device", lambda torch_module: torch.device("cpu"))
    monkeypatch.setattr(visual_baselines, "select_dtype", lambda device, torch_module: torch.float32)
    monkeypatch.setattr(visual_baselines, "release_cuda_cache", lambda torch_module: released.append("cache"))
    monkeypatch.setattr("transformers.ColPaliForRetrieval", FakeModelClass)
    monkeypatch.setattr("transformers.ColPaliProcessor", FakeProcessorClass)

    retriever = visual_baselines.ColPaliRetriever(
        paths, "mock/colpali", None, batch_size=3, score_batch_size=2
    )

    assert retriever.image_embeddings.shape[0] == 6
    assert "cache" in released
    released.clear()

    hits = retriever.retrieve("some question", top_k=3)
    chunked = [(path, score) for path, score in hits]

    assert processor.score_batches == [2, 2, 2]
    assert all(count <= 2 for count in processor.score_batches)
    assert len(chunked) == 3
    assert "cache" in released

    query_embeddings = model(**processor(text=["some question"])).embeddings.detach().cpu()
    full_scores = processor.score_retrieval(query_embeddings, retriever.image_embeddings)[0]
    assert processor.score_batches == [2, 2, 2, 6]
    indices = torch.topk(full_scores, k=3).indices.tolist()
    unchunked = [(paths[index], float(full_scores[index])) for index in indices]

    assert chunked == unchunked


def test_colpali_embeddings_never_cat_and_preserve_order(monkeypatch, tmp_path: Path) -> None:
    paths = _image_paths(tmp_path, count=5)
    released: list[str] = []

    class OrderedModel:
        def __init__(self, seq: int = 4, dim: int = 8) -> None:
            self.seq = seq
            self.dim = dim
            self.encoded = 0

        def __call__(self, **inputs):
            n = inputs.get("_n", 1)
            start = self.encoded * self.seq * self.dim
            stop = start + n * self.seq * self.dim
            self.encoded += n
            return _FakeOutput(torch.arange(start, stop, dtype=torch.float32).reshape(n, self.seq, self.dim))

        def to(self, device):
            return self

        def eval(self):
            return self

    model = OrderedModel()

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return model

    class FakeProcessorClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return _FakeProcessor()

    def fail_cat(*args, **kwargs):
        raise AssertionError("torch.cat must not be used for bounded batching")

    monkeypatch.setattr(visual_baselines, "torch_device", lambda torch_module: torch.device("cpu"))
    monkeypatch.setattr(visual_baselines, "select_dtype", lambda device, torch_module: torch.float32)
    monkeypatch.setattr(visual_baselines, "release_cuda_cache", lambda torch_module: released.append("cache"))
    monkeypatch.setattr("transformers.ColPaliForRetrieval", FakeModelClass)
    monkeypatch.setattr("transformers.ColPaliProcessor", FakeProcessorClass)
    monkeypatch.setattr(torch, "cat", fail_cat)

    retriever = visual_baselines.ColPaliRetriever(paths, "mock/colpali", None, batch_size=2)

    assert model.encoded == 5
    assert retriever.image_embeddings.shape == (5, model.seq, model.dim)
    expected = torch.arange(5 * model.seq * model.dim, dtype=torch.float32).reshape(5, model.seq, model.dim)
    assert torch.equal(retriever.image_embeddings, expected)
    assert "cache" in released


def test_colpali_scores_never_cat_and_preserve_order(monkeypatch, tmp_path: Path) -> None:
    paths = _image_paths(tmp_path, count=5)
    processor = _FakeProcessor()
    model = _FakeEmbeddingModel()
    released: list[str] = []

    class FakeModelClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return model

    class FakeProcessorClass:
        @classmethod
        def from_pretrained(cls, *args, **kwargs):
            return processor

    def fail_cat(*args, **kwargs):
        raise AssertionError("torch.cat must not be used for bounded batching")

    monkeypatch.setattr(visual_baselines, "torch_device", lambda torch_module: torch.device("cpu"))
    monkeypatch.setattr(visual_baselines, "select_dtype", lambda device, torch_module: torch.float32)
    monkeypatch.setattr(visual_baselines, "release_cuda_cache", lambda torch_module: released.append("cache"))
    monkeypatch.setattr("transformers.ColPaliForRetrieval", FakeModelClass)
    monkeypatch.setattr("transformers.ColPaliProcessor", FakeProcessorClass)
    monkeypatch.setattr(torch, "cat", fail_cat)

    retriever = visual_baselines.ColPaliRetriever(
        paths, "mock/colpali", None, batch_size=2, score_batch_size=2
    )
    released.clear()

    hits = retriever.retrieve("some question", top_k=5)
    chunked = [(path, score) for path, score in hits]

    assert processor.score_batches == [2, 2, 1]
    assert all(count <= 2 for count in processor.score_batches)
    assert "cache" in released

    query_embeddings = model(**processor(text=["some question"])).embeddings.detach().cpu()
    full_scores = processor.score_retrieval(query_embeddings, retriever.image_embeddings)[0]
    indices = torch.topk(full_scores, k=5).indices.tolist()
    unchunked = [(paths[index], float(full_scores[index])) for index in indices]

    assert chunked == unchunked


def test_visrag_embeddings_never_concatenate_and_preserve_order(monkeypatch, tmp_path: Path) -> None:
    paths = _image_paths(tmp_path, count=5)
    dim = 8
    released: list[str] = []
    encode_calls: list[tuple[int, int]] = []

    def fake_encode(self, values: list) -> np.ndarray:
        rows = len(values)
        offset = sum(size for _, size in encode_calls)
        encode_calls.append((offset, rows))
        return np.arange(offset * dim, (offset + rows) * dim, dtype=np.float32).reshape(rows, dim)

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

    def fail_concatenate(*args, **kwargs):
        raise AssertionError("np.concatenate must not be used for bounded batching")

    monkeypatch.setattr(visual_baselines, "torch_device", lambda torch_module: torch.device("cpu"))
    monkeypatch.setattr(visual_baselines, "select_dtype", lambda device, torch_module: torch.float32)
    monkeypatch.setattr(visual_baselines, "release_cuda_cache", lambda torch_module: released.append("cache"))
    monkeypatch.setattr("transformers.AutoTokenizer", FakeTokenizerClass)
    monkeypatch.setattr("transformers.AutoModel", FakeModelClass)
    monkeypatch.setattr(visual_baselines.VisRAGRetriever, "_encode", fake_encode)
    monkeypatch.setattr(np, "concatenate", fail_concatenate)

    retriever = visual_baselines.VisRAGRetriever(paths, "mock/visrag", None, batch_size=2)

    assert encode_calls == [(0, 2), (2, 2), (4, 1)]
    assert retriever.image_embeddings.shape == (5, dim)
    expected = np.arange(5 * dim, dtype=np.float32).reshape(5, dim)
    assert np.array_equal(retriever.image_embeddings, expected)
    assert "cache" in released


def test_visual_batch_shape_validation_rejects_scalar_and_mismatch() -> None:
    with pytest.raises(ValueError, match="produced a scalar"):
        visual_baselines._validate_batch_shape((), 1, None, what="embedding")
    with pytest.raises(ValueError, match="produced 2 rows, expected 1"):
        visual_baselines._validate_batch_shape((2, 4), 1, None, what="embedding")
    with pytest.raises(ValueError, match="per-row shape"):
        visual_baselines._validate_batch_shape(
            (1, 5),
            1,
            np.empty((2, 4), dtype=np.float32),
            what="embedding",
        )


def test_visual_batch_storage_validation_rejects_dtype_mismatch() -> None:
    with pytest.raises(ValueError, match="produced dtype"):
        visual_baselines._validate_batch_storage(
            np.empty((1, 4), dtype=np.float64),
            np.empty((2, 4), dtype=np.float32),
            what="embedding",
        )


def test_colpali_empty_corpus_fails_clearly(monkeypatch) -> None:
    monkeypatch.setattr(visual_baselines, "torch_device", lambda torch_module: "cpu")
    with pytest.raises(ValueError, match="non-empty visual corpus"):
        visual_baselines.ColPaliRetriever([], "mock/colpali", None)


def test_build_visual_retriever_empty_corpus_fails_clearly(monkeypatch, tmp_path: Path) -> None:
    settings = AppSettings(project_root=tmp_path)
    with pytest.raises(ValueError, match="non-empty visual corpus"):
        visual_baselines.build_visual_retriever("colpali", _FakeRepository([]), settings)
    with pytest.raises(ValueError, match="non-empty visual corpus"):
        visual_baselines.build_visual_retriever("visrag", _FakeRepository([]), settings)


def test_optional_fraction_validates_open_interval(monkeypatch) -> None:
    monkeypatch.delenv("FAAR_MAX_GPU_MEMORY_FRACTION", raising=False)
    assert resource_limits._optional_fraction("FAAR_MAX_GPU_MEMORY_FRACTION") is None
    monkeypatch.setenv("FAAR_MAX_GPU_MEMORY_FRACTION", "0.5")
    assert resource_limits._optional_fraction("FAAR_MAX_GPU_MEMORY_FRACTION") == 0.5
    for bad in ("0", "1.5", "-0.1", "abc"):
        monkeypatch.setenv("FAAR_MAX_GPU_MEMORY_FRACTION", bad)
        with pytest.raises(ValueError, match="positive fraction in \\(0, 1\\]"):
            resource_limits._optional_fraction("FAAR_MAX_GPU_MEMORY_FRACTION")
    monkeypatch.setenv("FAAR_MAX_GPU_MEMORY_FRACTION", " ")
    assert resource_limits._optional_fraction("FAAR_MAX_GPU_MEMORY_FRACTION") is None


def test_enforce_gpu_memory_fraction_sets_once_with_mock(monkeypatch) -> None:
    calls: list[tuple[float, int]] = []

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def set_per_process_memory_fraction(fraction: float, device: int) -> None:
            calls.append((fraction, device))

    resource_limits._GPU_MEMORY_FRACTION_APPLIED = False
    monkeypatch.setenv("FAAR_MAX_GPU_MEMORY_FRACTION", "0.6")
    resource_limits.enforce_gpu_memory_fraction(SimpleNamespace(cuda=FakeCuda))
    resource_limits.enforce_gpu_memory_fraction(SimpleNamespace(cuda=FakeCuda))
    assert calls == [(0.6, 0)]


def test_enforce_gpu_memory_fraction_noop_without_env(monkeypatch) -> None:
    monkeypatch.delenv("FAAR_MAX_GPU_MEMORY_FRACTION", raising=False)
    resource_limits._GPU_MEMORY_FRACTION_APPLIED = False

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

        @staticmethod
        def set_per_process_memory_fraction(fraction, device) -> None:
            raise AssertionError("must not cap without a configured fraction")

    resource_limits.enforce_gpu_memory_fraction(SimpleNamespace(cuda=FakeCuda))


def test_enforce_gpu_memory_fraction_noop_without_cuda(monkeypatch) -> None:
    monkeypatch.setenv("FAAR_MAX_GPU_MEMORY_FRACTION", "0.5")
    resource_limits._GPU_MEMORY_FRACTION_APPLIED = False

    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return False

        @staticmethod
        def set_per_process_memory_fraction(fraction, device) -> None:
            raise AssertionError("must not cap without CUDA")

    resource_limits.enforce_gpu_memory_fraction(SimpleNamespace(cuda=FakeCuda))


def test_enforce_gpu_memory_fraction_raises_when_api_unavailable(monkeypatch) -> None:
    class FakeCuda:
        @staticmethod
        def is_available() -> bool:
            return True

    resource_limits._GPU_MEMORY_FRACTION_APPLIED = False
    monkeypatch.setenv("FAAR_MAX_GPU_MEMORY_FRACTION", "0.5")
    with pytest.raises(RuntimeError, match="set_per_process_memory_fraction is unavailable"):
        resource_limits.enforce_gpu_memory_fraction(SimpleNamespace(cuda=FakeCuda))


def test_retrieval_model_loads_call_gpu_fraction_guard(monkeypatch) -> None:
    calls: list[str] = []
    monkeypatch.setattr(
        retrieval,
        "enforce_gpu_memory_fraction",
        lambda torch_module: calls.append("fraction"),
    )
    monkeypatch.setattr(retrieval, "torch_device", lambda torch_module: SimpleNamespace(type="cpu"))
    monkeypatch.setattr(retrieval, "select_dtype", lambda device, torch_module: "fp32")
    monkeypatch.setattr(retrieval, "SentenceTransformer", lambda *args, **kwargs: object())
    monkeypatch.setattr(retrieval, "CrossEncoder", lambda *args, **kwargs: object())
    retrieval._load_embedding_model.cache_clear()
    retrieval._load_reranker.cache_clear()
    retrieval._load_embedding_model("mock-embed", None)
    retrieval._load_reranker("mock-rerank", None)
    assert calls == ["fraction", "fraction"]
    retrieval._load_embedding_model.cache_clear()
    retrieval._load_reranker.cache_clear()


def test_hybrid_retriever_drops_corpus_embedding_array(monkeypatch, tmp_path: Path) -> None:
    chunks = [
        Chunk(chunk_id="c0", example_id="e0", doc_name="doc", page_id=1, text="the quick brown fox"),
        Chunk(chunk_id="c1", example_id="e0", doc_name="doc", page_id=2, text="jumps over the lazy dog"),
    ]
    encode_calls: list[tuple[list[str], dict]] = []

    class FakeEmbedder:
        def encode(self, sentences, **kwargs):
            encode_calls.append((sentences, kwargs))
            return np.random.default_rng(0).normal(size=(len(sentences), 8))

    class FakeReranker:
        def predict(self, pairs, **kwargs):
            return [0.5] * len(pairs)

    monkeypatch.setattr(retrieval, "_load_embedding_model", lambda *args: FakeEmbedder())
    monkeypatch.setattr(retrieval, "_load_reranker", lambda *args: FakeReranker())
    monkeypatch.setattr(retrieval, "SentenceTransformer", lambda *args, **kwargs: object())
    monkeypatch.setattr(retrieval, "CrossEncoder", lambda *args, **kwargs: object())

    settings = RetrievalSettings()
    settings.embed_batch_size = 4
    retriever = retrieval.HybridRetriever(chunks, settings)

    assert retriever._dense_index.ntotal == 2
    assert not hasattr(retriever, "_corpus_embeddings")
    assert encode_calls[0][1]["batch_size"] == 4

    hits = retriever.retrieve("fox", top_k=2)
    assert len(hits) == 2
    assert encode_calls[1][1]["batch_size"] == 4
