from __future__ import annotations

import hashlib
import io
import json
import math
import os
import tempfile
from datetime import UTC, datetime
from importlib import import_module
from pathlib import Path
from typing import Any, Protocol

import numpy as np

from .api_logging import is_valid_api_usage, vlm_cost_rates, zero_api_usage
from .asset_preparation import sha256_file as _stream_sha256
from .benchmarks import BenchmarkRepository
from .metrics import exact_match, token_f1
from .operations import ProgressReporter, check_termination
from .recovery import VisualFallback
from .resource_limits import (
    enforce_gpu_memory_fraction,
    enforce_memory_budget,
    is_fatal_resource_error,
    release_cuda_cache,
    select_dtype,
    torch_device,
)
from .run_io import atomic_write_json, run_fingerprint, safe_checkpoint_stem
from .settings import AppSettings


class VisualRetriever(Protocol):
    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]: ...


VISUAL_CACHE_SCHEMA_VERSION = 2

_HASH_PROGRESS_THRESHOLD = 256
_HASH_PROGRESS_INTERVAL = 128


def _compute_content_hashes(
    image_paths: list[Path],
    *,
    chunk_size: int = 1024 * 1024,
) -> list[list[str]]:
    """Deterministic ordered [(path, sha256)] identity for the visual corpus.

    Streams one image at a time so the whole corpus is never buffered in
    memory at once. Prints progress when the corpus is large enough that
    hashing is likely to take noticeable time.
    """
    total = len(image_paths)
    if total >= _HASH_PROGRESS_THRESHOLD:
        print(
            f"[faar] hashing {total} visual corpus images for cache identity...",
            flush=True,
        )
    hashes: list[list[str]] = []
    for index, path in enumerate(image_paths, start=1):
        hashes.append([str(path), _stream_sha256(path, chunk_size)])
        if total >= _HASH_PROGRESS_THRESHOLD and (
            index % _HASH_PROGRESS_INTERVAL == 0 or index == total
        ):
            print(f"[faar] hashed visual corpus {index}/{total}", flush=True)
    return hashes


def _ordered_content_hashes(
    image_paths: list[Path], recorded: list[tuple[Path, str]] | None
) -> list[list[str]]:
    """Ordered path->hash identity, preferring hashes recorded in the locked asset manifest."""
    if recorded is not None and len(recorded) == len(image_paths):
        return [[str(path), str(sha)] for path, sha in recorded]
    return _compute_content_hashes(image_paths)


def _visual_cache_key(
    schema_version: int,
    model_name: str,
    revision: str | None,
    dtype_name: str,
    image_paths: list[Path],
    content_hashes: list[list[str]],
) -> str:
    material = json.dumps(
        {
            "schema_version": schema_version,
            "model": model_name,
            "revision": revision,
            "dtype": dtype_name,
            "image_hashes": content_hashes,
        },
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(material.encode("utf-8")).hexdigest()


def _load_visual_cache(
    cache_dir: Path,
    key: str,
    schema_version: int,
    model_name: str,
    revision: str | None,
    dtype_name: str,
    image_paths: list[Path],
    content_hashes: list[list[str]],
) -> np.ndarray | None:
    try:
        meta = json.loads((cache_dir / f"{key}.meta.json").read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    expected_paths = [str(path) for path in image_paths]
    if not (
        isinstance(meta, dict)
        and meta.get("schema_version") == schema_version
        and meta.get("model") == model_name
        and meta.get("revision") == revision
        and meta.get("dtype") == dtype_name
        and meta.get("image_count") == len(image_paths)
        and meta.get("image_paths") == expected_paths
        and meta.get("image_hashes") == content_hashes
    ):
        return None
    try:
        with np.load(cache_dir / f"{key}.npz", allow_pickle=False) as data:
            if "embeddings" not in data:
                return None
            return data["embeddings"]
    except (OSError, ValueError):
        return None


def _atomic_write_bytes(path: Path, data: bytes) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent))
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(data)
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(tmp, path)
    except BaseException:
        try:
            os.unlink(tmp)
        except OSError:
            pass
        raise


def _save_visual_cache(
    cache_dir: Path,
    key: str,
    embeddings: np.ndarray,
    model_name: str,
    revision: str | None,
    dtype_name: str,
    image_paths: list[Path],
    content_hashes: list[list[str]],
) -> None:
    buffer = io.BytesIO()
    np.savez(buffer, embeddings=embeddings, allow_pickle=False)
    _atomic_write_bytes(cache_dir / f"{key}.npz", buffer.getvalue())
    atomic_write_json(
        cache_dir / f"{key}.meta.json",
        {
            "schema_version": VISUAL_CACHE_SCHEMA_VERSION,
            "model": model_name,
            "revision": revision,
            "dtype": dtype_name,
            "image_paths": [str(path) for path in image_paths],
            "image_hashes": content_hashes,
            "image_count": len(image_paths),
            "created_at": datetime.now(UTC).isoformat(),
        },
    )


def _with_oom_retry(torch, stage: str, fn, batch_size: int) -> tuple[Any, int]:
    while True:
        try:
            return fn(batch_size), batch_size
        except (torch.cuda.OutOfMemoryError, RuntimeError) as exc:
            if not isinstance(exc, torch.cuda.OutOfMemoryError):
                if "out of memory" not in str(exc).lower():
                    raise
            if batch_size <= 1:
                raise
            previous = batch_size
            batch_size = max(1, batch_size // 2)
            print(
                f"[faar] CUDA OOM during {stage}; halving batch size {previous}->{batch_size}",
                flush=True,
            )
            release_cuda_cache(torch)


def _validate_batch_shape(
    batch_shape: tuple[int, ...],
    expected_rows: int,
    reference: Any | None,
    *,
    what: str,
) -> None:
    if not batch_shape:
        raise ValueError(f"{what} batch produced a scalar; expected {expected_rows} rows")
    if batch_shape[0] != expected_rows:
        raise ValueError(f"{what} batch produced {batch_shape[0]} rows, expected {expected_rows}")
    if reference is not None and tuple(batch_shape[1:]) != tuple(reference.shape[1:]):
        raise ValueError(
            f"{what} batch produced per-row shape {tuple(batch_shape[1:])}, "
            f"expected {tuple(reference.shape[1:])}"
        )


def _validate_batch_storage(batch: Any, reference: Any | None, *, what: str) -> None:
    if reference is None:
        return
    if batch.dtype != reference.dtype:
        raise ValueError(
            f"{what} batch produced dtype {batch.dtype}, expected {reference.dtype}"
        )
    batch_device = getattr(batch, "device", None)
    reference_device = getattr(reference, "device", None)
    if batch_device != reference_device:
        raise ValueError(
            f"{what} batch produced device {batch_device}, expected {reference_device}"
        )


class ColPaliRetriever:
    def __init__(
        self,
        image_paths: list[Path],
        model_name: str,
        revision: str | None,
        batch_size: int = 4,
        score_batch_size: int = 32,
        cache_dir: Path | None = None,
        content_hashes: list[list[str]] | None = None,
    ) -> None:
        torch = import_module("torch")
        image_module = import_module("PIL.Image")
        transformers = import_module("transformers")
        model_class = transformers.ColPaliForRetrieval
        processor_class = transformers.ColPaliProcessor

        self.image_paths = list(image_paths)
        if not self.image_paths:
            raise ValueError("ColPali requires a non-empty visual corpus; found no image paths.")
        self.batch_size = max(1, batch_size)
        self.score_batch_size = max(1, score_batch_size)
        self.device = torch_device(torch)
        dtype = select_dtype(self.device, torch)
        enforce_gpu_memory_fraction(torch)
        self.model = (
            model_class.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
            )
            .to(self.device)
            .eval()
        )
        enforce_memory_budget("ColPali model load", torch)
        self.processor = processor_class.from_pretrained(model_name, revision=revision)
        image_embeddings = None
        cache_data = None
        cache_key = None
        if cache_dir is not None:
            if content_hashes is None or len(content_hashes) != len(self.image_paths):
                content_hashes = _compute_content_hashes(self.image_paths)
            cache_key = _visual_cache_key(
                VISUAL_CACHE_SCHEMA_VERSION,
                model_name,
                revision,
                str(dtype),
                self.image_paths,
                content_hashes,
            )
            cache_data = _load_visual_cache(
                cache_dir,
                cache_key,
                VISUAL_CACHE_SCHEMA_VERSION,
                model_name,
                revision,
                str(dtype),
                self.image_paths,
                content_hashes,
            )
        try:
            if cache_data is not None:
                self.image_embeddings = torch.from_numpy(cache_data)
            else:

                def encode_slice(start: int, batch_size: int) -> Any:
                    stop = min(start + batch_size, len(self.image_paths))
                    images = []
                    inputs = None
                    outputs = None
                    try:
                        for path in self.image_paths[start:stop]:
                            with image_module.open(path) as source:
                                images.append(source.convert("RGB"))
                        enforce_memory_budget("ColPali batch", torch)
                        inputs = self.processor(images=images).to(self.device)
                        with torch.inference_mode():
                            outputs = self.model(**inputs)
                        return outputs.embeddings.detach().cpu()
                    finally:
                        for image in images:
                            image.close()
                        del images, inputs, outputs

                start = 0
                while start < len(self.image_paths):
                    batch_embeddings, self.batch_size = _with_oom_retry(
                        torch,
                        "ColPali embedding",
                        lambda batch_size: encode_slice(start, batch_size),
                        self.batch_size,
                    )
                    stop = min(start + self.batch_size, len(self.image_paths))
                    _validate_batch_shape(
                        batch_embeddings.shape,
                        expected_rows=stop - start,
                        reference=image_embeddings,
                        what="ColPali embedding",
                    )
                    _validate_batch_storage(
                        batch_embeddings, image_embeddings, what="ColPali embedding"
                    )
                    if image_embeddings is None:
                        image_embeddings = torch.empty(
                            (len(self.image_paths), *batch_embeddings.shape[1:]),
                            dtype=batch_embeddings.dtype,
                            device=batch_embeddings.device,
                        )
                    image_embeddings[start:stop] = batch_embeddings
                    del batch_embeddings
                    start = stop
                self.image_embeddings = image_embeddings
                if cache_dir is not None:
                    _save_visual_cache(
                        cache_dir,
                        cache_key,
                        image_embeddings.numpy(),
                        model_name,
                        revision,
                        str(dtype),
                        self.image_paths,
                        content_hashes,
                    )
        finally:
            del image_embeddings, cache_data, cache_key
            release_cuda_cache(torch)

    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]:
        torch = import_module("torch")

        inputs = self.processor(text=[query]).to(self.device)
        query_embeddings = None
        batch_scores = None
        scores = None
        try:
            with torch.no_grad():
                query_embeddings = self.model(**inputs).embeddings.detach().cpu()

            def score_slice(start: int, batch_size: int, embeddings: Any) -> Any:
                stop = min(start + batch_size, len(self.image_paths))
                enforce_memory_budget("ColPali score batch", torch)
                return self.processor.score_retrieval(
                    embeddings,
                    self.image_embeddings[start:stop],
                )[0]

            start = 0
            while start < len(self.image_paths):
                batch_scores, self.score_batch_size = _with_oom_retry(
                    torch,
                    "ColPali score",
                    lambda batch_size: score_slice(
                        start, batch_size, query_embeddings
                    ),
                    self.score_batch_size,
                )
                stop = min(start + self.score_batch_size, len(self.image_paths))
                _validate_batch_shape(
                    batch_scores.shape,
                    expected_rows=stop - start,
                    reference=scores,
                    what="ColPali score",
                )
                _validate_batch_storage(batch_scores, scores, what="ColPali score")
                if scores is None:
                    scores = torch.empty(
                        (len(self.image_paths), *batch_scores.shape[1:]),
                        dtype=batch_scores.dtype,
                        device=batch_scores.device,
                    )
                scores[start:stop] = batch_scores
                start = stop
            indices = torch.topk(
                scores, k=min(top_k, len(self.image_paths))
            ).indices.tolist()
            return [(self.image_paths[index], float(scores[index])) for index in indices]
        finally:
            inputs = query_embeddings = batch_scores = scores = None
            release_cuda_cache(torch)


class VisRAGRetriever:
    INSTRUCTION = "Represent this query for retrieving relevant documents: "

    def __init__(
        self,
        image_paths: list[Path],
        model_name: str,
        revision: str | None,
        batch_size: int = 4,
        cache_dir: Path | None = None,
        content_hashes: list[list[str]] | None = None,
    ) -> None:
        torch = import_module("torch")
        image_module = import_module("PIL.Image")
        transformers = import_module("transformers")

        self.image_paths = list(image_paths)
        if not self.image_paths:
            raise ValueError("VisRAG requires a non-empty visual corpus; found no image paths.")
        self.batch_size = max(1, batch_size)
        self.device = torch_device(torch)
        dtype = select_dtype(self.device, torch)
        enforce_gpu_memory_fraction(torch)
        self.tokenizer = transformers.AutoTokenizer.from_pretrained(
            model_name,
            revision=revision,
            trust_remote_code=True,
        )
        self.model = (
            transformers.AutoModel.from_pretrained(
                model_name,
                revision=revision,
                torch_dtype=dtype,
                low_cpu_mem_usage=True,
                trust_remote_code=True,
            )
            .to(self.device)
            .eval()
        )
        enforce_memory_budget("VisRAG model load", torch)
        image_embeddings = None
        cache_data = None
        cache_key = None
        if cache_dir is not None:
            if content_hashes is None or len(content_hashes) != len(self.image_paths):
                content_hashes = _compute_content_hashes(self.image_paths)
            cache_key = _visual_cache_key(
                VISUAL_CACHE_SCHEMA_VERSION,
                model_name,
                revision,
                str(dtype),
                self.image_paths,
                content_hashes,
            )
            cache_data = _load_visual_cache(
                cache_dir,
                cache_key,
                VISUAL_CACHE_SCHEMA_VERSION,
                model_name,
                revision,
                str(dtype),
                self.image_paths,
                content_hashes,
            )
        try:
            if cache_data is not None:
                self.image_embeddings = cache_data
            else:

                def encode_slice(start: int, batch_size: int) -> np.ndarray:
                    stop = min(start + batch_size, len(self.image_paths))
                    images = []
                    try:
                        for path in self.image_paths[start:stop]:
                            with image_module.open(path) as source:
                                images.append(source.convert("RGB"))
                        enforce_memory_budget("VisRAG batch", torch)
                        return self._encode(images)
                    finally:
                        for image in images:
                            image.close()
                        del images

                start = 0
                while start < len(self.image_paths):
                    batch_embeddings, self.batch_size = _with_oom_retry(
                        torch,
                        "VisRAG embedding",
                        lambda batch_size: encode_slice(start, batch_size),
                        self.batch_size,
                    )
                    stop = min(start + self.batch_size, len(self.image_paths))
                    _validate_batch_shape(
                        batch_embeddings.shape,
                        expected_rows=stop - start,
                        reference=image_embeddings,
                        what="VisRAG embedding",
                    )
                    _validate_batch_storage(
                        batch_embeddings, image_embeddings, what="VisRAG embedding"
                    )
                    if image_embeddings is None:
                        image_embeddings = np.empty(
                            (len(self.image_paths), *batch_embeddings.shape[1:]),
                            dtype=batch_embeddings.dtype,
                        )
                    image_embeddings[start:stop] = batch_embeddings
                    del batch_embeddings
                    start = stop
                self.image_embeddings = image_embeddings
                if cache_dir is not None:
                    _save_visual_cache(
                        cache_dir,
                        cache_key,
                        image_embeddings,
                        model_name,
                        revision,
                        str(dtype),
                        self.image_paths,
                        content_hashes,
                    )
        finally:
            del image_embeddings, cache_data, cache_key
            release_cuda_cache(torch)

    def _encode(self, values: list[Any]) -> np.ndarray:
        torch = import_module("torch")
        functional = import_module("torch.nn.functional")

        is_text = isinstance(values[0], str)
        inputs = {
            "text": values if is_text else [""] * len(values),
            "image": [None] * len(values) if is_text else values,
            "tokenizer": self.tokenizer,
        }
        with torch.inference_mode():
            outputs = self.model(**inputs)
            mask = outputs.attention_mask
            weighted_mask = mask * mask.cumsum(dim=1)
            summed = torch.sum(outputs.last_hidden_state * weighted_mask.unsqueeze(-1).float(), dim=1)
            pooled = summed / weighted_mask.sum(dim=1, keepdim=True).float()
            return functional.normalize(pooled, p=2, dim=1).detach().cpu().numpy()

    def retrieve(self, query: str, top_k: int) -> list[tuple[Path, float]]:
        query_embedding = self._encode([self.INSTRUCTION + query])
        scores = (query_embedding @ self.image_embeddings.T)[0]
        indices = np.argsort(-scores, kind="stable")[:top_k]
        return [(self.image_paths[int(index)], float(scores[int(index)])) for index in indices]


def build_visual_retriever(mode: str, repo: BenchmarkRepository, settings: AppSettings) -> VisualRetriever:
    images = repo.corpus_image_paths()
    if not images:
        raise ValueError(f"{mode} requires a non-empty visual corpus; found no image paths.")
    cache_dir = None
    if hasattr(settings, "project_root"):
        cache_dir = Path(os.getenv("FAAR_CACHE_DIR", str(settings.project_root / "cache"))) / "visual_embeddings"
    recorded_hashes = getattr(repo, "corpus_image_hashes", lambda: None)()
    content_hashes = None
    if recorded_hashes is not None and len(recorded_hashes) == len(images):
        content_hashes = [[str(path), str(sha)] for path, sha in recorded_hashes]
    if mode == "colpali":
        return ColPaliRetriever(
            images,
            settings.retrieval.colpali_model,
            settings.retrieval.colpali_revision,
            settings.retrieval.visual_batch_size,
            score_batch_size=settings.retrieval.visual_score_batch_size,
            cache_dir=cache_dir,
            content_hashes=content_hashes,
        )
    if mode == "visrag":
        return VisRAGRetriever(
            images,
            settings.retrieval.visrag_model,
            settings.retrieval.visrag_revision,
            settings.retrieval.visual_batch_size,
            cache_dir=cache_dir,
            content_hashes=content_hashes,
        )
    raise ValueError(f"Unsupported visual baseline: {mode}")


def _ndcg_at_5(relevance: list[float], gold_size: int) -> float:
    ideal = sum(1.0 / math.log2(index + 2) for index in range(min(len(relevance), gold_size)))
    if ideal <= 0:
        return 0.0
    dcg = sum(rel / math.log2(index + 2) for index, rel in enumerate(relevance))
    return dcg / ideal


def run_visual_baseline(
    settings: AppSettings,
    repo: BenchmarkRepository,
    mode: str,
    max_examples: int | None = None,
    *,
    resume: bool = False,
    output_dir: Path | None = None,
) -> list[dict[str, Any]]:
    dataset = getattr(repo, "dataset", None)
    split = getattr(repo, "split", None)
    fingerprint = run_fingerprint(
        settings,
        profile=mode,
        dataset=dataset,
        split=split,
        manifest_sha256=getattr(repo, "manifest_sha256", None),
    )
    example_ids = repo.list_example_ids()
    if max_examples is not None:
        example_ids = example_ids[: max(0, max_examples)]
    base_output = output_dir or (settings.project_root / "logs/phase3" / mode)
    base_output.mkdir(parents=True, exist_ok=True)
    rows_by_id: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    if resume:
        for example_id in example_ids:
            cached = _load_visual_checkpoint(_visual_checkpoint_path(base_output, example_id))
            if cached is not None and _is_valid_visual_checkpoint(cached, example_id, mode, fingerprint):
                rows_by_id[example_id] = cached
            else:
                pending.append(example_id)
    else:
        pending = list(example_ids)
    retriever = None
    fallback = None
    if pending:
        retriever = build_visual_retriever(mode, repo, settings)
        fallback = VisualFallback(settings)
    if resume and rows_by_id:
        print(
            f"[faar] resume: reused {len(rows_by_id)}/{len(example_ids)} examples from checkpoints",
            flush=True,
        )
    page_map = repo.corpus_image_page_map()
    reporter = ProgressReporter(f"profile:{mode}", len(pending))
    failures: list[tuple[str, str]] = []
    processed = 0
    for example_id in pending:
        check_termination()
        enforce_memory_budget(f"profile:{mode} before example {example_id}")
        assert retriever is not None
        assert fallback is not None
        try:
            example = repo.get_example(example_id)
            retrieved = retriever.retrieve(example.question, settings.retrieval.top_k)
            image_paths = [path for path, _ in retrieved]
            gold_evidence = {(example.doc_name, page_id) for page_id in example.page_ids}
            top_relevance = [
                1.0 if page_map.get(path) in gold_evidence else 0.0 for path, _ in retrieved[:5]
            ]
            recall_at_5 = min(sum(top_relevance) / len(gold_evidence), 1.0) if gold_evidence else 0.0
            ndcg_at_5 = _ndcg_at_5(top_relevance, len(gold_evidence))
            answer_result = fallback.answer(example.question, image_paths, "")
            if answer_result.get("status") == "failed":
                raise RuntimeError(
                    f"example {example_id!r}: paid VLM call failed with reason "
                    f"{answer_result.get('reason', 'vlm_call_failed')!r}; "
                    "refusing to score or checkpoint a failed row"
                )
            prediction = str(answer_result.get("answer", ""))
            api_usage = answer_result.get("api_usage")
            if api_usage is None:
                raise ValueError(f"example {example_id!r} returned no API usage")
            elif not is_valid_api_usage(api_usage):
                raise ValueError(f"example {example_id!r} returned invalid API usage")
            row = {
                "profile": mode,
                "example_id": example_id,
                "question": example.question,
                "gold_answer": example.correct_answer,
                "predicted_answer": prediction,
                "failure_type": "visual_retrieval",
                "policy_action": "invoke_vlm",
                "request_model": answer_result.get("request_model"),
                "response_model": answer_result.get("response_model"),
                "completed_at_utc": answer_result.get("completed_at_utc"),
                "cost_rates": answer_result.get("cost_rates") or vlm_cost_rates(settings.recovery.vlm_backend),
                "api_usage": api_usage,
                "action_outcome": {
                    "action": "invoke_vlm",
                    "status": answer_result.get("status", "unknown"),
                    "reason": f"{mode}_retrieval_then_vlm",
                },
                "metrics": {
                    "ndcg@5": ndcg_at_5,
                    "recall@5": recall_at_5,
                    "em": exact_match(prediction, example.correct_answer),
                    "f1": token_f1(prediction, example.correct_answer),
                },
                "visual_hits": [
                    {"image_path": str(path), "score": round(score, 6)} for path, score in retrieved
                ],
                "source_assets": {"ocr_text_path": "", "image_paths": [str(path) for path in image_paths]},
                "run_metadata": {
                    "profile": mode,
                    "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                    "run_fingerprint": fingerprint,
                    "api_enabled": settings.recovery.api_enabled,
                    "vlm_backend": settings.recovery.vlm_backend,
                    "openai_model": settings.recovery.openai_model,
                    "vlm_model": settings.vlm_request_model(),
                    "cost_rates": vlm_cost_rates(settings.recovery.vlm_backend),
                    "evaluation_size": len(example_ids),
                    "selection": {"max_examples": max_examples},
                    "dataset": dataset,
                    "split": split,
                },
            }
        except Exception as exc:
            if is_fatal_resource_error(exc):
                raise
            reason = f"{type(exc).__name__}: {exc}"
            failed_row = {
                "profile": mode,
                "example_id": example_id,
                "action_outcome": {"action": "failed", "status": "failed", "reason": reason},
                "api_usage": zero_api_usage(),
                "error": reason,
                "run_metadata": {
                    "profile": mode,
                    "run_id": datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ"),
                    "run_fingerprint": fingerprint,
                    "evaluation_size": len(example_ids),
                    "selection": {"max_examples": max_examples},
                    "dataset": dataset,
                    "split": split,
                },
            }
            atomic_write_json(_visual_checkpoint_path(base_output, example_id), failed_row)
            failures.append((example_id, reason))
            print(f"[faar] example {example_id!r} FAILED and was not scored: {reason}", flush=True)
            processed += 1
            reporter.update(processed)
            continue
        rows_by_id[example_id] = row
        atomic_write_json(_visual_checkpoint_path(base_output, example_id), row)
        enforce_memory_budget(f"profile:{mode} after example {example_id}")
        processed += 1
        reporter.update(processed)
    if failures:
        ids = ", ".join(example_id for example_id, _ in failures)
        raise RuntimeError(
            f"{len(failures)} example(s) failed and were not scored: {ids}. "
            "Completed examples remain checkpointed; fix the cause and resume."
        )
    reporter.finish()
    return [rows_by_id[example_id] for example_id in example_ids if example_id in rows_by_id]


def _visual_checkpoint_path(base_output: Path, example_id: str) -> Path:
    return base_output / f"{safe_checkpoint_stem(example_id)}.json"


def _load_visual_checkpoint(path: Path) -> dict[str, Any] | None:
    try:
        row = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None
    return row if isinstance(row, dict) else None


def _is_valid_visual_checkpoint(
    row: dict[str, Any],
    example_id: str,
    mode: str,
    fingerprint: str,
) -> bool:
    metadata = row.get("run_metadata")
    metrics = row.get("metrics")
    if not isinstance(metadata, dict) or not isinstance(metrics, dict):
        return False
    metric_values = [metrics.get(name) for name in ("ndcg@5", "recall@5", "em", "f1")]
    if any(
        isinstance(value, bool)
        or not isinstance(value, (int, float))
        or not math.isfinite(float(value))
        for value in metric_values
    ):
        return False
    return (
        row.get("example_id") == example_id
        and row.get("profile") == mode
        and metadata.get("run_fingerprint") == fingerprint
        and isinstance(row.get("question"), str)
        and isinstance(row.get("gold_answer"), str)
        and isinstance(row.get("predicted_answer"), str)
        and isinstance(row.get("action_outcome"), dict)
        and row["action_outcome"].get("status") != "failed"
        and isinstance(row.get("visual_hits"), list)
        and isinstance(row.get("source_assets"), dict)
        and is_valid_api_usage(row.get("api_usage"))
    )
