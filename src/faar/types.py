from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Literal


FailureType = Literal["semantic", "word_level", "structural"]
PolicyAction = Literal["answer_direct", "correct_text", "retry_retrieval", "invoke_vlm"]


@dataclass
class Phase0Example:
    example_id: str
    doc_name: str
    question: str
    correct_answer: str
    page_ids: list[int]
    ocr_text: str
    ocr_text_path: Path
    gt_text_path: Path | None
    image_paths: list[Path]
    metadata: dict[str, Any]


@dataclass
class Chunk:
    chunk_id: str
    example_id: str
    doc_name: str
    page_id: int
    text: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass
class RetrievalHit:
    chunk: Chunk
    bm25_score: float
    dense_score: float
    fused_score: float

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["chunk"] = self.chunk.to_dict()
        return payload

