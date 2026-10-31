"""Render every page of every ohr_dev_v1 document from the locked PDF archive.

Two renders per page: 150 DPI PNG for the human inspection packet (the same
DPI as the frozen pilot packet) and a model-input JPEG for the gold-page image
headroom condition. Output goes under the analysis directory's ``pages/``,
which is git-ignored; ``render_record.json`` records every file hash so the
renders can be checked after regeneration.

    .local/venv-prebaseline/bin/python scripts/analysis/ohr_dev_v1_failures/render_pages.py
"""

from __future__ import annotations

import hashlib
import io
import json
import zipfile

import pypdfium2
from common import OUT_DIR, PDF_ARCHIVE, PILOT_DIR, read_json

INSPECTION_DPI = 150
# Model input: long side capped at 2048 px, the largest size OpenAI's high-detail path keeps before tiling.
MODEL_LONG_SIDE = 2048
MODEL_JPEG_QUALITY = 90


def safe_name(doc_id: str, page_idx: int) -> str:
    return f"{hashlib.sha256(doc_id.encode()).hexdigest()[:12]}_p{page_idx}"


def main() -> None:
    runtime = read_json(PILOT_DIR / "runtime_manifest.json")
    pages_dir = OUT_DIR / "pages"
    pages_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    with zipfile.ZipFile(PDF_ARCHIVE) as archive:
        for doc in runtime["documents"]:
            data = archive.read(doc["pdf"]["member"])
            if hashlib.sha256(data).hexdigest() != doc["pdf"]["sha256"]:
                raise SystemExit(f"{doc['doc_id']}: archive member hash differs from the runtime manifest")
            pdf = pypdfium2.PdfDocument(data)
            for page in doc["pages"]:
                idx = page["page_idx"]
                name = safe_name(doc["doc_id"], idx)
                image = pdf[idx].render(scale=INSPECTION_DPI / 72).to_pil()
                png = io.BytesIO()
                image.save(png, format="PNG", optimize=True)
                (pages_dir / f"{name}.png").write_bytes(png.getvalue())
                scale = min(1.0, MODEL_LONG_SIDE / max(image.size))
                model_image = image.convert("RGB")
                if scale < 1.0:
                    model_image = model_image.resize((round(image.width * scale), round(image.height * scale)))
                jpg = io.BytesIO()
                model_image.save(jpg, format="JPEG", quality=MODEL_JPEG_QUALITY)
                (pages_dir / f"{name}.model.jpg").write_bytes(jpg.getvalue())
                rows.append(
                    {
                        "doc_id": doc["doc_id"],
                        "page_idx": idx,
                        "png": f"pages/{name}.png",
                        "png_sha256": hashlib.sha256(png.getvalue()).hexdigest(),
                        "png_size": list(image.size),
                        "model_jpg": f"pages/{name}.model.jpg",
                        "model_jpg_sha256": hashlib.sha256(jpg.getvalue()).hexdigest(),
                        "model_jpg_size": list(model_image.size),
                    }
                )
            pdf.close()
    record = {
        "renderer": {"package": "pypdfium2", "version": pypdfium2.V_PYPDFIUM2},
        "inspection_dpi": INSPECTION_DPI,
        "model_image": {"long_side_max": MODEL_LONG_SIDE, "format": "JPEG", "quality": MODEL_JPEG_QUALITY, "source_dpi": INSPECTION_DPI},
        "pages": rows,
    }
    (OUT_DIR / "render_record.json").write_text(json.dumps(record, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"rendered {len(rows)} pages into {pages_dir}")


if __name__ == "__main__":
    main()
