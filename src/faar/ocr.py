from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from transformers import AutoModelForImageTextToText, AutoProcessor

from .resource_limits import (
    enforce_gpu_memory_fraction,
    enforce_memory_budget,
    release_cuda_cache,
    select_dtype,
    torch_device,
)

GOT_OCR_MODEL = "stepfun-ai/GOT-OCR-2.0-hf"


@lru_cache(maxsize=1)
def _load_got_ocr(model_name: str, revision: str | None):
    import torch

    processor = AutoProcessor.from_pretrained(model_name, revision=revision, use_fast=True)
    device = torch_device(torch)
    dtype = select_dtype(device, torch)
    enforce_gpu_memory_fraction(torch)
    model = AutoModelForImageTextToText.from_pretrained(
        model_name,
        revision=revision,
        torch_dtype=dtype,
        low_cpu_mem_usage=True,
    ).to(device)
    return model.eval(), processor


def extract_got_ocr(
    image_path: Path,
    model_name: str = GOT_OCR_MODEL,
    revision: str | None = None,
) -> str:
    if not image_path.exists():
        raise FileNotFoundError(f"GOT-OCR input image is missing: {image_path}")
    import torch

    model, processor = _load_got_ocr(model_name, revision)
    enforce_memory_budget("GOT-OCR inference")
    inputs = processor(str(image_path), return_tensors="pt").to(model.device)
    generated = None
    try:
        with torch.inference_mode():
            generated = model.generate(
                **inputs,
                do_sample=False,
                tokenizer=processor.tokenizer,
                stop_strings="<|im_end|>",
                max_new_tokens=4096,
            )
        prompt_length = inputs["input_ids"].shape[1]
        return processor.decode(generated[0, prompt_length:], skip_special_tokens=True).strip()
    finally:
        if generated is not None:
            del generated
        del inputs
        release_cuda_cache(torch)
