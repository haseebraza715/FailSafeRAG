from __future__ import annotations

import json
import math
import os
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any


@dataclass
class ApiCallRecord:
    created_at_utc: str
    provider: str
    model: str
    operation: str
    status: str
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    metadata: dict[str, Any] | None = None


class ApiCallLogger:
    def __init__(self, log_path: Path, enabled: bool = True) -> None:
        self.log_path = log_path
        self.enabled = enabled
        self.call_count = 0
        self.total_cost_usd = 0.0

    def log(self, record: ApiCallRecord) -> None:
        if record.status == "started":
            self.call_count += 1
        self.total_cost_usd += record.cost_usd
        if not self.enabled:
            return
        self.log_path.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(record)
        with self.log_path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload, sort_keys=True) + "\n")
        _try_wandb_log(payload)


def make_vlm_logger(project_root: Path, enabled: bool) -> ApiCallLogger:
    return ApiCallLogger(project_root / "logs/vlm_calls.jsonl", enabled=enabled)


def new_record(
    *,
    provider: str,
    model: str,
    operation: str,
    status: str,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
    metadata: dict[str, Any] | None = None,
) -> ApiCallRecord:
    return ApiCallRecord(
        created_at_utc=datetime.now(UTC).isoformat(),
        provider=provider,
        model=model,
        operation=operation,
        status=status,
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost_usd=round(cost_usd, 8),
        metadata=metadata or {},
    )


ANTHROPIC_VLM_BACKENDS = {"claude-sonnet-4-5", "anthropic", "claude"}


def anthropic_cost_rates() -> dict[str, float | str]:
    return {
        "provider": "anthropic",
        "currency": "USD",
        "input_usd_per_million_tokens": float(os.getenv("ANTHROPIC_INPUT_USD_PER_MTOK", "3.0")),
        "output_usd_per_million_tokens": float(os.getenv("ANTHROPIC_OUTPUT_USD_PER_MTOK", "15.0")),
    }


def vlm_cost_rates(vlm_backend: str) -> dict[str, float | str] | None:
    if vlm_backend == "openai":
        return openai_cost_rates()
    if vlm_backend in ANTHROPIC_VLM_BACKENDS:
        return anthropic_cost_rates()
    return None


def openai_cost_rates() -> dict[str, float | str]:
    return {
        "provider": "openai",
        "currency": "USD",
        "input_usd_per_million_tokens": float(os.getenv("OPENAI_INPUT_USD_PER_MTOK", "2.50")),
        "output_usd_per_million_tokens": float(os.getenv("OPENAI_OUTPUT_USD_PER_MTOK", "10.00")),
    }


def _cost_usd_from_rates(
    prompt_tokens: int,
    completion_tokens: int,
    rates: dict[str, float | str],
) -> float:
    return (
        prompt_tokens / 1_000_000 * float(rates["input_usd_per_million_tokens"])
        + completion_tokens / 1_000_000 * float(rates["output_usd_per_million_tokens"])
    )


def estimate_anthropic_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    return _cost_usd_from_rates(prompt_tokens, completion_tokens, anthropic_cost_rates())


def estimate_openai_cost_usd(
    prompt_tokens: int,
    completion_tokens: int,
    rates: dict[str, float | str] | None = None,
) -> float:
    rates = rates or openai_cost_rates()
    return _cost_usd_from_rates(prompt_tokens, completion_tokens, rates)


def make_api_usage(
    api_requests: int = 0,
    prompt_tokens: int = 0,
    completion_tokens: int = 0,
    cost_usd: float = 0.0,
) -> dict[str, int | float]:
    return {
        "api_requests": int(api_requests),
        "prompt_tokens": int(prompt_tokens),
        "completion_tokens": int(completion_tokens),
        "cost_usd": round(float(cost_usd), 8),
    }


def zero_api_usage() -> dict[str, int | float]:
    return make_api_usage()


def is_valid_api_usage(value: Any) -> bool:
    if not isinstance(value, dict):
        return False
    for key in ("api_requests", "prompt_tokens", "completion_tokens"):
        count = value.get(key)
        if isinstance(count, bool) or not isinstance(count, int) or count < 0:
            return False
    cost = value.get("cost_usd")
    return (
        not isinstance(cost, bool)
        and isinstance(cost, (int, float))
        and math.isfinite(float(cost))
        and float(cost) >= 0
    )


def _try_wandb_log(payload: dict[str, Any]) -> None:
    if not os.getenv("WANDB_PROJECT"):
        return
    try:  # pragma: no cover - optional external tracking
        import wandb

        if wandb.run is None:
            wandb.init(project=os.getenv("WANDB_PROJECT"), mode=os.getenv("WANDB_MODE", "offline"))
        wandb.log(
            {
                "vlm/call_count": 1 if payload.get("status") == "started" else 0,
                "vlm/cost_usd": payload.get("cost_usd", 0.0),
                "vlm/prompt_tokens": payload.get("prompt_tokens", 0),
                "vlm/completion_tokens": payload.get("completion_tokens", 0),
            }
        )
    except Exception:
        return
