from __future__ import annotations

import math
import re
from collections import Counter


def normalize_text(text: str) -> str:
    lowered = text.lower().strip()
    cleaned = re.sub(r"[^a-z0-9\s]", " ", lowered)
    return re.sub(r"\s+", " ", cleaned).strip()


def exact_match(prediction: str, gold: str) -> float:
    return 1.0 if normalize_text(prediction) == normalize_text(gold) else 0.0


def token_f1(prediction: str, gold: str) -> float:
    pred_tokens = normalize_text(prediction).split()
    gold_tokens = normalize_text(gold).split()
    if not pred_tokens and not gold_tokens:
        return 1.0
    if not pred_tokens or not gold_tokens:
        return 0.0
    common = Counter(pred_tokens) & Counter(gold_tokens)
    overlap = sum(common.values())
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(gold_tokens)
    return (2 * precision * recall) / (precision + recall)


def recall_at_k(hit_texts: list[str], gold: str, k: int = 5) -> float:
    gold_norm = normalize_text(gold)
    for text in hit_texts[:k]:
        if gold_norm and gold_norm in normalize_text(text):
            return 1.0
    return 0.0


def ndcg_at_k(hit_texts: list[str], gold: str, k: int = 5) -> float:
    gold_norm = normalize_text(gold)
    relevances = [1.0 if gold_norm and gold_norm in normalize_text(text) else 0.0 for text in hit_texts[:k]]
    dcg = 0.0
    for idx, rel in enumerate(relevances, start=1):
        dcg += (2**rel - 1) / math.log2(idx + 1)
    ideal = sorted(relevances, reverse=True)
    idcg = 0.0
    for idx, rel in enumerate(ideal, start=1):
        idcg += (2**rel - 1) / math.log2(idx + 1)
    if idcg == 0:
        return 0.0
    return dcg / idcg

