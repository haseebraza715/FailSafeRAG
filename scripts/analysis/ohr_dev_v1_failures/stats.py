"""Paired comparisons for the headroom conditions. Pure functions over per-question score lists.

- ``cluster_bootstrap_diff``: paired bootstrap of mean(condition) - mean(baseline), resampling whole documents
  (study brief section 8: questions sharing a document are not independent). The question-level version is
  reported next to it for reference.
- ``mcnemar_exact``: two-sided exact McNemar test on paired binary outcomes (EM).
- ``mcnemar_sample_size``: questions needed to detect a paired EM gain with McNemar's test (Connor 1987 normal
  approximation), for a given discordant-pair rate, optionally inflated by a document design effect.
- ``icc_oneway``: one-way ANOVA intraclass correlation of a per-question score within documents.
- ``cohens_kappa``: agreement of two label lists.
"""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Sequence

import numpy as np
from scipy.stats import binomtest, norm

BOOTSTRAP_SEED = 20261031
BOOTSTRAP_RESAMPLES = 10_000


def cluster_bootstrap_diff(
    clusters: Sequence[str], base: Sequence[float], cond: Sequence[float], *, by_cluster: bool = True,
    resamples: int = BOOTSTRAP_RESAMPLES, seed: int = BOOTSTRAP_SEED,
) -> dict[str, float]:
    base_a, cond_a = np.asarray(base, float), np.asarray(cond, float)
    diff = cond_a - base_a
    rng = np.random.default_rng(seed)
    if by_cluster:
        labels = sorted(set(clusters))
        index = {c: i for i, c in enumerate(labels)}
        ids = np.array([index[c] for c in clusters])
        sums = np.bincount(ids, weights=diff, minlength=len(labels))
        counts = np.bincount(ids, minlength=len(labels)).astype(float)
        draws = rng.integers(0, len(labels), size=(resamples, len(labels)))
        stats = sums[draws].sum(axis=1) / counts[draws].sum(axis=1)
    else:
        draws = rng.integers(0, len(diff), size=(resamples, len(diff)))
        stats = diff[draws].mean(axis=1)
    low, high = np.percentile(stats, [2.5, 97.5])
    return {"diff": float(diff.mean()), "ci_low": float(low), "ci_high": float(high)}


def mcnemar_exact(base: Sequence[int], cond: Sequence[int]) -> dict[str, float | int]:
    gained = sum(1 for b, c in zip(base, cond) if b == 0 and c == 1)
    lost = sum(1 for b, c in zip(base, cond) if b == 1 and c == 0)
    n = gained + lost
    p = 1.0 if n == 0 else binomtest(min(gained, lost), n, 0.5, alternative="two-sided").pvalue
    return {"gained": gained, "lost": lost, "discordant": n, "p_value": float(p)}


def mcnemar_sample_size(delta: float, discordant_rate: float, *, alpha: float = 0.05, power: float = 0.80,
                        design_effect: float = 1.0) -> int | None:
    """Questions needed so McNemar detects a paired gain ``delta`` = p(gain) - p(loss), given p(gain) + p(loss)."""
    if discordant_rate <= abs(delta) or discordant_rate <= 0:
        return None
    za, zb = norm.ppf(1 - alpha / 2), norm.ppf(power)
    n = (za * math.sqrt(discordant_rate) + zb * math.sqrt(discordant_rate - delta**2)) ** 2 / delta**2
    return math.ceil(n * design_effect)


def icc_oneway(clusters: Sequence[str], values: Sequence[float]) -> dict[str, float]:
    groups: dict[str, list[float]] = {}
    for c, v in zip(clusters, values):
        groups.setdefault(c, []).append(float(v))
    k, n = len(groups), len(values)
    grand = sum(values) / n
    ssb = sum(len(g) * (sum(g) / len(g) - grand) ** 2 for g in groups.values())
    ssw = sum(sum((x - sum(g) / len(g)) ** 2 for x in g) for g in groups.values())
    msb, msw = ssb / (k - 1), ssw / (n - k)
    m0 = (n - sum(len(g) ** 2 for g in groups.values()) / n) / (k - 1)
    icc = (msb - msw) / (msb + (m0 - 1) * msw) if (msb + (m0 - 1) * msw) > 0 else 0.0
    mean_size = n / k
    return {"icc": icc, "mean_cluster_size": mean_size, "design_effect": 1 + (mean_size - 1) * max(icc, 0.0)}


def cohens_kappa(first: Sequence[str], second: Sequence[str]) -> dict[str, float | int]:
    if len(first) != len(second) or not first:
        raise ValueError("two non-empty label lists of equal length are needed")
    n = len(first)
    observed = sum(a == b for a, b in zip(first, second)) / n
    ca, cb = Counter(first), Counter(second)
    expected = sum(ca[label] * cb[label] for label in set(ca) | set(cb)) / n**2
    kappa = 1.0 if expected == 1.0 else (observed - expected) / (1 - expected)
    return {"n": n, "observed_agreement": observed, "expected_agreement": expected, "kappa": kappa}
