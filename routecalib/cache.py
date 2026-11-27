"""Bounded exact + trigram-cosine semantic cache with TTL staleness enforcement."""

from __future__ import annotations

import hashlib
import math
import re
import time
from collections import Counter, OrderedDict
from dataclasses import dataclass
from typing import Callable

from routecalib.schemas import ModelResponse


_NUM_TOKEN_RE = re.compile(r"\b\d+(?:\.\d+)?\b")


def _extract_numbers(norm_text: str) -> tuple[str, ...]:
    return tuple(_NUM_TOKEN_RE.findall(norm_text))


def _normalize_prompt(prompt: str) -> str:
    return " ".join(prompt.strip().lower().split())


def _trigram_vector(norm_text: str) -> tuple[dict[str, float], float]:
    padded = f"  {norm_text}  "
    if len(padded) < 3:
        return {}, 0.0
    counts = Counter(padded[i : i + 3] for i in range(len(padded) - 2))
    norm = math.sqrt(sum(v * v for v in counts.values()))
    if norm <= 0.0:
        return {}, 0.0
    return {k: float(v) for k, v in counts.items()}, norm


def _cosine_similarity(
    vec_a: dict[str, float], norm_a: float, vec_b: dict[str, float], norm_b: float
) -> float:
    if norm_a <= 0.0 or norm_b <= 0.0:
        return 0.0
    if len(vec_a) > len(vec_b):
        vec_a, vec_b = vec_b, vec_a
    dot = sum(v * vec_b.get(k, 0.0) for k, v in vec_a.items())
    return float(dot / (norm_a * norm_b))


@dataclass
class _CacheEntry:
    exact_key: str
    domain: str
    numbers: tuple[str, ...]
    trigrams: dict[str, float]
    norm: float
    response: ModelResponse
    expires_at_s: float


class BoundedSemanticCache:
    """LRU-bounded exact + character-trigram semantic cache with strict TTL expiry."""

    def __init__(
        self,
        *,
        max_entries: int = 512,
        ttl_seconds: float = 3600.0,
        similarity_threshold: float = 0.92,
        clock_fn: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_entries <= 0:
            raise ValueError("max_entries must be > 0.")
        if ttl_seconds <= 0.0:
            raise ValueError("ttl_seconds must be > 0.")
        if not (0.5 <= similarity_threshold <= 1.0):
            raise ValueError("similarity_threshold must be in [0.5, 1.0].")

        self.max_entries = int(max_entries)
        self.ttl_seconds = float(ttl_seconds)
        self.similarity_threshold = float(similarity_threshold)
        self._clock = clock_fn
        self._entries: OrderedDict[str, _CacheEntry] = OrderedDict()

        self.exact_hits: int = 0
        self.semantic_hits: int = 0
        self.misses: int = 0
        self.expired_evictions: int = 0
        self.lru_evictions: int = 0

    def _make_key(self, prompt: str, domain: str) -> tuple[str, str]:
        norm = _normalize_prompt(prompt)
        digest = hashlib.sha256(f"{domain}\0{norm}".encode("utf-8")).hexdigest()
        return digest, norm

    def _purge_expired(self, now: float) -> None:
        stale_keys = [k for k, entry in self._entries.items() if now >= entry.expires_at_s]
        for k in stale_keys:
            del self._entries[k]
            self.expired_evictions += 1

    def get(self, prompt: str, domain: str) -> tuple[ModelResponse | None, float]:
        """Look up exact match first, then domain-scoped trigram semantic match."""
        now = float(self._clock())
        self._purge_expired(now)

        key, norm = self._make_key(prompt, domain)
        if key in self._entries:
            entry = self._entries[key]
            self._entries.move_to_end(key)
            self.exact_hits += 1
            return entry.response, 1.0

        q_nums = _extract_numbers(norm)
        q_vec, q_norm = _trigram_vector(norm)
        best_key: str | None = None
        best_sim = 0.0
        for k, entry in self._entries.items():
            if entry.domain != domain or entry.numbers != q_nums:
                continue
            sim = _cosine_similarity(q_vec, q_norm, entry.trigrams, entry.norm)
            if sim >= self.similarity_threshold and sim > best_sim:
                best_sim = sim
                best_key = k

        if best_key is not None:
            entry = self._entries[best_key]
            self._entries.move_to_end(best_key)
            self.semantic_hits += 1
            return entry.response, round(best_sim, 6)

        self.misses += 1
        return None, 0.0

    def put(self, prompt: str, domain: str, response: ModelResponse) -> None:
        now = float(self._clock())
        self._purge_expired(now)

        key, norm = self._make_key(prompt, domain)
        trigrams, t_norm = _trigram_vector(norm)
        self._entries[key] = _CacheEntry(
            exact_key=key,
            domain=domain,
            numbers=_extract_numbers(norm),
            trigrams=trigrams,
            norm=t_norm,
            response=response,
            expires_at_s=now + self.ttl_seconds,
        )
        self._entries.move_to_end(key)

        while len(self._entries) > self.max_entries:
            self._entries.popitem(last=False)
            self.lru_evictions += 1

    def stats(self) -> dict[str, int | float]:
        total = self.exact_hits + self.semantic_hits + self.misses
        hit_rate = (self.exact_hits + self.semantic_hits) / total if total > 0 else 0.0
        return {
            "entries": len(self._entries),
            "exact_hits": self.exact_hits,
            "semantic_hits": self.semantic_hits,
            "misses": self.misses,
            "hit_rate": round(hit_rate, 4),
            "expired_evictions": self.expired_evictions,
            "lru_evictions": self.lru_evictions,
        }
