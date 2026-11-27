"""Pluggable model providers: OpenAI-compatible HTTP client and deterministic local provider."""

from __future__ import annotations

import hashlib
import json
import os
import time
import urllib.error
import urllib.request
from typing import Any, Callable, Mapping, Protocol

from routecalib.features import estimate_tokens
from routecalib.schemas import ModelResponse, ModelTier


class ModelProvider(Protocol):
    """Protocol implemented by all live and deterministic model backends."""

    def invoke(self, tier: ModelTier, prompt: str, *, max_tokens: int = 256) -> ModelResponse:
        ...


class OpenAICompatibleProvider:
    """Zero-dependency HTTP client for any OpenAI-compatible /v1/chat/completions endpoint.

    Works with local servers (Ollama, vLLM, llama.cpp server) or hosted endpoints via
    `tier.endpoint_url` and an optional environment variable API key.
    """

    def __init__(
        self,
        *,
        api_key_env: str = "ROUTECALIB_API_KEY",
        timeout_secs: float = 30.0,
    ) -> None:
        self.api_key_env = api_key_env
        self.timeout_secs = float(timeout_secs)

    def invoke(self, tier: ModelTier, prompt: str, *, max_tokens: int = 256) -> ModelResponse:
        if not tier.endpoint_url:
            raise ValueError(f"Tier '{tier.tier_id}' has no endpoint_url configured for live HTTP invocation.")
        url = tier.endpoint_url.rstrip("/")
        if not url.endswith("/chat/completions"):
            url = f"{url}/v1/chat/completions"

        payload = {
            "model": tier.model_name or tier.tier_id,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": int(max_tokens),
            "temperature": 0.0,
        }
        headers = {"Content-Type": "application/json"}
        api_key = os.environ.get(self.api_key_env, "").strip()
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"

        req = urllib.request.Request(
            url,
            data=json.dumps(payload).encode("utf-8"),
            headers=headers,
            method="POST",
        )
        t0 = time.perf_counter()
        try:
            with urllib.request.urlopen(req, timeout=self.timeout_secs) as resp:
                body = json.loads(resp.read().decode("utf-8"))
        except urllib.error.URLError as exc:
            raise RuntimeError(f"HTTP invocation failed for tier '{tier.tier_id}' at {url}: {exc}") from exc
        elapsed_ms = (time.perf_counter() - t0) * 1000.0

        choices = body.get("choices", [])
        text = ""
        if choices:
            msg = choices[0].get("message", {})
            text = str(msg.get("content", ""))
        usage = body.get("usage", {})
        in_tok = int(usage.get("prompt_tokens", estimate_tokens(prompt)))
        out_tok = int(usage.get("completion_tokens", estimate_tokens(text)))
        cost = tier.estimate_cost_usd(in_tok, out_tok)
        return ModelResponse(
            tier_id=tier.tier_id,
            text=text,
            input_tokens=in_tok,
            output_tokens=out_tok,
            latency_ms=round(elapsed_ms, 3),
            cost_usd=round(cost, 8),
            quality_passed=None,
            metadata={"endpoint_url": url, "model": tier.model_name or tier.tier_id},
        )


class DeterministicLocalProvider:
    """Deterministic offline provider for reproducible local demos, testing, and benchmarks."""

    def __init__(
        self,
        *,
        outcome_lookup: Mapping[tuple[str, str], tuple[bool, int, float]] | None = None,
        response_formatter: Callable[[ModelTier, str], str] | None = None,
    ) -> None:
        # Optional map: (prompt_sha256, tier_id) -> (quality_passed, output_tokens, latency_ms)
        self._lookup = dict(outcome_lookup) if outcome_lookup is not None else {}
        self._formatter = response_formatter

    @staticmethod
    def prompt_key(prompt: str) -> str:
        return hashlib.sha256(prompt.encode("utf-8")).hexdigest()

    def invoke(self, tier: ModelTier, prompt: str, *, max_tokens: int = 256) -> ModelResponse:
        in_tok = estimate_tokens(prompt)
        pkey = self.prompt_key(prompt)
        if (pkey, tier.tier_id) in self._lookup:
            passed, out_tok, lat_ms = self._lookup[(pkey, tier.tier_id)]
        else:
            passed = True
            out_tok = min(max_tokens, max(32, int(in_tok * 0.8)))
            lat_ms = tier.estimate_latency_ms(out_tok)

        text = (
            self._formatter(tier, prompt)
            if self._formatter is not None
            else f"[{tier.display_name}] Verified deterministic response for prompt ({in_tok} input tokens)."
        )
        cost = tier.estimate_cost_usd(in_tok, out_tok)
        return ModelResponse(
            tier_id=tier.tier_id,
            text=text,
            input_tokens=in_tok,
            output_tokens=int(out_tok),
            latency_ms=round(float(lat_ms), 3),
            cost_usd=round(float(cost), 8),
            quality_passed=bool(passed),
            metadata={"provider": "deterministic_local"},
        )
