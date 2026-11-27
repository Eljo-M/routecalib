"""Sub-millisecond deterministic lexical & structural difficulty feature extractor."""

from __future__ import annotations

import math
import re
from collections import Counter

import numpy as np

from routecalib.schemas import DOMAINS

_WORD_RE = re.compile(r"[A-Za-z_][A-Za-z0-9_]*")
_NUMBER_RE = re.compile(r"\b\d+(?:\.\d+)?\b")
_MATH_SYM_RE = re.compile(r"(?:\s[+\-*/]\s|[=<>^%]|\b(?:prove|theorem|integral|derivative|matrix|eigen|modulo|sum|probability|variance|gradient|bound|solve)\b)", re.IGNORECASE)
_CODE_SYM_RE = re.compile(
    r"```|\b(?:def|class|return|import|async|await|struct|template|mutex|deadlock|O\(|complexity|refactor|debug|segfault|sql|regex|algorithm|pytest|unittest)\b",
    re.IGNORECASE,
)
_REASON_RE = re.compile(
    r"\b(?:compare|contrast|trade-?off|step by step|justify|counterexample|constraint|edge case|invariant|why does|analyze|evaluate|derive|multi-step)\b",
    re.IGNORECASE,
)
_EXTRACT_RE = re.compile(
    r"\b(?:extract|json|csv|table|list the|summarize|format as|parse|classify|headline|bullet)\b",
    re.IGNORECASE,
)
_CONSTRAINT_RE = re.compile(
    r"\b(?:must|never|only|exactly|at least|at most|without|strictly|no more than|ensure|require)\b",
    re.IGNORECASE,
)

FEATURE_NAMES: tuple[str, ...] = (
    "log_char_len",
    "log_token_est",
    "line_count_norm",
    "code_density",
    "math_density",
    "reasoning_density",
    "extract_density",
    "constraint_count_norm",
    "number_density",
    "punctuation_nesting",
    "char_entropy",
    "unique_word_ratio",
    "question_marks_norm",
    "dom_extraction",
    "dom_qa",
    "dom_reasoning",
    "dom_math",
    "dom_coding",
)


def estimate_tokens(text: str) -> int:
    """Fast deterministic token-count estimator (~4 chars/token floor)."""
    if not text:
        return 0
    return max(1, int(math.ceil(len(text) / 4.0)))


def infer_domain(prompt: str, explicit_domain: str = "auto") -> str:
    """Infer task domain deterministically when not explicitly supplied."""
    cleaned = (explicit_domain or "auto").strip().lower()
    if cleaned in DOMAINS:
        return cleaned

    code_hits = len(_CODE_SYM_RE.findall(prompt))
    math_hits = len(_MATH_SYM_RE.findall(prompt))
    reason_hits = len(_REASON_RE.findall(prompt))
    extract_hits = len(_EXTRACT_RE.findall(prompt))

    scores = {
        "coding": code_hits * 2.2,
        "math": math_hits * 2.0,
        "reasoning": reason_hits * 1.8,
        "extraction": extract_hits * 1.9,
        "qa": 1.0,
    }
    return max(scores.items(), key=lambda item: item[1])[0]


def _shannon_entropy(text: str) -> float:
    if not text:
        return 0.0
    counts = Counter(text)
    total = float(len(text))
    ent = 0.0
    for c in counts.values():
        p = c / total
        ent -= p * math.log2(p)
    return ent / 6.5  # normalized roughly to [0, 1]


def extract_features(prompt: str, explicit_domain: str = "auto") -> tuple[np.ndarray, str, dict[str, float]]:
    """Extract a fixed-size float64 feature vector, inferred domain, and summary dict."""
    domain = infer_domain(prompt, explicit_domain=explicit_domain)
    n_chars = max(len(prompt), 1)
    tok_est = estimate_tokens(prompt)
    words = _WORD_RE.findall(prompt.lower())
    n_words = max(len(words), 1)
    n_lines = prompt.count("\n") + 1

    code_hits = len(_CODE_SYM_RE.findall(prompt))
    math_hits = len(_MATH_SYM_RE.findall(prompt))
    reason_hits = len(_REASON_RE.findall(prompt))
    extract_hits = len(_EXTRACT_RE.findall(prompt))
    constraint_hits = len(_CONSTRAINT_RE.findall(prompt))
    num_hits = len(_NUMBER_RE.findall(prompt))
    nesting = sum(prompt.count(ch) for ch in "()[]{}")

    vec = np.array(
        [
            math.log1p(n_chars) / 8.0,
            math.log1p(tok_est) / 7.0,
            min(n_lines / 40.0, 2.0),
            min(code_hits / (n_words / 8.0 + 1.0), 3.0),
            min(math_hits / (n_words / 8.0 + 1.0), 3.0),
            min(reason_hits / (n_words / 10.0 + 1.0), 3.0),
            min(extract_hits / (n_words / 10.0 + 1.0), 3.0),
            min(constraint_hits / 6.0, 2.5),
            min(num_hits / (n_words / 6.0 + 1.0), 3.0),
            min(nesting / (n_chars / 15.0 + 1.0), 3.0),
            _shannon_entropy(prompt),
            len(set(words)) / float(n_words),
            min(prompt.count("?") / 4.0, 2.0),
            1.0 if domain == "extraction" else 0.0,
            1.0 if domain == "qa" else 0.0,
            1.0 if domain == "reasoning" else 0.0,
            1.0 if domain == "math" else 0.0,
            1.0 if domain == "coding" else 0.0,
        ],
        dtype=np.float64,
    )
    summary = {
        "token_est": float(tok_est),
        "code_hits": float(code_hits),
        "math_hits": float(math_hits),
        "reasoning_hits": float(reason_hits),
        "constraint_hits": float(constraint_hits),
        "char_entropy": round(float(vec[10]), 4),
    }
    return vec, domain, summary
