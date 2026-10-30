"""
Thin wrapper around the Google Gemini API.

Key-optional: with no GEMINI_API_KEY (or GOOGLE_API_KEY) set, `call_gemini`
raises LLMUnavailable and the caller falls back to the rule engine. Every other
failure raises LLMError. The caller is responsible for the wall-clock timeout.
"""

from __future__ import annotations

import os

try:
    from dotenv import load_dotenv

    load_dotenv(
        os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env")
    )
except Exception:  # noqa: BLE001 - dotenv is a convenience, not a requirement
    pass

# A "-lite" flash model: ~1s round trip, fast enough for an interactive demo.
DEFAULT_MODEL = os.environ.get("GEMINI_MODEL", "gemini-flash-lite-latest")
# Tried in order if the primary model 404s (deprecated) or 503s (overloaded).
FALLBACK_MODELS = [
    m.strip() for m in os.environ.get(
        "GEMINI_FALLBACK_MODELS", "gemini-3.5-flash-lite,gemini-3.1-flash-lite"
    ).split(",") if m.strip()
]


class LLMUnavailable(RuntimeError):
    """No API key configured, or the SDK is not installed."""


class LLMError(RuntimeError):
    """The API call failed or returned nothing usable."""


def _api_key() -> str | None:
    return os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")


def is_configured() -> bool:
    # bool(), not "is not None": an empty GEMINI_API_KEY= is not a key (/health said "configured" for it)
    return bool(_api_key())


def call_gemini(prompt: str, model: str = DEFAULT_MODEL,
                temperature: float = 0.1, max_output_tokens: int = 600) -> str:
    key = _api_key()
    if not key:
        raise LLMUnavailable("no GEMINI_API_KEY / GOOGLE_API_KEY set")

    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:  # pragma: no cover
        raise LLMUnavailable(f"google-genai not installed: {exc}") from exc

    client = genai.Client(api_key=key)
    config = types.GenerateContentConfig(
        temperature=temperature,
        max_output_tokens=max_output_tokens,
        response_mime_type="application/json",
    )

    candidates = [model] + [m for m in FALLBACK_MODELS if m != model]
    last_exc: Exception | None = None
    for candidate in candidates:
        try:
            resp = client.models.generate_content(
                model=candidate, contents=prompt, config=config
            )
        except Exception as exc:  # noqa: BLE001 - network, auth, quota, 404, 503
            last_exc = exc
            continue
        text = (getattr(resp, "text", None) or "").strip()
        if text:
            return text
        last_exc = LLMError(f"{candidate} returned an empty response")

    raise LLMError(f"Gemini call failed (tried {candidates}): {last_exc}")
