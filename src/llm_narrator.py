"""
LLM Recommendation Narrator.

This is the one genuine LLM call in the prototype: takes the optimizer's
ACTUAL computed numbers (never raw data, never anything the LLM could
embellish) and asks it to write the plain-English "why" a marketer would
read in the approval view. The LLM explains; it does not decide -- the
optimizer already made the decision deterministically before this module
is ever called.

PROVIDER CHAIN (tries each in order, falls through on failure):
  1. Local open-source model via Ollama (default: llama3.2) -- no API key,
     no cost, no third-party data exposure, no external uptime dependency.
     This is the DEFAULT and recommended path for this prototype.
  2. Anthropic API (Claude) -- used only if ANTHROPIC_API_KEY is set AND
     Ollama isn't reachable. Optional, not required.
  3. Deterministic template -- zero hallucination risk by construction,
     fires if neither LLM backend is available. The approval view never
     depends on any LLM being available to function.

HALLUCINATION CONTROL (applies regardless of which backend generated the
text -- the requirements explicitly ask how you'd control this):
  1. Constrained context: the prompt contains ONLY the numbers being
     explained -- no broader dataset access, no tool use, nothing it could
     misremember or invent additional context from.
  2. Explicit instruction: told directly not to introduce any figure not
     present in the provided data.
  3. Post-hoc groundedness check: every numeric token in the output is
     extracted and checked against the numbers actually present in the
     input. Ungrounded numbers get the response rejected, not silently
     shown -- a failed check falls back to the deterministic template
     rather than risking a fabricated figure reaching a marketer.

Why Ollama by default: for a system touching ad-spend and customer data,
keeping inference local avoids sending any of it to a third party, costs
nothing per call, and removes the live demo's dependency on an external
API being up. Install: https://ollama.com (single installer, no GPU
required). Pull a small model once with `ollama pull llama3.2`.
"""
import os
import re
from dataclasses import dataclass

ANTHROPIC_MODEL = "claude-sonnet-5-5"
OLLAMA_MODEL = os.environ.get("OLLAMA_MODEL", "llama3.2")
OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://localhost:11434")


@dataclass
class NarrativeResult:
    text: str
    provider: str  # "ollama", "anthropic", or "template"
    used_llm: bool
    groundedness_check_passed: bool = True
    fallback_reason: str = ""


_NUMBER_RE = re.compile(r"\d+(?:\.\d+)?")


def _extract_numbers(text: str) -> list[tuple[float, int]]:
    """Pull every numeric token out of a string as (value, decimal places).

    Handles $, %, x-suffixed multipliers ('3.5x') and thousands separators.
    Signs are dropped on purpose: a model correctly writing "a 13% decrease"
    for a change of -13.0% must not be flagged. The decimal count is kept so
    each number can be checked at the precision it was written with."""
    cleaned = re.sub(r"(?<=\d),(?=\d{3})", "", text)
    out = []
    for tok in _NUMBER_RE.findall(cleaned):
        decimals = len(tok.split(".")[1]) if "." in tok else 0
        out.append((float(tok), decimals))
    return out


def _build_allowed_numbers(row: dict, summary: dict) -> set[float]:
    """Every numeric input the model was given, unrounded and unsigned."""
    allowed = set()
    for v in list(row.values()) + list(summary.values()):
        if isinstance(v, (int, float)) and not isinstance(v, bool):
            allowed.add(abs(float(v)))
    # the project's own cited targets (4.0x blended ROAS, $45 NC-CPA) -- NOT
    # generic round numbers, which would let fabricated values slip through
    allowed.update({4.0, 45.0})
    return allowed


def _derived_numbers(row: dict) -> set[float]:
    """Figures a model can correctly derive from two given inputs -- the
    dollar change in daily spend -- which models state often."""
    try:
        return {abs(float(row["recommended_daily_spend"]) - float(row["current_daily_spend"]))}
    except (KeyError, TypeError, ValueError):
        return set()


def _is_grounded(value: float, decimals: int, sources: set[float]) -> bool:
    """Precision-aware match: a number written with d decimal places is
    grounded only if some source value rounds to exactly that number at d
    places. "2.94x" must match an input that rounds to 2.94; "3x" may be a
    rounding of 3.03; but "99.9" can never pass as a rounding of 100."""
    return any(abs(round(s, decimals) - value) < 1e-9 for s in sources)


def _ungrounded_numbers(text: str, row: dict, summary: dict) -> list[float]:
    sources = _build_allowed_numbers(row, summary) | _derived_numbers(row)
    return [v for v, d in _extract_numbers(text) if not _is_grounded(v, d, sources)]


def _template_narrative(row: dict, summary: dict) -> str:
    direction = "increase" if row["change_pct"] > 0 else "decrease" if row["change_pct"] < 0 else "hold"
    flag = " This channel is showing diminishing returns." if row.get("diminishing_returns_flag") else ""
    return (
        f"Recommend a {abs(row['change_pct']):.1f}% {direction} in {row['channel']} daily spend "
        f"(${row['current_daily_spend']:,.0f} → ${row['recommended_daily_spend']:,.0f}). "
        f"Current ROAS is {row['current_roas']:.2f}x against a projected {row['projected_roas']:.2f}x "
        f"after the shift, based on marginal ROAS of {row['current_marginal_roas']:.2f}x at current spend "
        f"versus {row['projected_marginal_roas']:.2f}x after.{flag}"
    )


def _build_prompt(row: dict, summary: dict) -> str:
    return f"""You are writing a one-paragraph explanation for a marketer reviewing a budget reallocation recommendation. Use ONLY the numbers given below. Do not introduce any number, percentage, or dollar figure that is not explicitly listed here.

Channel: {row['channel']}
Current daily spend: ${row['current_daily_spend']:,.0f}
Recommended daily spend: ${row['recommended_daily_spend']:,.0f}
Change: {row['change_pct']:.1f}%
Current ROAS: {row['current_roas']:.2f}x
Projected ROAS after change: {row['projected_roas']:.2f}x
Current marginal ROAS: {row['current_marginal_roas']:.2f}x
Projected marginal ROAS after change: {row['projected_marginal_roas']:.2f}x
Diminishing returns flagged: {row.get('diminishing_returns_flag', False)}
Blended ROAS across all channels (current): {summary['current_blended_roas']:.2f}x
Blended ROAS across all channels (projected): {summary['projected_blended_roas']:.2f}x

Write 2-3 sentences a marketer would read before approving or rejecting this change. Plain language, no jargon, no invented context about seasonality or campaigns not mentioned above."""


def _try_ollama(prompt: str) -> str | None:
    try:
        import ollama
        response = ollama.chat(
            model=OLLAMA_MODEL,
            messages=[{"role": "user", "content": prompt}],
            options={"num_predict": 200},
        )
        return response["message"]["content"].strip()
    except Exception:
        return None  # not installed, not running, model not pulled, etc. -- fall through silently


def _try_anthropic(prompt: str) -> str | None:
    api_key = os.environ.get("ANTHROPIC_API_KEY")
    if not api_key:
        return None
    try:
        import anthropic
        client = anthropic.Anthropic(api_key=api_key)
        response = client.messages.create(
            model=ANTHROPIC_MODEL, max_tokens=300,
            # Short factual paragraph -- no reasoning needed. Turning thinking off keeps
            # the 300-token cap for the answer itself and keeps cost low.
            thinking={"type": "between_tools"},
            messages=[{"role": "user", "content": prompt}],
        )
        text = "".join(b.text for b in response.content if b.type == "text").strip()
        return text or None
    except Exception:
        return None


def _check_groundedness(text: str, row: dict, summary: dict) -> bool:
    return not _ungrounded_numbers(text, row, summary)


def generate_recommendation_narrative(row: dict, summary: dict) -> NarrativeResult:
    prompt = _build_prompt(row, summary)

    # Tier 1: local open-source model (default, free, no key, no third-party exposure)
    text = _try_ollama(prompt)
    if text:
        if _check_groundedness(text, row, summary):
            return NarrativeResult(text=text, provider="ollama", used_llm=True)
        return NarrativeResult(
            text=_template_narrative(row, summary), provider="template", used_llm=False,
            groundedness_check_passed=False,
            fallback_reason="Ollama output contained ungrounded figures -- rejected, "
                             "falling back to template rather than risk a fabricated number.",
        )

    # Tier 2: Anthropic API (optional, only if a key is set)
    text = _try_anthropic(prompt)
    if text:
        if _check_groundedness(text, row, summary):
            return NarrativeResult(text=text, provider="anthropic", used_llm=True)
        return NarrativeResult(
            text=_template_narrative(row, summary), provider="template", used_llm=False,
            groundedness_check_passed=False,
            fallback_reason="Anthropic output contained ungrounded figures -- rejected, "
                             "falling back to template rather than risk a fabricated number.",
        )

    # Tier 3: deterministic template -- zero hallucination risk by construction
    if os.environ.get("ANTHROPIC_API_KEY"):
        reason = (f"No local Ollama model reachable at {OLLAMA_HOST}, and the Claude API call "
                  f"failed (check the key and account credits) -- using deterministic template.")
    else:
        reason = (f"No local Ollama model reachable at {OLLAMA_HOST} and no "
                  f"ANTHROPIC_API_KEY set -- using deterministic template.")
    return NarrativeResult(
        text=_template_narrative(row, summary), provider="template", used_llm=False,
        fallback_reason=reason,
    )


if __name__ == "__main__":
    fake_row = {
        "channel": "Google", "current_daily_spend": 3161.0, "recommended_daily_spend": 3698.0,
        "change_pct": 17.0, "current_roas": 3.03, "projected_roas": 2.94,
        "current_marginal_roas": 2.48, "projected_marginal_roas": 2.33,
        "diminishing_returns_flag": False,
    }
    fake_summary = {"current_blended_roas": 2.91, "projected_blended_roas": 2.92}

    result = generate_recommendation_narrative(fake_row, fake_summary)
    print(f"Provider used: {result.provider}")
    print(f"Used LLM: {result.used_llm}")
    print(f"Groundedness check passed: {result.groundedness_check_passed}")
    if result.fallback_reason:
        print(f"Fallback reason: {result.fallback_reason}")
    print(f"\nNarrative:\n{result.text}")
