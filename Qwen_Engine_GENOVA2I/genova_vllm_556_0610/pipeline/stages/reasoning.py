"""
pipeline/stages/reasoning.py — Two-call SLM clinical reasoning stage.

Splitting the original single call into two focused calls reduces hallucination
by separating the evidence-grounded narrative from the inclusion decision:

  Call 1 — reasoning:  step-by-step clinical reasoning using ONLY cited evidence.
                       Prompt: prompts/reasoning.txt
  Call 2 — second_triage: structured INCLUDE/EXCLUDE decision with justification.
                           Prompt: prompts/second_triage.txt
                           Format: Include-case / Exclude-case / Decision (3 lines)

The two outputs are concatenated before returning so that the REASONING display
section in pipeline.py is self-contained.

Public API:
    run_one(variant_context, llm) -> str
    parse_inclusion_decision(reasoning_text) -> tuple[str, str]
        Returns ("INCLUDE"|"EXCLUDE", "Include: <...> | Exclude: <...>").
"""

from __future__ import annotations

import logging
import re
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_REASONING_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "reasoning.txt"
_SCORING_PROMPT_PATH   = Path(__file__).parent.parent.parent / "prompts" / "second_triage.txt"

# Call 1: step-by-step narrative — no score, citation-only rules apply
MAX_NEW_TOKENS_REASONING = 1500
# Call 2: three-line Include-case / Exclude-case / Decision block
# 200 gives headroom for any residual preamble before the 3 structured lines.
MAX_NEW_TOKENS_SCORING   = 200

_INCLUDE_CASE_RE = re.compile(r"Include-case:\s*(.+)", re.IGNORECASE)
_EXCLUDE_CASE_RE = re.compile(r"Exclude-case:\s*(.+)", re.IGNORECASE)
_DECISION_RE     = re.compile(r"Decision:\s*(INCLUDE|EXCLUDE)", re.IGNORECASE)


# ── Prompt loaders ────────────────────────────────────────────────────────────

def _load_reasoning_prompt() -> str:
    if _REASONING_PROMPT_PATH.exists():
        return _REASONING_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Reasoning prompt not found at {_REASONING_PROMPT_PATH}."
    )


def _load_scoring_prompt() -> str:
    if _SCORING_PROMPT_PATH.exists():
        return _SCORING_PROMPT_PATH.read_text(encoding="utf-8")
    raise FileNotFoundError(
        f"Second-triage prompt not found at {_SCORING_PROMPT_PATH}."
    )


# ── Score parser ──────────────────────────────────────────────────────────────

def parse_inclusion_decision(reasoning_text: str) -> tuple[str, str]:
    """
    Parse the INCLUDE/EXCLUDE decision and justification from run_one output.

    Expected embedded format (after the "SECOND TRIAGE:" marker):
        Include-case: <strongest reason to include, ≤15 words>
        Exclude-case: <strongest reason to exclude, ≤15 words>
        Decision: INCLUDE or EXCLUDE

    Blank lines are stripped before parsing (Qwen without thinking sometimes
    inserts them between lines).

    Returns:
        ("INCLUDE" | "EXCLUDE", justification)
        justification = "Include: <include_case> | Exclude: <exclude_case>"
        Defaults to ("INCLUDE", "parse error — defaulting to INCLUDE") on failure.
    """
    cleaned = "\n".join(line for line in reasoning_text.splitlines() if line.strip())

    include_m  = _INCLUDE_CASE_RE.search(cleaned)
    exclude_m  = _EXCLUDE_CASE_RE.search(cleaned)
    decision_m = _DECISION_RE.search(cleaned)

    include_str = include_m.group(1).strip() if include_m else ""
    exclude_str = exclude_m.group(1).strip() if exclude_m else ""

    logger.debug(
        "[SecondTriage] Include-case: %r | Exclude-case: %r | raw Decision line: %r",
        include_str, exclude_str,
        decision_m.group(0) if decision_m else "(not found)",
    )

    if decision_m:
        decision = decision_m.group(1).upper()
        justification = (
            f"Include: {include_str} | Exclude: {exclude_str}"
            if (include_str or exclude_str)
            else decision
        )
        return (decision, justification)

    # Fallback: model may have ignored the three-line format; search the tail
    tail = reasoning_text[-150:]
    m2 = re.search(r"\b(INCLUDE|EXCLUDE)\b", tail, re.IGNORECASE)
    if m2:
        logger.warning(
            "[SecondTriage] Decision: prefix not found — matched bare word in tail: %r",
            m2.group(0),
        )
        return (m2.group(1).upper(), f"Include: {include_str} | Exclude: {exclude_str}")

    logger.warning(
        "[SecondTriage] Could not parse INCLUDE/EXCLUDE decision — defaulting to INCLUDE. "
        "Raw tail: %r",
        reasoning_text[-200:],
    )
    return ("INCLUDE", "parse error — defaulting to INCLUDE")


# ── Main entry points ──────────────────────────────────────────────────────────

def run_reasoning(
    variant_context: str,
    llm: "LLMClient",
    sibling_context_block: str = "",
    inheritance_mode_block: str = "",
) -> str:
    """
    Stage 2a — Call 1: step-by-step clinical reasoning grounded in cited evidence only.
    The model is instructed not to introduce external knowledge without a reference
    present in the provided context.

    Split out from the former combined run_one() so the pipeline can run reasoning
    for every kept variant in a gene group before any of them proceeds to
    second_triage — needed so second_triage can see sibling variants' reasoning
    (compound-het safeguard for AR/XLR genes).

    Args:
        variant_context:         Per-variant context string from Stage 1 (retrieval).
        llm:                     Shared LLMClient instance.
        sibling_context_block:   Optional block describing other kept variants in the
                                 same gene (only populated for AR/XLR genes with ≥2
                                 kept variants and this variant heterozygous) — built
                                 from raw variant evidence (not sibling reasoning,
                                 which doesn't exist yet at Call 1) so this first
                                 reasoning pass already weighs the variant jointly
                                 with its gene-mate instead of in isolation. Empty
                                 string when not applicable.
        inheritance_mode_block:  Backend-determined gene inheritance mode (AD/AR/
                                 XLD/XLR/unknown), stated as a fact so the model
                                 reports it in the summary table rather than
                                 re-deriving it from prose evidence.

    Returns:
        Step-by-step reasoning text.
    """
    logger.info("[Reasoning] Call 1/2 — step-by-step reasoning...")
    reasoning_template = _load_reasoning_prompt()
    reasoning_user     = (
        reasoning_template
        .replace("{augmented_context}", variant_context)
        .replace("{sibling_context_block}", sibling_context_block)
        .replace("{inheritance_mode_block}", inheritance_mode_block)
    )

    return llm.generate(
        system="You are an expert clinical geneticist.",
        user=reasoning_user,
        max_tokens=MAX_NEW_TOKENS_REASONING,
    )


def run_second_triage(
    variant_context: str,
    reasoning_text: str,
    llm: "LLMClient",
    sibling_context_block: str = "",
) -> str:
    """
    Stage 2b — Call 2: structured INCLUDE/EXCLUDE decision with justification.

    Args:
        variant_context:        Per-variant context string from Stage 1 (retrieval).
        reasoning_text:          Output of run_reasoning() for this variant.
        llm:                     Shared LLMClient instance.
        sibling_context_block:   Optional block describing other kept variants in
                                 the same gene (only populated for AR/AD-AR genes
                                 with ≥2 kept variants) — lets second_triage apply
                                 the compound-het safeguard instead of judging this
                                 variant in isolation. Empty string when not applicable.

    Returns:
        Combined text: step-by-step reasoning followed by the inclusion decision
        on its own line. Compatible with parse_inclusion_decision() and with the
        REASONING display section in pipeline.py.
    """
    logger.info("[Reasoning] Call 2/2 — second_triage inclusion decision (INCLUDE/EXCLUDE)...")
    scoring_template = _load_scoring_prompt()
    scoring_user     = (
        scoring_template
        .replace("{augmented_context}", variant_context)
        .replace("{reasoning}", reasoning_text)
        .replace("{sibling_context_block}", sibling_context_block)
    )

    decision_text = llm.generate(
        system="You are an expert clinical geneticist.",
        user=scoring_user,
        max_tokens=MAX_NEW_TOKENS_SCORING,
    )

    logger.debug("[SecondTriage] Full model output:\n%s", decision_text.strip())

    # parse_inclusion_decision() searches for the Decision: line within the
    # "SECOND TRIAGE:" block.  The full three-line output is preserved so the
    # REASONING display section shows Include-case / Exclude-case / Decision.
    combined = reasoning_text.rstrip() + "\n\nSECOND TRIAGE:\n" + decision_text.strip()
    return combined
