"""
pipeline/stages/final_conclusion.py — Cross-MOI clinical conclusion stage (Layer 8).

The last synthesis stage. Reads the per-MOI-layer variant blocks produced by
stages/moi_denovo.py, moi_dominant.py, moi_recessive.py, moi_xlinked.py (each
already carrying its own base+delta ACMG total) plus any Unclassified base
conclusions, and generates the short "# Clinical Conclusion" paragraph that
closes the final report — now reasoning across MOI layers, not just across a
flat variant list, since a patient can have independently-explanatory
findings under different inheritance mechanisms (e.g. an AD_AR gene's variant
appearing in both the dominant and recessive layers).

This is separated from stages/conclusion.py (Layer 2) so each variant's
structured block can be generated individually (reducing SLM memory pressure)
while the overall summary still has visibility across everything.

Prompt loaded from prompts/clinical_conclusion.txt.

Public API:
    run(layer_outputs, unclassified_conclusions, patient_phenotype, llm) -> str
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH        = Path(__file__).parent.parent.parent / "prompts" / "clinical_conclusion.txt"
_REVISE_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "final_conclusion_revise.txt"

MAX_NEW_TOKENS_FINAL_CONCLUSION = 1500

# With --max-model-len 16384, leave 2000 tokens for prompt template + output.
# ~4 chars per token → 14384 × 4 = 57536 chars safe budget for the conclusions block.
_MAX_CONCLUSIONS_CHARS = 50_000


def _load_prompt(path: Path = _PROMPT_PATH) -> str:
    if path.exists():
        return path.read_text(encoding="utf-8")
    raise FileNotFoundError(f"Prompt not found at {path}.")


def _build_layers_text(layer_outputs: dict[str, list[str]], unclassified_conclusions: list[str]) -> str:
    """Render each MOI layer's blocks under its own labelled header, plus an
    Unclassified section if present, so the model sees which layer each
    finding came from rather than a flat undifferentiated list."""
    parts = []
    for layer_name, blocks in layer_outputs.items():
        if not blocks:
            continue
        parts.append(f"=== {layer_name.upper()} LAYER ===\n\n" + "\n\n---\n\n".join(blocks))
    if unclassified_conclusions:
        parts.append(
            "=== UNCLASSIFIED (no MOI-specific analysis; base ACMG score only) ===\n\n"
            + "\n\n---\n\n".join(unclassified_conclusions)
        )
    return "\n\n---\n\n".join(parts)


def run(
    layer_outputs: dict[str, list[str]],
    unclassified_conclusions: list[str],
    patient_phenotype: str,
    llm: "LLMClient",
) -> str:
    """
    Cross-MOI clinical conclusion synthesis (Layer 8).

    Args:
        layer_outputs:     {layer_name: [block, ...]} — one entry per MOI layer
                           that produced at least one finding (e.g. "De Novo",
                           "Dominant-Inherited", "Recessive", "X-Linked"). Each
                           block already carries its own "Total ACMG points:
                           N → Classification" line (base + that layer's delta).
        unclassified_conclusions: Base-layer-only conclusion blocks for included
                           variants whose gene MOI never resolved to any layer.
        patient_phenotype: Free-text patient phenotype string from the request.
        llm:               Shared LLMClient instance.

    Returns:
        The "# Clinical Conclusion" section, now naming which MOI layer(s)
        jointly explain the phenotype rather than picking one variant from a
        flat list.
    """
    logger.info("[FinalConclusion] Synthesising cross-MOI clinical conclusion...")

    conclusions_text = _build_layers_text(layer_outputs, unclassified_conclusions)

    if len(conclusions_text) > _MAX_CONCLUSIONS_CHARS:
        logger.warning(
            "[FinalConclusion] Combined layer outputs (%d chars) exceeds budget — truncating.",
            len(conclusions_text),
        )
        conclusions_text = conclusions_text[:_MAX_CONCLUSIONS_CHARS] + "\n\n[... truncated ...]"

    template    = _load_prompt()
    user_prompt = (template
        .replace("{patient_phenotype}", patient_phenotype)
        .replace("{conclusions}", conclusions_text))

    draft = llm.generate(
        system="You are an expert clinical geneticist. Limit your response to 1000 words maximum.",
        user=user_prompt,
        max_tokens=MAX_NEW_TOKENS_FINAL_CONCLUSION,
    )

    # Second pass: hand the draft back to the model as a fresh call and ask it to
    # check for contradictions against the per-variant conclusions. A self-check
    # instruction folded into the same generation as the draft doesn't reliably
    # catch its own mistakes (single-pass, buried instruction); a separate call
    # with the draft as input text to critique works — same pattern as the
    # /chat follow-up endpoint, which reliably corrects these when asked directly.
    revise_template = _load_prompt(_REVISE_PROMPT_PATH)
    revise_prompt = (revise_template
        .replace("{conclusions}", conclusions_text)
        .replace("{draft}", draft))

    try:
        revised = llm.generate(
            system="You are an expert clinical geneticist, fact-checking a draft report against source data.",
            user=revise_prompt,
            max_tokens=MAX_NEW_TOKENS_FINAL_CONCLUSION,
        )
    except Exception as exc:
        logger.warning("[FinalConclusion] Revise pass failed (%s) — using unrevised draft.", exc)
        return draft

    if "# Clinical Conclusion" not in revised:
        logger.warning("[FinalConclusion] Revise pass produced malformed output — using unrevised draft.")
        return draft

    return revised
