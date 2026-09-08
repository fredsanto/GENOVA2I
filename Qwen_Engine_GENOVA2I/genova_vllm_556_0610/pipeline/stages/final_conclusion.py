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
import re
from pathlib import Path
from typing import TYPE_CHECKING

from pipeline.core.acmg_points import recompute_and_fix_totals

if TYPE_CHECKING:
    from pipeline.llm.base import LLMClient

logger = logging.getLogger(__name__)

_PROMPT_PATH        = Path(__file__).parent.parent.parent / "prompts" / "clinical_conclusion.txt"
_REVISE_PROMPT_PATH = Path(__file__).parent.parent.parent / "prompts" / "final_conclusion_revise.txt"

MAX_NEW_TOKENS_FINAL_CONCLUSION = 3000

# With --max-model-len 16384, leave 2000 tokens for prompt template + output.
# ~4 chars per token → 14384 × 4 = 57536 chars safe budget for the conclusions block.
_MAX_CONCLUSIONS_CHARS = 50_000

# A real past failure: the model burned its whole token budget on rambling
# self-revision in STEP 2 (e.g. "*Wait*, looking closer... Re-reading the
# rule...") and got cut off mid-sentence in section 2, before ever reaching
# section 5 — but still contained the "# Clinical Conclusion" header, so the
# header-only check below let the truncated output through silently. Requiring
# every numbered section marker catches this: a genuinely truncated response
# is missing at least "5)" (the last section written).
_REQUIRED_SECTION_MARKERS = ("1)", "2)", "3)", "4)", "5)")


def _is_well_formed(text: str) -> bool:
    return "# Clinical Conclusion" in text and all(m in text for m in _REQUIRED_SECTION_MARKERS)


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


def _build_actionable_text(actionable_variants: list[dict] | None) -> str:
    """Render the ACMG SF actionable-variant list (ground truth for the
    Clinical Conclusion's own "Actionable findings" section) as plain lines
    of gene/HGVS/condition/classification — or an explicit "None" line so the
    model never has to guess whether the list was simply omitted."""
    if not actionable_variants:
        return "None — no variant in this run met ACMG SF actionable-gene criteria."
    lines = []
    for v in actionable_variants:
        lines.append(
            f"- {v['gene']} ({v['hgvs']}) — condition: {v['condition']}; "
            f"zygosity: {v['zygosity']}; classification: {v['classification']}"
        )
    return "\n".join(lines)


# ── Deterministic causative-list completeness check ─────────────────────────
#
# recompute_and_fix_totals() (pipeline.core.acmg_points) guarantees a stated
# total is arithmetically correct. It says nothing about whether a qualifying
# variant was silently dropped from section 2 entirely — a real observed
# failure: the synthesis model picked a single "winner" gene across layers
# (e.g. a De Novo finding) and demoted a separately-qualifying
# Dominant-Inherited finding in a different gene to an "also present, possibly
# coincidental" footnote, even though prompts/clinical_conclusion.txt's
# CAUSATIVE THRESHOLD RULE explicitly forbids treating this as a competition.
# This check re-derives, independently of the LLM, which variants the rule
# requires in section 2, and mechanically appends any the model dropped.

_CAUSATIVE_THRESHOLD = 6.0

_JOINT_STATUS_RE = re.compile(
    r"\*\*Joint compound-het classification:\*\*\s*(CAUSATIVE|COMPOUND VUS)"
)


def _qualifying_causative_findings(
    layer_outputs: dict[str, list[str]], unclassified_conclusions: list[str]
) -> list[dict]:
    """
    Collect every variant the CAUSATIVE THRESHOLD RULE requires in section 2:
    any block (De Novo / Dominant-Inherited / X-Linked / Unclassified / a
    homozygous-solo Recessive block) whose own final total is >= 6 points, or
    both variants of a compound-het PAIR block whose own "Joint compound-het
    classification" line says CAUSATIVE — never a COMPOUND VUS pair, even
    when one partner's individual total alone is >= 6 (see
    moi_recessive.py's _joint_compound_het_status).
    """
    findings: list[dict] = []

    def _process(block: str, layer_name: str) -> None:
        joint_m = _JOINT_STATUS_RE.search(block)
        if joint_m:
            if joint_m.group(1) != "CAUSATIVE":
                return  # COMPOUND VUS pair — neither partner is causative here
            headers = list(_HEADER_LINE_RE.finditer(block))
            totals = list(_ANY_TOTAL_RE.finditer(block))
            for header_m, total_m in zip(headers, totals):
                try:
                    points = float(total_m.group(1))
                except ValueError:
                    continue
                findings.append({
                    "gene":   header_m.group("gene").strip(),
                    "detail": header_m.group("detail").strip(),
                    "points": points,
                    "label":  total_m.group(2).strip(),
                    "layer":  layer_name,
                })
            return
        f = _extract_variant_finding(block)
        if f and f["points"] >= _CAUSATIVE_THRESHOLD:
            f["layer"] = layer_name
            findings.append(f)

    for layer_name, blocks in layer_outputs.items():
        for block in blocks:
            _process(block, layer_name)
    for block in unclassified_conclusions:
        _process(block, "Unclassified")
    return findings


def _missing_from_section_2(text: str, findings: list[dict]) -> list[dict]:
    """
    The subset of `findings` whose gene name is not mentioned in the
    "2) Causative variant(s):" section of `text` — checked against that
    section's own text only, since a gene demoted to a reasoning-prose
    footnote or the VUS section does not count as being listed as causative.
    """
    m2 = re.search(r"^\s*2\)\s", text, re.MULTILINE)
    if not m2:
        return findings  # no section 2 at all — everything is "missing"
    m3 = re.search(r"^\s*3\)\s", text, re.MULTILINE)
    section2 = text[m2.start(): m3.start() if m3 else len(text)]
    return [f for f in findings if not re.search(rf"\b{re.escape(f['gene'])}\b", section2)]


def _enforce_causative_completeness(
    text: str,
    layer_outputs: dict[str, list[str]],
    unclassified_conclusions: list[str],
) -> str:
    """
    Append a mechanically-assembled addendum, right before section 3, for any
    variant the CAUSATIVE THRESHOLD RULE requires in section 2 but that the
    LLM synthesis dropped. No-op (returns `text` unchanged) when nothing is
    missing.
    """
    findings = _qualifying_causative_findings(layer_outputs, unclassified_conclusions)
    missing = _missing_from_section_2(text, findings)
    if not missing:
        return text

    logger.warning(
        "[FinalConclusion] Completeness check: %d causative-threshold variant(s) "
        "missing from section 2 (%s) — appending mechanically.",
        len(missing), ", ".join(f["gene"] for f in missing),
    )
    addendum_lines = [
        "",
        "[COMPLETENESS CHECK — the following variant(s) independently meet the "
        "CAUSATIVE THRESHOLD RULE (own layer block total ACMG points >= 6, or "
        "part of a CAUSATIVE compound-het pair) but were not named above; "
        "listed here mechanically from each block's own already-verified "
        "total. Consult the MOI-layer blocks for full evidence, segregation, "
        "and citations.]",
    ]
    for f in missing:
        addendum_lines.append(
            f"- {f['gene']} ({f['detail']}) — [{f['layer']} layer] "
            f"{_fmt_points(f['points'])} pts total → {f['label']}"
        )
    addendum = "\n".join(addendum_lines) + "\n"

    m3 = re.search(r"^\s*3\)\s", text, re.MULTILINE)
    if m3:
        return text[: m3.start()] + addendum + "\n" + text[m3.start():]
    return text + "\n" + addendum


# ── Deterministic last-resort fallback ──────────────────────────────────────
#
# Used only when the LLM fails to produce a well-formed "# Clinical
# Conclusion" even after one retry (see run() below) — e.g. the SPATA13/
# arithmetic-rambling failure this module's docstring-level comments already
# describe, where the model burns its entire token budget re-litigating a
# single variant and never reaches the actual conclusion. Rather than ship a
# report with zero synthesis (previously: "whichever malformed pass happened
# to be longer"), assemble a minimal but guaranteed-well-formed conclusion
# mechanically from each block's own header + already-verified total line —
# every MOI/base stage (moi_denovo.py, moi_dominant.py, moi_recessive.py,
# moi_xlinked.py, conclusion.py) already ran recompute_and_fix_totals on its
# output before this stage ever saw it, so these numbers need no re-checking.

_HEADER_LINE_RE = re.compile(
    r"^#\s*[^\n]*?—\s*(?P<gene>[^(\n]+?)\s*\((?P<detail>[^\n]*)\)\s*$", re.MULTILINE
)
_ANY_TOTAL_RE = re.compile(
    r"\*\*(?:Total )?ACMG points:\*\*\s*([-+]?\d+(?:\.\d+)?)\s*→\s*([A-Za-z /()]+)"
)


def _fmt_points(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)


def _extract_variant_finding(block: str) -> dict | None:
    """
    Best-effort deterministic extraction of (gene, detail, points, label)
    from one layer/base block's own header line and its own final total
    line (the LAST "ACMG points"/"Total ACMG points" match — a MOI-layer
    block states a "Base ACMG points" line first and the delta-adjusted
    "Total ACMG points" line after it; the last match is always the correct,
    fully-adjusted one). Returns None if either piece can't be found — a
    missing finding here is preferable to a fabricated one.
    """
    header_m = _HEADER_LINE_RE.search(block)
    total_ms = list(_ANY_TOTAL_RE.finditer(block))
    if not header_m or not total_ms:
        return None
    points_str, label = total_ms[-1].groups()
    try:
        points = float(points_str)
    except ValueError:
        return None
    return {
        "gene":   header_m.group("gene").strip(),
        "detail": header_m.group("detail").strip(),
        "points": points,
        "label":  label.strip(),
    }


def _deterministic_fallback(
    layer_outputs: dict[str, list[str]],
    unclassified_conclusions: list[str],
    patient_phenotype: str,
    actionable_text: str,
) -> str:
    """
    Last-resort synthesis, guaranteed to pass _is_well_formed(), assembled
    with zero LLM involvement so it cannot itself ramble or truncate. Trades
    the LLM's prose/citations/segregation narrative for a guarantee that the
    report never ships with no answer at all.
    """
    findings = []
    for layer_name, blocks in layer_outputs.items():
        for block in blocks:
            f = _extract_variant_finding(block)
            if f:
                f["layer"] = layer_name
                findings.append(f)
    for block in unclassified_conclusions:
        f = _extract_variant_finding(block)
        if f:
            f["layer"] = "Unclassified"
            findings.append(f)

    causative = [f for f in findings if f["points"] >= 6]
    vus       = [f for f in findings if 4 <= f["points"] < 6]

    lines = [
        "# Clinical Conclusion",
        "",
        "[AUTOMATED FALLBACK — the clinical-conclusion synthesis stage did not "
        "produce a well-formed response even after a retry; this section was "
        "assembled mechanically from each finding's own already-verified ACMG "
        "total rather than narrative synthesis. Consult the MOI-layer blocks "
        "above for full evidence, segregation, and citations.]",
        "",
        f"1) Summary of clinical phenotype: {patient_phenotype}",
        "",
        "2) Causative variant(s):",
    ]
    if causative:
        for f in causative:
            lines.append(
                f"- {f['gene']} ({f['detail']}) — [{f['layer']} layer] "
                f"{_fmt_points(f['points'])} pts total → {f['label']}"
            )
    else:
        lines.append("None identified.")
    lines += [
        "",
        "3) Actionable findings (ACMG SF):",
        actionable_text,
        "",
        "4) Notable VUS (ACMG >= 4 and < 6 points):",
    ]
    if vus:
        for f in vus:
            lines.append(f"- {f['gene']} ({f['detail']}) — {_fmt_points(f['points'])} pts")
    else:
        lines.append("None identified.")
    lines += [
        "",
        "5) Summary: See causative variant(s) in section 2 above — full "
        "narrative synthesis was unavailable for this run; refer to each "
        "variant's own MOI-layer block for detailed evidence.",
    ]
    return "\n".join(lines)


def run(
    layer_outputs: dict[str, list[str]],
    unclassified_conclusions: list[str],
    patient_phenotype: str,
    llm: "LLMClient",
    actionable_variants: list[dict] | None = None,
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
        actionable_variants: ACMG SF actionable-gene findings (gene, hgvs,
                           condition, zygosity, classification dicts) — ground
                           truth for the Clinical Conclusion's own "Actionable
                           findings" section, computed independently upstream
                           (pipeline.py's acmg_sf.build_actionable_set).

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

    actionable_text = _build_actionable_text(actionable_variants)

    template    = _load_prompt()
    user_prompt = (template
        .replace("{patient_phenotype}", patient_phenotype)
        .replace("{conclusions}", conclusions_text)
        .replace("{actionable_variants}", actionable_text))

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
        .replace("{patient_phenotype}", patient_phenotype)
        .replace("{conclusions}", conclusions_text)
        .replace("{actionable_variants}", actionable_text)
        .replace("{draft}", draft))

    def _finalize(candidate: str) -> str:
        fixed = recompute_and_fix_totals(candidate)
        return _enforce_causative_completeness(fixed, layer_outputs, unclassified_conclusions)

    def _recover_from_malformed_draft() -> str:
        """
        Both draft and revised failed _is_well_formed() — a real recurring
        failure (see clinical_conclusion.txt's SPATA13/arithmetic-rambling
        notes): the model burns its whole token budget re-litigating a
        single variant and never reaches "# Clinical Conclusion" at all.
        Retry the draft generation once more (fresh call, same prompt —
        Qwen3.5 is non-deterministic enough at temperature>0 that a repeat
        call frequently avoids the same rabbit hole) before giving up. If
        the retry also fails, fall back to a deterministic, zero-LLM
        synthesis rather than shipping "whichever malformed pass happened
        to be longer" — a raw rambling transcript with no actual answer.
        """
        logger.warning(
            "[FinalConclusion] Both draft and revise passes are malformed/truncated "
            "(missing a numbered section) — retrying draft generation once more."
        )
        retry_draft = llm.generate(
            system=(
                "You are an expert clinical geneticist. Limit your response to "
                "1000 words maximum. Decide each variant's status ONCE and move "
                "on immediately — do not write multiple rounds of "
                "reconsideration or re-derive any stated ACMG total."
            ),
            user=user_prompt,
            max_tokens=MAX_NEW_TOKENS_FINAL_CONCLUSION,
        )
        if _is_well_formed(retry_draft):
            logger.info("[FinalConclusion] Retry succeeded — using retried draft.")
            return _finalize(retry_draft)

        logger.warning(
            "[FinalConclusion] Retry also malformed — falling back to a "
            "deterministic conclusion assembled from each layer's own "
            "already-verified totals."
        )
        return _enforce_causative_completeness(
            _deterministic_fallback(
                layer_outputs, unclassified_conclusions, patient_phenotype, actionable_text,
            ),
            layer_outputs, unclassified_conclusions,
        )

    try:
        revised = llm.generate(
            system="You are an expert clinical geneticist, fact-checking a draft report against source data.",
            user=revise_prompt,
            max_tokens=MAX_NEW_TOKENS_FINAL_CONCLUSION,
        )
    except Exception as exc:
        logger.warning("[FinalConclusion] Revise pass failed (%s) — using unrevised draft.", exc)
        if _is_well_formed(draft):
            return _finalize(draft)
        return _recover_from_malformed_draft()

    if not _is_well_formed(revised):
        if _is_well_formed(draft):
            logger.warning(
                "[FinalConclusion] Revise pass produced malformed/truncated output "
                "(missing a numbered section) — using unrevised draft."
            )
            return _finalize(draft)
        return _recover_from_malformed_draft()

    # Re-sum every "**ACMG classification:** Label (N pts total)" line, then
    # re-check causative-list completeness, here — against its own
    # immediately-preceding criteria bullets / the full layer-output set —
    # before handing the report to the user. This is the block the user
    # actually reads, and either failure surviving every upstream stage's own
    # check is still visible here even if it wasn't visible earlier. See
    # acmg_points.recompute_and_fix_totals and _enforce_causative_completeness
    # above for the recurring failures these guard against.
    return _finalize(revised)
