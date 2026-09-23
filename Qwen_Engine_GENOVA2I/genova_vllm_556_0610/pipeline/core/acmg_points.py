"""
pipeline/core/acmg_points.py — shared "**ACMG points:** N → Classification"
line parsing/rewriting, used by every mechanical ACMG-criterion validator
(acmg_pp3.py, acmg_pp2_bp1.py, ...) that strips a criterion the SLM applied
without adequate grounding and needs to keep the stated total consistent.
"""

import logging
import re

logger = logging.getLogger(__name__)

# Stage-4 conclusion.py's own "**ACMG criteria:**" bullet block followed by
# its "**ACMG points:** N → Label" total — the ONE canonical, already-
# computed base score for a variant. See extract_base_acmg() below.
_BASE_CRITERIA_BLOCK_RE = re.compile(
    r"\*\*ACMG criteria:\*\*\s*\n((?:-[^\n]*\n?)+)\*\*ACMG points:\*\*\s*"
    r"([-+]?\d+(?:\.\d+)?)\s*→\s*[A-Za-z /()]+"
)

_POINTS_LINE_RE = re.compile(
    r"(\*\*(?:Total )?ACMG points:\*\*\s*)([-+]?\d+(?:\.\d+)?)(\s*→\s*)([A-Za-z /()]+)"
)

# moi_*.py's "**Base ACMG points:** N" line (the copied-verbatim base total,
# before that layer's own delta) — a third total-line format alongside
# _POINTS_LINE_RE ("ACMG points:"/"Total ACMG points:") and
# _CLASSIFICATION_LINE_RE below ("ACMG classification:"). Kept separate from
# _POINTS_LINE_RE (rather than making "Total" one of several optional
# prefixes) so adjust_points_line()/relabel_all_points_lines() — used
# elsewhere against the "Total" total specifically — don't start matching
# the "Base" line too.
#
# The trailing "→ Label" is OPTIONAL here, unlike _POINTS_LINE_RE, because
# every moi_*.py prompt template (moi_dominant.txt, moi_recessive.txt,
# moi_denovo.txt, moi_xlinked.txt, moi_recessive_homozygous.txt) explicitly
# instructs this line as a bare copied number with no label — "**Base ACMG
# points:** [copy the numeric value ... verbatim]" — never "N → Label". A
# real, significant observed failure: this regex previously REQUIRED the
# arrow+label suffix, so it silently never matched the Base line's actual
# real-world shape in ANY MOI layer block, meaning recompute_and_fix_totals's
# Base-line correction (first pass, below) and its Total-line reconciliation
# (second pass, which depends on successfully matching the Base line to read
# its value) both silently no-op'd for every single moi_*.py block ever
# generated — the mechanical arithmetic safety net this function exists to
# provide was never actually running for MOI-layer totals at all.
_BASE_POINTS_LINE_RE = re.compile(
    r"(\*\*Base ACMG points:\*\*\s*)([-+]?\d+(?:\.\d+)?)(\s*→\s*[A-Za-z /()]+)?"
)

# final_conclusion.py's per-variant format: "**ACMG classification:** Label
# (N pts total)" — distinct from the "**ACMG points:** N -> Label" format
# above (conclusion.py / moi_*.py), so recompute_and_fix_totals() needs to
# recognize and rewrite both.
_CLASSIFICATION_LINE_RE = re.compile(
    r"(\*\*ACMG classification:\*\*\s*)([A-Za-z /()]+?)(\s*\(\s*)([-+]?\d+(?:\.\d+)?)(\s*pts total\s*\))"
)

# A criterion bullet's own point tag, e.g. "[VeryStrong, +8]" or
# "[Moderate, +2 pts]" — every criterion line carries exactly one of these
# per the fixed format both conclusion.txt and clinical_conclusion.txt
# require.
_CRITERION_TAG_RE = re.compile(r"\[[A-Za-z\s]+,\s*([+-]?\d+(?:\.\d+)?)\s*(?:pts?)?\]")

# Every standard ACMG/AMP criterion code.
_CRITERION_CODE_RE = r"(?:PVS1|PS[1-4]|PM[1-6]|PP[1-5]|BA1|BS[1-4]|BP[1-7])"

# One applied-criterion bullet, in ANY of the shapes the different prompt
# templates (and the SLM's own free variation on them) actually render:
#   "- PVS1 [VeryStrong, +8]: ..."
#   "*   **PS2** (Strong, +4 pts): ..."
#   "[PM6] (Moderate, +2 pts): ..."
#   "PM3 (Supporting, +0.5 pts): ..."
#   "**ACMG criteria:** PS1 (Strong, +4 pts): ...; PM2 (Moderate, +2 pts): ...; PP3 (...): ..."
# — an optional leading bullet marker, optional bold/bracket wrapping
# around the code, then a strength+points tag in either [] or () directly
# after it. Deliberately position-independent: matched anywhere in the
# text a code+tag pair occurs, not anchored to line starts.
#
# Previously anchored to (?:^|\n) before the optional bullet marker — this
# silently broke on final_conclusion.py's own semicolon-joined single-line
# rendering of multiple criteria (e.g. "PS1 (...); PM2 (...); PP3 (...)"
# all on one line after "**ACMG criteria:** "), where every criterion after
# the first isn't preceded by a newline. Real observed failure: a
# PS1+PM2+PP3+PM3 = 7.5 list rendered this way had gather_criteria() find
# only PS1 (the sole line-start match) or nothing at all when even PS1 sat
# mid-line after the header text, so recompute_and_fix_totals() either
# left the model's own wrong stated total untouched or "corrected" it down
# to a partial sum — either way shipping "4 pts total (VUS)" for a report
# whose own listed criteria summed to 7.5 (Likely Pathogenic). The trailing
# requirement of an immediately-following "[Strength, +N]"/"(Strength, +N
# pts)" tag already excludes bare narrative mentions of a code (e.g.
# "supported by prior evidence (PS1)" or "(PS1)" with no strength/points
# inside the same parens) without needing a line-start anchor as a second
# line of defense — verified: neither false-positive shape matches this
# pattern regardless of position in the text.
#
# The strength word is optional: moi_recessive_homozygous.txt renders its
# delta bullet as "PM3 [+0.5]" (no strength). Requiring one dropped that PM3
# from the Total line's re-sum, so every homozygous Total was silently
# "corrected" back down to the Base value (Base 5, delta +0.5, Total 5).
_CRITERION_MENTION_RE = re.compile(
    r"[ \t]*[-*•]?[ \t]*\**\[?\b(" + _CRITERION_CODE_RE + r")\b\]?\**"
    r"[ \t]*[\(\[](?:[A-Za-z][A-Za-z \t]*,\s*)?([+-]?\d+(?:\.\d+)?)\s*(?:pts?)?\s*[\)\]]"
)


def gather_criteria(text: str) -> list[tuple[str, float]]:
    """
    Scans `text` for every ACMG-criterion bullet (any rendered shape — see
    _CRITERION_MENTION_RE) and returns (code, points) pairs in first-seen
    order, one per unique code.

    Exists because the position-based approach (walk back N contiguous
    non-blank lines above a stated total, as recompute_and_fix_totals used
    to) breaks the instant the SLM's own formatting deviates even slightly
    from the prompt template's exact layout (an extra blank line, a
    differently-worded label) — a fragility that has silently disabled the
    arithmetic safety net multiple times (see recompute_and_fix_totals's
    docstring). This instead finds every criterion mention in the text
    regardless of where it sits, so blank lines, reordering, or reworded
    headers around it don't matter.

    A criterion code mentioned more than once with the SAME point value is
    silently deduplicated (redundant restatement, e.g. the SLM echoing a
    criterion in prose after already listing it as a bullet). A code
    mentioned twice with DIFFERENT point values is a genuine inconsistency
    in the SLM's own output — the first occurrence wins and the disagreement
    is logged, rather than silently summing or averaging two numbers that
    can't both be right.
    """
    seen: dict[str, float] = {}
    order: list[str] = []
    for code, points_str in _CRITERION_MENTION_RE.findall(text):
        points = float(points_str)
        if code in seen:
            if seen[code] != points:
                logger.warning(
                    "acmg_points.gather_criteria: %s mentioned twice with "
                    "different point values (%s vs %s) in the same block — "
                    "keeping the first, dropping the duplicate",
                    code, seen[code], points,
                )
            continue
        seen[code] = points
        order.append(code)
    return [(code, seen[code]) for code in order]


def sum_criteria(text: str) -> float:
    """Sum of gather_criteria(text)'s deduplicated per-code point values."""
    return sum(points for _, points in gather_criteria(text))


# moi_*.py's own "**Base ACMG criteria (...):**" header — the boundary a
# Base-line's criteria are scoped to start from (excludes that layer's own
# earlier "<Layer> criteria applied" bullet, e.g. PS2, which must count
# toward the Total line but not the Base line). Worded slightly differently
# across templates ("BASE CONCLUSION's" vs "the BASE CONCLUSION's") — match
# on the stable "Base ACMG criteria" prefix only.
_BASE_HEADER_RE = re.compile(r"\*\*Base ACMG criteria\b[^\n]*\*\*")

# The header that opens a self-contained "one variant's worth of ACMG
# criteria" sub-block, in either of the two shapes the pipeline renders:
# moi_*.py's "**<Layer> criteria applied:**" (e.g. "De novo criteria
# applied") or conclusion.py's/final_conclusion.py's own "**ACMG
# criteria:**". Used to bound how far back recompute_and_fix_totals()
# looks for criteria belonging to a given total line, so a concatenated
# multi-variant text (final_conclusion.py's combined output, or a
# compound-het pair's two variant blocks in one moi_recessive.py result)
# doesn't pull an earlier variant's criteria into a later variant's total.
_BLOCK_START_RE = re.compile(
    r"\*\*(?:[A-Za-z][A-Za-z ]* criteria applied|ACMG criteria):\*\*"
)

# (inclusive lower bound, label) — highest first; matches the thresholds
# block at the bottom of prompts/conclusion.txt.
_THRESHOLDS = [
    (10.0, "Pathogenic"),
    (6.0,  "Likely Pathogenic"),
    (0.0,  "Uncertain Significance (VUS)"),
    (-6.0, "Likely Benign"),
    (float("-inf"), "Benign"),
]


def classify(points: float) -> str:
    for lo, label in _THRESHOLDS:
        if points >= lo:
            return label
    return "Benign"


def adjust_points_line(text: str, delta: float) -> str:
    """
    Adds `delta` to the stated "**ACMG points:** N → Label" total and
    rewrites the classification label to match. No-op if the line isn't
    found. Safe to call repeatedly (e.g. once per stripped criterion) since
    it re-reads the current value from `text` each time.
    """
    def _adjust(m: re.Match) -> str:
        prefix, points_str, arrow, _old_label = m.groups()
        try:
            new_points = float(points_str) + delta
        except ValueError:
            return m.group(0)
        new_points_str = str(int(new_points)) if new_points == int(new_points) else str(new_points)
        return f"{prefix}{new_points_str}{arrow}{classify(new_points)}"

    return _POINTS_LINE_RE.sub(_adjust, text, count=1)


def relabel_all_points_lines(text: str) -> str:
    """
    Re-derives the classification label on EVERY "**[Total ]ACMG points:** N →
    Label" line in `text` from N via classify(), leaving N itself unchanged.
    Fixes the SLM occasionally writing an internally-inconsistent label for
    its own stated total (e.g. "4.5 → Likely Pathogenic" when 4.5 is in the
    0-5 VUS band, not the 6-9 Likely Pathogenic band) — a real observed
    failure where a homozygous ClinVar Likely-Pathogenic variant scored 4.5
    points, got mislabeled "Likely Pathogenic" instead of "Uncertain
    Significance (VUS)", and as a result fell through every section of the
    Clinical Conclusion (not causative since 4.5 < 6, not Notable VUS since
    its label wasn't "VUS", so the SCOPE RULE omitted it from the report
    entirely). Unlike adjust_points_line(), fixes ALL matches in the text
    (a compound-het pair block has two such lines), not just the first.
    """

    def _relabel(m: re.Match) -> str:
        prefix, points_str, arrow, _old_label = m.groups()
        try:
            points = float(points_str)
        except ValueError:
            return m.group(0)
        return f"{prefix}{points_str}{arrow}{classify(points)}"

    return _POINTS_LINE_RE.sub(_relabel, text)


def _sum_str(n: float) -> str:
    return str(int(n)) if n == int(n) else str(n)


def recompute_and_fix_totals(text: str) -> str:
    """
    Deterministically re-sums the ACMG criteria that belong to each stated
    total line via gather_criteria() — a position-independent scan for
    every criterion mention in the relevant scope, deduplicated by code —
    and overwrites that total, and its classification label, whenever it
    disagrees with the SLM's own stated number.

    A real, RECURRING observed failure — the exact scenario conclusion.txt's
    own "real past failure" warning already describes, which happened again
    anyway: a criteria list totalling PS3(+4)+PM2(+2)+PP4(+1) = 7 stated as
    "ACMG points: 11". The stale 11 then carried forward unchanged through a
    MOI layer's "+2" delta into a final reported total of 13, with every
    downstream stage trusting the upstream total rather than the actual
    listed criteria. Prompt instructions to self-check this arithmetic
    exist and are followed only sometimes — this makes the check
    unconditional.

    This replaces an earlier, position-based version (walk back over the
    contiguous run of non-blank lines immediately above the total) that
    broke silently the instant the SLM's own rendering deviated even
    slightly from the prompt template's exact spacing — e.g. a single blank
    line the SLM inserted before "**Base ACMG points:**" stopped the
    backward scan before it ever reached the criteria bullets, leaving a
    stale, wrong total (and the classification derived from it) shipped
    unchanged, with no error or warning anywhere. gather_criteria() instead
    finds every criterion mention within a bounded scope regardless of
    exactly where it sits relative to blank lines or restated delta lines.

    Handles all three rendered total-line formats used across the pipeline,
    each with its own scope:
    - "**ACMG points:**" (conclusion.py) / "**ACMG classification:** Label
      (N pts total)" (final_conclusion.py) — scoped from the nearest
      preceding "**ACMG criteria:**" header to the total line, so a
      concatenated multi-variant text (final_conclusion.py's combined
      output) doesn't pull an earlier variant's criteria into this one's
      total.
    - "**Base ACMG points:** N" (each moi_*.py layer's copied-verbatim base
      total) — scoped from the nearest preceding "**Base ACMG criteria**"
      header, which deliberately excludes that layer's own earlier
      "<Layer> criteria applied" delta bullet (e.g. PS2) — the Base line
      must reflect the base score alone.
    - "**Total ACMG points:** N" (moi_*.py's base+delta total) — scoped
      from the nearest preceding "**<Layer> criteria applied:**" header,
      which — unlike the Base line's scope — DOES include that layer's own
      delta bullet, giving base + delta in one pass without needing a
      separate reconciliation step.
    Falls back to scoping from the start of `text` if no bounding header is
    found, matching the old function's behavior for any block shape that
    predates this convention.
    """

    out = text

    # (regex to match the total line, header marking the start of its
    #  criteria scope, index of the group holding the currently-stated
    #  numeric value, rewrite function given (match, corrected_total))
    passes = (
        (
            _BASE_POINTS_LINE_RE,
            _BASE_HEADER_RE,
            2,
            lambda m, actual: f"{m.group(1)}{_sum_str(actual)} → {classify(actual)}",
        ),
        (
            _POINTS_LINE_RE,
            _BLOCK_START_RE,
            2,
            lambda m, actual: f"{m.group(1)}{_sum_str(actual)}{m.group(3)}{classify(actual)}",
        ),
        (
            _CLASSIFICATION_LINE_RE,
            _BLOCK_START_RE,
            4,
            lambda m, actual: (
                f"{m.group(1)}{classify(actual)}{m.group(3)}"
                f"{_sum_str(actual)}{m.group(5)}"
            ),
        ),
    )

    for regex, header_re, value_group, fmt in passes:
        pos = 0
        pieces = []
        for m in regex.finditer(out):
            pieces.append(out[pos : m.start()])
            start = 0
            for hm in header_re.finditer(out, 0, m.start()):
                start = hm.end()
            crit = gather_criteria(out[start : m.start()])
            if crit:
                actual = sum(v for _, v in crit)
                if actual != float(m.group(value_group)):
                    pieces.append(fmt(m, actual))
                    pos = m.end()
                    continue
            pieces.append(m.group(0))
            pos = m.end()
        pieces.append(out[pos:])
        out = "".join(pieces)

    return out


def extract_base_acmg(base_conclusion: str) -> tuple[str, float] | None:
    """
    Pulls the frozen, already-computed "**ACMG criteria:**" bullet list and
    its "**ACMG points:** N → Label" total straight out of a variant's own
    Stage-4 conclusion.py output (conclusions[i]) — the ONE canonical base
    score for that variant. Every MOI layer (de novo, dominant, recessive,
    X-linked) must splice in this exact block mechanically rather than
    asking the SLM to re-transcribe or re-derive it from context on every
    layer call.

    A real observed failure this replaces: the same variant's "Base
    ACMG points" came out as three different numbers (8, 4, 2) across its
    three MOI-layer blocks in one run, none of which even matched their own
    listed criteria in that same block — despite every layer prompt
    instructing "copy verbatim, do not invent" from the identical frozen
    base_conclusion text. The model was re-deriving the number under each
    layer's framing instead of copying it, and no mechanical check caught
    the disagreement because each layer's "Base ACMG points" line was
    validated only against criteria the model ALSO transcribed itself in
    that same call — a check that can never catch a model that transcribes
    a wrong number and a matching-but-wrong criteria list together.

    Returns (block_text, points), where block_text is ready to splice in
    as-is — a "**Base ACMG criteria:**" bullet list followed by a "**Base
    ACMG points:** N → Label" line, with the label re-derived via
    classify() (not copied from the base conclusion's own label) so it can
    never disagree with N. Returns None if base_conclusion doesn't contain
    the expected Stage-4 shape (caller should then fall back to whatever
    the LLM produces rather than injecting nothing).
    """
    m = _BASE_CRITERIA_BLOCK_RE.search(base_conclusion)
    if not m:
        return None
    bullets, points_str = m.groups()
    points = float(points_str)
    block = (
        "**Base ACMG criteria (ground truth — mechanically copied from this "
        "variant's own Stage-4 conclusion; not re-derived at this layer):**\n"
        f"{bullets.rstrip()}\n"
        f"**Base ACMG points:** {_sum_str(points)} → {classify(points)}"
    )
    return block, points
