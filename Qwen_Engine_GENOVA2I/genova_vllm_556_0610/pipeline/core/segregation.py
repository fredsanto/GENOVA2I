"""
pipeline/core/segregation.py — Deterministic allelic-balance segregation logic.

Promotes AB-interpretation rules that previously existed only as LLM prompt
instructions (prompts/reasoning.txt) or as a display-only summary listing
(pipeline.py's _build_segregation_analysis) into reusable, testable Python
functions consumed by the MOI-layer stages (de novo, dominant, recessive,
X-linked). Thresholds match what was already established elsewhere in this
codebase rather than inventing new numbers:
  - 0.3-0.7  -> het   (prompts/reasoning.txt)
  - >0.8     -> hom   (prompts/reasoning.txt)
  - <0.1     -> absent / de-novo signal (pipeline.py's existing 0.1 threshold)

Public API:
    has_parent_data(ab_entry) -> bool
    classify_ab_ratio(value) -> "het" | "hom" | "absent" | "uncertain"
    classify_segregation(proband_ab, mother_ab, father_ab) -> str
    classify_phase(segregation_a, segregation_b) -> "cis" | "trans" | "unknown"
    classify_xlinked_ab(proband_ab, mother_ab) -> "XLR" | "XLD" | "uncertain"
"""

from __future__ import annotations


def _is_present(value) -> bool:
    """True if value is a non-empty, non-NA string/number."""
    if value is None:
        return False
    s = str(value).strip()
    return s != "" and s.upper() != "NA"


def has_parent_data(ab_entry: dict | None) -> bool:
    """True only when the 3-column proband/mother/father allelic-balance shape
    (extract_parental_ab in normalizer.py) is present with non-empty, non-NA
    mother AND father values. A 1- or 2-column shape (proband-only, or
    proband+one unlabeled extra column) does not count as parent data."""
    if not ab_entry:
        return False
    return _is_present(ab_entry.get("mother")) and _is_present(ab_entry.get("father"))


def has_any_parent_data(ab_entry: dict | None) -> bool:
    """True when at least one of mother/father is present (trio OR partial
    one-parent-only shape) — used together with has_parent_data() to
    distinguish "partial (one parent)" from "none" for Layer 3's de novo
    parental-data-status branching."""
    if not ab_entry:
        return False
    return _is_present(ab_entry.get("mother")) or _is_present(ab_entry.get("father"))


def classify_ab_ratio(value) -> str:
    """Bucket a single allelic-balance value. Unparseable/missing values are
    "uncertain", not silently treated as 0 or skipped — callers must handle
    "uncertain" explicitly rather than assuming a numeric fallback."""
    if not _is_present(value):
        return "uncertain"
    try:
        v = float(value)
    except (TypeError, ValueError):
        return "uncertain"
    if v < 0.1:
        return "absent"
    if 0.3 <= v <= 0.7:
        return "het"
    if v > 0.8:
        return "hom"
    return "uncertain"  # falls in the 0.1-0.3 or 0.7-0.8 gap — genuinely ambiguous


def classify_segregation(proband_ab, mother_ab, father_ab) -> str:
    """Classify a trio's segregation pattern for one variant.

    Returns one of:
      "de_novo"           — proband het, both parents show AB<0.1 (absent)
      "maternal"           — proband het (or hemi/hom for XL), mother het, father absent
      "paternal"           — proband het (or hemi/hom for XL), father het, mother absent
      "both_carriers"       — proband hom, both parents het (classic AR)
      "homozygous_parent"   — either parent shows AB~1.0 (hom) for this variant
      "uncertain"           — parent data present but doesn't match a clean pattern
      "insufficient_data"   — mother and/or father AB not present at all

    Deliberately does not use the proband's own classification to gate the
    "insufficient_data" return — a trio with no parental AB is insufficient
    regardless of how confident the proband's own zygosity call is."""
    if not _is_present(mother_ab) or not _is_present(father_ab):
        return "insufficient_data"

    proband_class = classify_ab_ratio(proband_ab)
    mother_class  = classify_ab_ratio(mother_ab)
    father_class  = classify_ab_ratio(father_ab)

    if proband_class == "het" and mother_class == "absent" and father_class == "absent":
        return "de_novo"
    if proband_class == "hom" and mother_class == "het" and father_class == "het":
        return "both_carriers"
    if mother_class == "hom" or father_class == "hom":
        return "homozygous_parent"
    if mother_class == "het" and father_class == "absent":
        return "maternal"
    if father_class == "het" and mother_class == "absent":
        return "paternal"
    return "uncertain"


def classify_phase(segregation_a: str, segregation_b: str) -> str:
    """Deterministic CIS/TRANS phase check for two variants in the same gene,
    fed by classify_segregation() outputs. Promotes the previously LLM-only
    literal parent-of-origin string-comparison pattern (prompts/
    final_conclusion.txt STEP 1 / final_conclusion_revise.txt's TRANS STATUS
    CHECK) into a Python gate that can run BEFORE any LLM call, rather than
    late at final-synthesis time.

    Only "maternal"/"paternal" segregation results carry a determinable
    parent-of-origin — same parent for both variants -> "cis" (a hard block on
    treating the pair as compound heterozygous); different parents -> "trans".
    Any other combination (de_novo, both_carriers, homozygous_parent,
    uncertain, insufficient_data on either side) yields "unknown" — phase
    cannot be determined, callers should treat this as "may proceed with
    caveat", not as an automatic pass or fail."""
    origin_map = {"maternal": "mother", "paternal": "father"}
    origin_a = origin_map.get(segregation_a)
    origin_b = origin_map.get(segregation_b)
    if origin_a is None or origin_b is None:
        return "unknown"
    return "cis" if origin_a == origin_b else "trans"


def classify_xlinked_ab(proband_ab, mother_ab) -> str:
    """X-linked AB pattern for a chrX gene's variant.

      "XLR" — proband AB ~1.0 (hemizygous male, or homozygous female) AND
              mother AB ~0.5 (carrier) -> classic X-linked recessive pattern.
      "XLD" — proband AB ~0.5 (heterozygous) -> consistent with X-linked
              dominant, but this is the NUMERIC half only; confirming XLD
              additionally requires the mother being clinically AFFECTED,
              which is phenotype-text information this function has no
              access to — the moi_xlinked prompt must independently confirm
              that half before finalizing an XLD call.
      "uncertain" — neither pattern matches, or proband AB is unparseable."""
    proband_class = classify_ab_ratio(proband_ab)
    mother_class  = classify_ab_ratio(mother_ab)
    if proband_class == "hom" and mother_class == "het":
        return "XLR"
    if proband_class == "het":
        return "XLD"
    return "uncertain"
