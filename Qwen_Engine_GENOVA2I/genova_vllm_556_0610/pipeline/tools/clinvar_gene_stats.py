"""
pipeline/tools/clinvar_gene_stats.py — ClinVar gene-level P/LP consequence counts
and per-variant submission classification tally.

Two independent pieces of evidence, both from ClinVar:
  1. Gene-level: counts of Pathogenic/Likely pathogenic ClinVar variants per gene,
     split by molecular consequence (missense vs. nonsense/frameshift), to ground
     PP2/BP1 in concrete gene-level evidence instead of gene-name plausibility.
     Gene-scoped and cached at class level, same pattern as GnomadConstraintTool /
     LitVar2SummaryTool._cgd_table.
  2. Variant-level: for THIS specific variant, how many individual ClinVar
     submitters classified it Pathogenic / Likely pathogenic / VUS / Likely
     benign / Benign — the submission-level classification tally. Previously
     this tally was only ever fetched inside WebSearchAgentTool's forced check,
     and only when the CSV's own ClinVar_class field already read Pathogenic/
     Likely pathogenic — so a stale or wrong CSV label meant the real ClinVar
     submitter breakdown was never seen at all. Fetched unconditionally here
     instead, for every variant with a resolvable ClinVar record.

     When the tally shows genuine submitter disagreement (more than one
     distinct classification bucket represented — ClinVar's own definition of
     "conflicting"), each individual Pathogenic/Likely pathogenic submission's
     rationale (Comment text + cited PMIDs, from both the Classification-level
     and AttributeSet-level Citation elements) is additionally extracted and
     tagged "[mentions functional/experimental evidence]" when its comment
     text contains a functional-study marker (patient-derived cells, in
     vitro/in vivo assay, minigene, enzymatic activity, etc.) — this is the
     PS3-relevant signal a bare P/LP classification label cannot provide on
     its own (P/LP-by-label is not PS3; PS3 requires the underlying functional
     data). This mirrors NCBIFetchTool._fetch_clinvar_submissions() in
     ncbi.py (the ReAct agent's on-demand version of the same lookup) — kept
     as an independent copy here since this path runs unconditionally for
     every variant rather than only when the ReAct agent chooses to fetch a
     ClinVar URL, and previously only the aggregate tally (not this
     per-submission evidence) reached this always-on path — meaning PS3's
     grounding in conclusion.txt (which expects exactly this per-submission
     functional-evidence tagging) had no reliable source to read it from for
     the vast majority of variants.
"""

import logging
import re
import threading
import xml.etree.ElementTree as ET

from pipeline.tools.base import NetworkTool
from pipeline.tools.websearch import _ncbi_get, _clean_xml_text, DEFAULT_TIMEOUT
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError

logger = logging.getLogger(__name__)

# Normalizes whatever GermlineClassification text ClinVar submitters used into
# one of the 5 standard tally buckets. Same mapping as NCBIFetchTool's in
# ncbi.py (kept as an independent copy — same duplication pattern as
# _FUNCTIONAL_MARKERS below).
_CLASS_BUCKETS = {
    "pathogenic": "Pathogenic",
    "likely pathogenic": "Likely pathogenic",
    "uncertain significance": "VUS",
    "likely benign": "Likely benign",
    "benign": "Benign",
}
_TALLY_ORDER = ["Pathogenic", "Likely pathogenic", "VUS", "Likely benign", "Benign"]

# Words that mark a submitter <Comment> as carrying functional/experimental
# evidence (PS3-relevant) rather than a generic classification remark. Same
# list as NCBIFetchTool._FUNCTIONAL_MARKERS in ncbi.py (kept as an
# independent copy — this tool runs unconditionally per variant; that one
# only runs on-demand inside the ReAct agent).
_FUNCTIONAL_MARKERS = (
    "functional stud", "in vitro", "in vivo", "assay", "minigene",
    "splicing assay", "reporter assay", "enzymatic activity",
    "protein function", "functional assay", "functional analysis",
    "functional characterization", "experimentally", "patient-derived",
    "patient derived", "fibroblast",
)

# A consequence class is "predominant" only when the OTHER class makes up
# less than this fraction of all P/LP variants in the gene — i.e. the gene is
# near-exclusively one mechanism. Deliberately strict: a looser ratio-based
# rule (e.g. 2x) flagged genes as "predominant" from mixed evidence, which
# over-applied BP1/PP2 on borderline genes.
MINOR_CLASS_FRACTION_THRESHOLD = 0.05


def classify_consequence_counts(missense: int, nonsense: int) -> str:
    """Verdict string from gene-level P/LP missense vs. nonsense/frameshift
    counts. Shared by run() and by the pipeline's gene-evidence summary table
    so both stay on the same threshold.

    Deliberately does NOT say "(supports BP1)" / "(supports PP2)" — this is a
    GENE-level fact (same for every variant in the gene regardless of that
    variant's own Type), and phrasing it as "supports BP1" primed the SLM to
    apply BP1 even to nonsense/frameshift candidate variants (BP1 requires the
    CANDIDATE to be missense; a nonsense candidate in a truncating-predominant
    gene is a mechanism MATCH — PVS1 territory — not BP1 territory). Criterion
    applicability, including the variant-Type gate, is decided entirely in
    prompts/conclusion.txt, not asserted here.
    """
    total = missense + nonsense
    if total == 0:
        return "insufficient data — no P/LP missense or nonsense/frameshift variants found in ClinVar"
    if nonsense / total < MINOR_CLASS_FRACTION_THRESHOLD:
        return "missense-predominant"
    if missense / total < MINOR_CLASS_FRACTION_THRESHOLD:
        return "nonsense/frameshift-predominant"
    return "balanced — neither consequence class clearly predominates"


class ClinVarGeneStatsTool(NetworkTool):
    """
    Fetches gene-level counts of ClinVar Pathogenic/Likely pathogenic variants,
    split into missense vs. nonsense/frameshift, via NCBI esearch (count-only).

    gate():  runs when Gene is present.
    run():   two esearch calls per gene, cached at class level.
    """

    name        = "clinvar_gene_stats"
    description = (
        "Fetches gene-level ClinVar P/LP variant counts split by missense vs. "
        "nonsense/frameshift consequence — grounds PP2/BP1 in concrete evidence."
    )

    timeout: int = DEFAULT_TIMEOUT

    # Class-level cache: {"GENE": {"missense": int, "nonsense": int} | None}
    _stats_cache: dict[str, dict | None] = {}
    _cache_lock = threading.Lock()

    def gate(self, variant: dict, context: ToolContext) -> bool:
        gene = context.field("Gene")
        return gene != "NA" and gene.strip() != ""

    @staticmethod
    def _pathogenic_base(gene: str) -> str:
        # ClinVar's Properties field indexes clinical significance as a
        # "clinsig <value>" compound phrase, not the bare value — e.g.
        # "clinsig pathogenic"[Properties], not pathogenic[Properties]. The
        # latter returns esearch's "phrasesnotfound" and silently matches
        # zero records for every gene (verified live: LDLR — one of the most
        # heavily ClinVar-curated genes there is — returned 0/0 under the
        # bare-value query). Similarly "nonsense variant"[molecular
        # consequence] isn't an indexed phrase; the correct token is bare
        # "nonsense".
        return (
            f'{gene}[gene] AND ("clinsig pathogenic"[Properties] '
            f'OR "clinsig likely pathogenic"[Properties])'
        )

    def _esearch_count(self, term: str) -> int:
        try:
            data = _ncbi_get(
                "esearch.fcgi",
                {"db": "clinvar", "term": term, "retmode": "json", "retmax": 0},
                self.timeout,
            ).json()
            return int(data["esearchresult"]["count"])
        except Exception as e:
            raise ToolFetchError(f"ClinVar esearch failed for term={term!r}: {e}") from e

    def _fetch_stats(self, gene: str) -> dict:
        base = self._pathogenic_base(gene)
        missense_term = f'{base} AND "missense variant"[molecular consequence]'
        nonsense_term = (
            f'{base} AND (nonsense[molecular consequence] '
            f'OR "frameshift variant"[molecular consequence])'
        )
        try:
            missense = self._esearch_count(missense_term)
            nonsense = self._esearch_count(nonsense_term)
        except ToolFetchError:
            raise
        except Exception as e:
            raise ToolParseError(f"Failed to parse ClinVar counts for {gene}: {e}") from e
        return {"missense": missense, "nonsense": nonsense}

    def _get_stats(self, gene: str) -> dict:
        cache_key = gene.upper()
        if cache_key not in self._stats_cache:
            with self._cache_lock:
                if cache_key not in self._stats_cache:   # double-checked locking
                    self._stats_cache[cache_key] = self._fetch_stats(gene)
        return self._stats_cache[cache_key]

    # ── variant-level submission classification tally ────────────────────────

    # Extracts the cDNA-change token out of a combined/compound HGVS string,
    # e.g. "RS1:NM_000330:exon4:c.214G>A:p.E72K" -> "c.214G>A". ClinVar's own
    # esearch [variant name] index only matches this clean token, not a
    # colon-glued compound annotation (same fix already applied in ncbi.py's
    # resolve_clinvar_id for the same reason). "(" excluded too — an HGVS
    # string of the form "c.1292T>A(p.Val431Asp)" otherwise swallows the
    # trailing protein annotation into the token, producing an unmatchable
    # esearch term and a false "not resolvable" even when ClinVar has the
    # variant (verified live: LARS1 c.1292T>A, ClinVar Variation ID 431849 —
    # found instantly by ClinGenAlleleTool's genomic-HGVS route, but this
    # tool's own cDNA-token esearch silently failed on the untruncated token).
    _CDNA_CHANGE_RE = re.compile(r"c\.[^\s:;()]+")

    def _resolve_variation_id(self, gene: str, hgvs: str) -> str | None:
        cdna_match = self._CDNA_CHANGE_RE.search(hgvs) if hgvs and hgvs != "NA" else None
        variant_term = cdna_match.group(0) if cdna_match else hgvs
        term = f"{gene}[gene] AND {variant_term}[variant name]" if variant_term and variant_term != "NA" else None
        if not term:
            return None
        try:
            data = _ncbi_get(
                "esearch.fcgi",
                {"db": "clinvar", "term": term, "retmode": "json", "retmax": 1},
                self.timeout,
            ).json()
            ids = data.get("esearchresult", {}).get("idlist", [])
        except Exception as e:
            logger.debug("ClinVar variation-ID resolution failed for %s %s: %s", gene, hgvs, e)
            return None
        return ids[0] if ids else None

    def _fetch_classification_tally(self, variation_id: str) -> dict | None:
        """
        Returns {"counts": {bucket: n, ...}, "pl_evidence": [line, ...]} or None.

        pl_evidence has one entry per individual Pathogenic/Likely-pathogenic
        submission — submitter, SCV accession, cited PMIDs (both
        Classification-level and AttributeSet-level Citation elements, since
        submitters use either or both), and the "[mentions functional/
        experimental evidence]" tag when the Comment text matches a
        _FUNCTIONAL_MARKERS keyword. Always populated regardless of whether
        the tally turns out conflicting — run() decides what to surface.
        """
        try:
            xml = _ncbi_get(
                "efetch.fcgi",
                {"db": "clinvar", "id": variation_id, "rettype": "vcv",
                 "is_variationid": "true", "retmode": "xml"},
                self.timeout,
            ).text
            root = ET.fromstring(xml)
        except Exception as e:
            logger.debug("ClinVar submission fetch failed for variation %s: %s", variation_id, e)
            return None

        va = root.find(".//VariationArchive")
        cr = va.find("ClassifiedRecord") if va is not None else None
        cal = cr.find("ClinicalAssertionList") if cr is not None else None
        assertions = cal.findall("ClinicalAssertion") if cal is not None else []
        if not assertions:
            return None

        counts: dict[str, int] = {}
        pl_evidence: list[str] = []
        for ca in assertions:
            cl = ca.find("Classification")
            if cl is None:
                continue
            desc_el = cl.find("GermlineClassification")
            desc = (desc_el.text or "").strip() if desc_el is not None and desc_el.text else "Not provided"
            bucket = _CLASS_BUCKETS.get(desc.lower(), desc)
            counts[bucket] = counts.get(bucket, 0) + 1

            if desc.lower() not in ("pathogenic", "likely pathogenic"):
                continue

            acc = ca.find("ClinVarAccession")
            submitter = acc.get("SubmitterName", "Unknown submitter") if acc is not None else "Unknown submitter"
            scv = acc.get("Accession", "") if acc is not None else ""

            comment_el = cl.find("Comment")
            comment = _clean_xml_text(comment_el.text) if comment_el is not None and comment_el.text else ""

            # Classification-level Citation only. A ClinicalAssertion's own
            # top-level AttributeSet/Citation is paired with an
            # Attribute Type="AssertionMethod" (e.g. "ACMG Guidelines, 2015")
            # — it cites the classification METHODOLOGY paper (Richards et al.
            # 2015, PMID 25741868), not variant-specific evidence. Verified
            # live: that PMID appeared on every submission in a real record
            # regardless of actual content, confirming it is method boilerplate,
            # not evidence — do not pull citations from that level.
            pmids = [c.find("ID").text for c in cl.findall("Citation")
                     if c.find("ID") is not None and c.find("ID").get("Source") == "PubMed"]

            line = f"- [{desc}] source: {submitter} ({scv})"
            if pmids:
                line += f" — cites PMID: {', '.join(pmids[:8])}"
            if comment:
                tag = " [mentions functional/experimental evidence]" if \
                    any(m in comment.lower() for m in _FUNCTIONAL_MARKERS) else ""
                line += f"{tag}\n  Rationale: {comment[:600]}"
            pl_evidence.append(line)

        return {"counts": counts, "pl_evidence": pl_evidence}

    def run(self, variant: dict, context: ToolContext) -> str | None:
        gene = context.field("Gene")
        stats = self._get_stats(gene)
        missense = stats["missense"]
        nonsense = stats["nonsense"]

        verdict = classify_consequence_counts(missense, nonsense)

        gene_block = (
            f"CLINVAR GENE-LEVEL P/LP VARIANT COUNTS ({gene}):\n"
            f"P/LP missense variants   : {missense}\n"
            f"P/LP nonsense/frameshift : {nonsense}\n"
            f"-> {verdict}"
        )

        hgvs = context.field("HGVS")
        variation_id = self._resolve_variation_id(gene, hgvs)
        result = self._fetch_classification_tally(variation_id) if variation_id else None

        if result is None:
            variant_block = (
                "CLINVAR VARIANT-LEVEL SUBMISSION TALLY:\n"
                "Not resolvable — no ClinVar record found for this specific variant "
                "(or no individual submissions listed)."
            )
        else:
            tally = result["counts"]
            total = sum(tally.values())
            known_lines = "\n".join(
                f"{label:<18}: {tally.get(label, 0)}" for label in _TALLY_ORDER
            )
            other = {k: v for k, v in tally.items() if k not in _TALLY_ORDER}
            other_line = (
                f"\nOther/unrecognized: {', '.join(f'{k}={v}' for k, v in other.items())}"
                if other else ""
            )
            variant_block = (
                f"CLINVAR VARIANT-LEVEL SUBMISSION TALLY (variation ID {variation_id}, "
                f"{total} individual submissions):\n"
                f"{known_lines}{other_line}"
            )

            # "Conflicting" per ClinVar's own definition: more than one distinct
            # classification bucket represented among individual submitters.
            # Only when genuinely conflicting do we pay for the deeper per-P/LP
            # evidence dig — a clean unanimous call doesn't need it.
            distinct_buckets = sum(1 for n in tally.values() if n > 0)
            if distinct_buckets > 1 and result["pl_evidence"]:
                variant_block += (
                    f"\n\nCLINVAR CONFLICTING — PATHOGENIC/LIKELY PATHOGENIC SUBMISSION "
                    f"EVIDENCE REVIEW (variation ID {variation_id}, {total} total "
                    f"submissions, {distinct_buckets} distinct classifications — "
                    f"submitters disagree, so each P/LP call's own evidentiary basis "
                    f"is broken out below rather than trusting the aggregate label):\n"
                    + "\n".join(result["pl_evidence"][:10])
                )
            elif distinct_buckets > 1 and not result["pl_evidence"]:
                variant_block += (
                    "\n\nCLINVAR CONFLICTING: submitters disagree on classification, "
                    "but no individual submission was itself Pathogenic/Likely "
                    "pathogenic (the P/LP portion of the aggregate, if any, traces to "
                    "expert panel review rather than a single traceable submission)."
                )

        return f"{gene_block}\n\n{variant_block}"
