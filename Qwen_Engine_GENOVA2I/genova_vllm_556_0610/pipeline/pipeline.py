"""
pipeline/pipeline.py — Top-level orchestrator.

Wires together:
  - LLM client (vLLM in production, HF in dev)
  - Tool instances
  - Manifest loader
  - Executor
  - Stage functions (retrieval → first_triage → reasoning → cross_analysis → conclusion → final_conclusion)

Design rule: server.py calls pipeline.run() and knows nothing about variants.
All logic lives here and in the stages/tools/core modules.

Public API:
    Pipeline(model_name, llm_kwargs)   — create and configure
    Pipeline.run(variants, phenotype)  — execute the full pipeline, return report string

Per-variant flow:

  retrieval:         variants → list[str]  (one context slice per variant)
  first_triage:      per slice → KEEP/DISCARD  (skipped if n <= TRIAGE_ENABLED_THRESHOLD)
  reasoning:         kept slices → dict[int, str]  (one reasoning block per kept variant)
  cross_analysis:    per gene with ≥2 tier-1 variants → dict[gene, str]
  conclusion:        tier-1 slices + reasoning + cross_analysis → dict[int, str]
  final_conclusion:  tier-1 conclusions → str  (the # Clinical Conclusion paragraph)
"""

from __future__ import annotations

import re
import logging
import itertools
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional

MAX_WORKERS          = 32   # retrieval (network tools, I/O bound)
MAX_WORKERS_LLM      = 16   # reasoning/conclusion (vLLM calls, GPU-bound)

from pipeline.config          import (
    DEFAULT_GENOME_BUILD,
    TRIAGE_ENABLED_THRESHOLD,
)
from pipeline.core.manifest  import ManifestLoader
from pipeline.core.executor  import Executor
from pipeline.core.normalizer import TARGET_COLUMNS
from pipeline.core            import moi
from pipeline.core            import segregation
from pipeline.core            import acmg_sf
from pipeline.llm.registry   import get_client
from pipeline.tools          import AutoPVS1Tool, LitVar2SummaryTool, SpliceAITool, WebSearchAgentTool
from pipeline.stages         import (
    retrieval, reasoning, conclusion, cross_analysis, final_conclusion, first_triage,
    moi_denovo, moi_dominant, moi_recessive, moi_xlinked, actionable,
)

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════════
# VARIANT STRING → DICT CONVERSION
# ═══════════════════════════════════════════════════════════════════════════════

def _parse_variant_str(variant_str: str) -> dict:
    """
    Parse a canonical key=value variant string into a dict.

    The normalizer produces strings of the form:
        "Variant=chr6:100896130 T>C, Chromosome=chr6, Position=100896130, ..."

    We extract each TARGET_COLUMN by matching:
        <ColumnName>=<value>
    where value runs up to the next ", <NextColumnName>=" or end of string.

    This is more robust than a naive split on ", " because field values may
    themselves contain commas (e.g. OMIM_phenotype).
    """
    result: dict[str, str] = {}
    for col in TARGET_COLUMNS:
        # Pattern: "Col=<capture>" stopped at ", NextCol=" or end of string
        m = re.search(
            rf"(?:^|,\s*){re.escape(col)}=(.*?)(?=,\s*(?:{'|'.join(re.escape(c) for c in TARGET_COLUMNS)})=|$)",
            variant_str,
        )
        result[col] = m.group(1).strip() if m else "NA"
    return result


# ═══════════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

# Matches the shared PATIENT DATA … VARIANT DATA:\n\n header within each slice
_SLICE_HEADER_RE = re.compile(r"^(PATIENT DATA:.*?VARIANT DATA:\n\n)", re.DOTALL)

# Matches the classification named on the conclusion's "ACMG points: N → Classification" line.
_ACMG_POINTS_LINE_RE = re.compile(r"ACMG points:?\*{0,2}\s*[^\n→]*→\s*([A-Za-z /()]+)", re.IGNORECASE)

# Matches the numeric point value on that same line (e.g. "ACMG points: 8 →" -> 8).
# Used to pull the base ACMG score out as a number so downstream MOI layers can
# compute base+delta without re-parsing prose or trusting the LLM's own arithmetic
# across two separate documents (base conclusion + layer-specific addition).
_ACMG_POINTS_VALUE_RE = re.compile(r"ACMG points:?\*{0,2}\s*(-?\d+)", re.IGNORECASE)


def _extract_acmg_points_value(conclusion_text: str) -> int | None:
    """Numeric base ACMG point total from a conclusion block, or None if the
    line is missing/unparseable (caller should treat that as unknown, not 0)."""
    m = _ACMG_POINTS_VALUE_RE.search(conclusion_text)
    if not m:
        return None
    return int(m.group(1))

# Checked most-specific-first so "likely pathogenic"/"likely benign" don't match
# the "pathogenic"/"benign" substring check meant for the standalone terms.
_ACMG_CLASSIFICATION_RANK = [
    ("likely pathogenic", 1),
    ("pathogenic", 0),
    ("uncertain significance", 2),
    ("vus", 2),
    ("likely benign", 3),
    ("benign", 4),
]


def _classification_rank(conclusion_text: str) -> int:
    """P=0, LP=1, VUS=2, LB=3, B=4; unrecognized/missing sorts last."""
    m = _ACMG_POINTS_LINE_RE.search(conclusion_text)
    if not m:
        return len(_ACMG_CLASSIFICATION_RANK) + 1
    label = m.group(1).strip().lower()
    for key, rank in _ACMG_CLASSIFICATION_RANK:
        if key in label:
            return rank
    return len(_ACMG_CLASSIFICATION_RANK) + 1


def _extract_classification_label(conclusion_text: str) -> str | None:
    """The classification name off a conclusion's 'ACMG points: N → Classification' line."""
    m = _ACMG_POINTS_LINE_RE.search(conclusion_text)
    return m.group(1).strip() if m else None


def _group_by_gene(variants: list[dict]) -> dict[str, list[int]]:
    """
    Return a mapping of gene symbol → list of 0-based variant indices.

    Variants with Gene == "NA" or missing Gene are excluded (they cannot
    participate in gene-level cross-analysis).
    """
    groups: dict[str, list[int]] = defaultdict(list)
    for i, variant in enumerate(variants):
        gene = variant.get("Gene", "NA")
        if gene and gene != "NA":
            groups[gene].append(i)
    return dict(groups)


def _reconstruct_full_context(context_slices: list[str]) -> str:
    """
    Reconstruct the full augmented context string from per-variant slices.

    Used only for display in the AUGMENTED CONTEXT section of the final output.
    Each slice contains a PATIENT DATA header + one VARIANT block; this helper
    emits the header once, then concatenates the VARIANT blocks from all slices.
    """
    if not context_slices:
        return ""

    m = _SLICE_HEADER_RE.match(context_slices[0])
    if not m:
        # Fallback: concatenate slices separated by a blank line
        return "\n\n".join(s.rstrip() for s in context_slices) + "\n"

    header     = m.group(1)
    header_len = len(header)

    variant_sections = [s[header_len:].rstrip() for s in context_slices]
    return header + "\n\n".join(variant_sections) + "\n"


def _build_process_details(
    process_log: list[dict],
    variants: list[dict],
) -> str:
    """
    Build the PROCESS DETAILS section from the executor's process log.
    One block per variant, one entry per tool.
    """
    SEP_VARIANT = "─" * 56
    lines = []

    by_variant: dict[int, list[dict]] = defaultdict(list)
    for entry in process_log:
        by_variant[entry["variant_index"]].append(entry)

    for i, variant in enumerate(variants):
        gene  = variant.get("Gene", "?")
        hgvs  = variant.get("HGVS", variant.get("Variant", "?"))
        lines.append(f"\n{SEP_VARIANT}")
        lines.append(f"VARIANT {i + 1} — {gene}  {hgvs}")
        lines.append(SEP_VARIANT)

        entries = by_variant.get(i, [])
        if not entries:
            lines.append("  (no tool entries recorded)")
            continue

        for entry in entries:
            tool_name = entry["tool_name"].upper()
            gate      = entry["gate"]

            if gate == "SKIP":
                reason = entry.get("gate_reason") or ""
                lines.append(f"\n[{tool_name}]  GATE: SKIPPED  ({reason})")
                continue

            lines.append(f"\n[{tool_name}]  GATE: PASS")

            metadata = entry.get("metadata") or {}

            # ── LitVar2 metadata ────────────────────────────────────────
            if "query" in metadata:
                query   = metadata.get("query") or "?"
                n_found = metadata.get("n_found") or 0
                kept    = metadata.get("kept") or []
                lines.append(f"  Query     : {query}")
                lines.append(f"  Articles  : {len(kept)} kept from {n_found} found")
                if kept:
                    lines.append("  Kept articles:")
                    for art in kept:
                        lines.append(f"    [PMID:{art['pmid']}] {art['title']}")

            # ── WebSearch trace ──────────────────────────────────────────
            if "trace" in metadata:
                trace = metadata.get("trace") or []
                if trace:
                    lines.append("  ReAct trace:")
                    for step in trace:
                        lines.append(f"    {step}")

            # ── Raw output ───────────────────────────────────────────────
            raw = entry.get("raw_output")
            if raw:
                lines.append("  Raw output:")
                for raw_line in raw.splitlines():
                    lines.append(f"    {raw_line}")
            else:
                lines.append("  Output: (none)")

    return "\n".join(lines)




# ═══════════════════════════════════════════════════════════════════════════════
# PIPELINE
# ═══════════════════════════════════════════════════════════════════════════════

class Pipeline:
    """
    Full variant analysis pipeline.

    Instantiate once at server startup; call run() per request.

    Args:
        model_name: Key in pipeline.llm.registry.MODELS (default "qwen3.5-9b").
        llm_kwargs: Forwarded to get_client() — e.g. base_url for VLLMClient.
    """

    def __init__(
        self,
        model_name: str = "qwen3.5-9b",
        llm_kwargs: Optional[dict] = None,
        llm=None,                              # pre-built LLMClient (overrides model_name)
    ):
        self._model_name = model_name
        self._llm_kwargs = llm_kwargs or {}

        # ── LLM client (loaded once at startup) ──────────────────────────────
        if llm is not None:
            logger.info("[Pipeline] Using pre-built LLM client: %s", type(llm).__name__)
            self._llm = llm
        else:
            logger.info("[Pipeline] Loading LLM client: %s", model_name)
            self._llm = get_client(model_name, **self._llm_kwargs)

        # ── Manifests ─────────────────────────────────────────────────────────
        loader = ManifestLoader()
        self._manifests = loader.load_all()
        logger.info(
            "[Pipeline] Loaded %d manifests: %s",
            len(self._manifests),
            [m["name"] for m in self._manifests],
        )

        # ── Tool instances (one per pipeline tool) ────────────────────────────
        self._tools = [
            LitVar2SummaryTool(),
            SpliceAITool(),
            AutoPVS1Tool(),
            WebSearchAgentTool(),
        ]

        # ── Executor ──────────────────────────────────────────────────────────
        self._executor = Executor(
            llm=self._llm,
            tools=self._tools,
            manifests=self._manifests,
        )
        logger.info("[Pipeline] Executor ready with %d tools.", len(self._executor._paired))

    # ── public ────────────────────────────────────────────────────────────────

    async def run(
        self,
        variants: list[dict],
        patient_phenotype: str,
        genome_build: str = DEFAULT_GENOME_BUILD,
        raw_rows: list[dict] | None = None,
        parental_ab: list[dict] | None = None,
        header_mapping_summary: str | None = None,
    ) -> str:
        """
        Execute the full per-variant pipeline and return the combined output.

        Pipeline flow:
            Stage 1   — retrieval:        variants → list[str] (one context slice per variant)
            Triage    — (if n > threshold) kept/discarded per variant
            Stage 2   — reasoning:        kept slices → dict[int, str]
            Tier select — split kept variants into included/excluded via inclusion_decisions
            Stage 3.5 — cross_analysis:   per gene with ≥2 tier-1 variants → dict[gene, str]
            Stage 4   — conclusion:        tier-1 variants → dict[int, str]
            Stage 5   — final_conclusion:  tier-1 conclusions → str

        When n <= TRIAGE_ENABLED_THRESHOLD the pipeline behaves identically to
        the no-triage path: all variants go through all stages, and the
        Tier 2/Tier 3 appendix sections are omitted from the output.

        Args:
            variants:          List of normalized variant dicts from normalize_upload().
            patient_phenotype: Free-text patient phenotype string from the request.
            genome_build:      Reference genome build for this run ("hg19" or "hg38").
            raw_rows:          Full original CSV rows, one dict per variant.
            parental_ab:       Per-variant allelic balance dicts from normalizer,
                               or None if no parental data available.
            header_mapping_summary: Human-readable text from normalizer.normalize_upload()
                               showing what the SLM understood the CSV header to mean —
                               surfaced in the final output as its own section, or None.

        Returns:
            Combined output string with PROCESS DETAILS, AUGMENTED CONTEXT,
            REASONING, and FINAL REPORT sections separated by === banners.

        Raises:
            ValueError if variants is empty.
        """
        if not variants:
            raise ValueError("variants list is empty — nothing to analyze.")

        n = len(variants)
        logger.info("[Pipeline] Processing %d variants.", n)

        SEP         = "=" * 60
        SEP_VARIANT = "─" * 56

        # ── Stage 1: Retrieval ────────────────────────────────────────────────
        context_slices: list[str] = retrieval.run(
            variants=variants,
            patient_phenotype=patient_phenotype,
            executor=self._executor,
            genome_build=genome_build,
            raw_rows=raw_rows,
            parental_ab=parental_ab,
        )

        process_details = _build_process_details(self._executor.process_log, variants)

        # ── ACMG Secondary Findings (SF v3.2) actionable-gene detection ─────────
        # Computed on ALL variants (not just kept ones) right after retrieval, so
        # it can force-exempt flagged variants from triage/second-triage discard
        # below — a P/LP variant in an SF gene must be reported regardless of
        # phenotype fit, so it must survive both filtering stages irrespective of
        # what the phenotype-driven KEEP/INCLUDE machinery would otherwise decide.
        litvar2_raw_by_variant: dict[int, str | None] = {
            entry["variant_index"]: entry.get("raw_output")
            for entry in self._executor.process_log
            if entry["tool_name"] == "litvar2_summary" and entry["gate"] == "PASS"
        }
        actionable_indices, actionable_reasons = acmg_sf.build_actionable_set(
            variants=variants,
            litvar2_raw_by_variant=litvar2_raw_by_variant,
            llm=self._llm,
        )
        if actionable_indices:
            logger.info(
                "[Pipeline] ACMG SF actionable variant(s) flagged: %s",
                {i + 1: variants[i].get("Gene", "?") for i in actionable_indices},
            )

        # ── Triage ────────────────────────────────────────────────────────────
        # triage_results maps original variant index → ("KEEP"|"DISCARD", justification)
        triage_results: dict[int, tuple[str, str]] = {}
        triage_ran = n > TRIAGE_ENABLED_THRESHOLD

        if triage_ran:
            # Identify multi-variant genes (compound het candidates) — exempt from discard
            gene_groups_all = _group_by_gene(variants)
            multi_gene_indices: set[int] = set()
            for gene, idxs in gene_groups_all.items():
                if len(idxs) >= 2:
                    multi_gene_indices.update(idxs)

            def _triage_one(i: int) -> tuple[int, str, str]:
                decision, justification = first_triage.run_one(
                    variant_context=context_slices[i],
                    patient_phenotype=patient_phenotype,
                    llm=self._llm,
                )
                return i, decision, justification

            workers = min(MAX_WORKERS, n)
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = {pool.submit(_triage_one, i): i for i in range(n)}
                for future in as_completed(futures):
                    i, decision, justification = future.result()
                    gene = variants[i].get("Gene", "?")
                    if decision == "DISCARD" and i in actionable_indices:
                        logger.info(
                            "[Pipeline] Triage override: variant %d (%s) is an ACMG SF "
                            "actionable finding — forced KEEP", i + 1, gene,
                        )
                        justification = (
                            f"[ACMG SF actionable exempt — original triage: DISCARD: {justification}]"
                        )
                        decision = "KEEP"
                    elif decision == "DISCARD" and i in multi_gene_indices:
                        logger.info(
                            "[Pipeline] Triage override: variant %d is compound-het "
                            "candidate in gene %s — forced KEEP", i + 1, gene,
                        )
                        justification = (
                            f"[compound-het exempt — original triage: DISCARD: {justification}]"
                        )
                        decision = "KEEP"
                    triage_results[i] = (decision, justification)
        else:
            # No triage — all variants kept
            for i in range(n):
                triage_results[i] = ("KEEP", "")

        kept_indices     = [i for i, (dec, _) in triage_results.items() if dec == "KEEP"]
        discarded_indices = [i for i, (dec, _) in triage_results.items() if dec == "DISCARD"]
        logger.info(
            "[Pipeline] Triage complete: %d kept, %d discarded.",
            len(kept_indices), len(discarded_indices),
        )

        # ── Compound-het gene grouping (AR / AD-AR genes with ≥2 kept variants) ─
        # Computed before reasoning (Stage 2a) so that sibling variant evidence
        # can be injected into each variant's FIRST reasoning call, not just
        # second_triage — Call 1 should already weigh the variant jointly with
        # its gene-mate instead of reasoning about it in isolation.
        # Scoped to recessive-relevant inheritance only — a second variant in a
        # purely dominant gene doesn't change the first variant's standing.
        # Inheritance-mode classification and compound-het gene grouping live in
        # pipeline/core/moi.py (relocated so the MOI-layer stage modules can
        # import this logic directly) — same behavior, just no longer inline here.
        kept_set = set(kept_indices)
        gene_mode_cache = moi.build_gene_mode_cache(
            variants, kept_indices, context_slices, _group_by_gene,
        )
        recessive_gene_groups = moi.build_recessive_gene_groups(
            variants, gene_mode_cache, kept_indices, _group_by_gene,
        )

        # Layer 1 (MOI-layer restructuring): per-variant phenotype pertinence,
        # sourced from the LitVar2 evidence already in each context slice — not
        # yet consumed by any scoring stage (that wiring lands with the Layer 2
        # base-ACMG stage, where PVS1's phenotype-fit gate will read this value
        # as a backend-determined fact instead of re-deriving it from prose).
        pertinence_cache = moi.build_pertinence_cache(variants, kept_indices, context_slices)
        logger.debug(
            "[Pipeline] Pertinence cache: %s",
            {variants[i].get("Gene", "?"): v for i, v in pertinence_cache.items()},
        )

        def _inheritance_mode_block(i: int) -> str:
            """Backend-determined gene inheritance mode for variant i, handed to the
            reasoning prompt as a stated fact so the model reports it rather than
            re-deriving (and possibly mis-deriving) it from prose evidence."""
            gene  = variants[i].get("Gene", "NA")
            mode  = gene_mode_cache.get(gene, "")
            label = moi.MODE_LABELS.get(mode, moi.MODE_LABELS[""])
            return f"\n--- GENE INHERITANCE MODE (backend-determined) ---\nGene: {gene} | Mode: {label}\n"

        def _pertinence_block(i: int) -> str:
            """Backend-determined gene-phenotype literature pertinence for
            variant i, appended to the reasoning text handed into the ACMG
            base-scoring stage (conclusion.py) as a stated fact — a hard
            negative constraint on Phenotype fit/PVS1 rather than letting the
            model claim "match" from prose alone when the backend LitVar2
            search found no gene-phenotype link at all. Injected via the
            existing `reasoning` argument (not a new run_one() parameter) so
            server_qwen.py's direct-mode monkeypatch of conclusion.run_one,
            which wraps the exact (variant_context, reasoning, cross_analysis,
            llm) signature for SSE progress ticks, keeps working unmodified."""
            gene  = variants[i].get("Gene", "NA")
            value = pertinence_cache.get(i)
            label = moi.PERTINENCE_LABELS[value]
            return f"\n--- GENE-PHENOTYPE LITERATURE PERTINENCE (backend-determined) ---\n{label}\n"

        def _sibling_evidence_block(i: int) -> str:
            """Sibling block built from raw retrieval evidence (Call 1 — no reasoning
            exists yet). Skipped only when this variant is a CONFIRMED homozygous
            call in an AR/XLR gene — that already explains the disease on its own
            and doesn't need a compound-het partner. Zygosity is otherwise left to
            the model (it sees the ALLELIC BALANCE block and is instructed to
            derive zygosity from it when Zygosity is NA)."""
            gene = variants[i].get("Gene", "NA")
            if moi.zygosity_is_confirmed_hom(variants[i].get("Zygosity", "")):
                return ""
            siblings = [j for j in recessive_gene_groups.get(gene, []) if j != i]
            if not siblings:
                return ""
            parts = ["\n--- SIBLING VARIANT(S) IN SAME GENE (autosomal recessive / X-linked recessive) ---"]
            for j in siblings:
                sib = variants[j]
                m = _SLICE_HEADER_RE.match(context_slices[j])
                sib_evidence = context_slices[j][m.end():].rstrip() if m else context_slices[j].rstrip()
                parts.append(
                    f"\nSibling Variant {j + 1} — {sib.get('Gene', '?')} "
                    f"{sib.get('HGVS', sib.get('Variant', '?'))} "
                    f"(zygosity: {sib.get('Zygosity', 'NA')})\n"
                    f"Sibling evidence:\n{sib_evidence}"
                )
            return "\n".join(parts) + "\n"

        # ── Stage 2a: Reasoning (only on kept variants) ───────────────────────
        # Split into its own wave (rather than one combined reasoning+second_triage
        # call per variant) so that, for compound-het candidate genes, every kept
        # variant's reasoning is available before any of them proceeds to
        # second_triage — second_triage needs to see sibling variants' evidence.
        reasoning_only: dict[int, str] = {}
        reasoning_failed: set[int] = set()

        def _reason_one(i: int) -> tuple[int, str]:
            sib_block  = _sibling_evidence_block(i)
            mode_block = _inheritance_mode_block(i)
            return i, reasoning.run_reasoning(
                variant_context=context_slices[i], llm=self._llm,
                sibling_context_block=sib_block,
                inheritance_mode_block=mode_block,
            )

        workers = min(MAX_WORKERS_LLM, len(kept_indices))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_reason_one, i): i for i in kept_indices}
            for future in as_completed(futures):
                i = futures[future]
                gene = variants[i].get("Gene", "?")
                try:
                    i, r = future.result()
                except Exception as exc:
                    logger.error(
                        "[Pipeline] Reasoning failed for variant %d (%s): %s — marking EXCLUDE",
                        i + 1, gene, exc,
                    )
                    reasoning_only[i] = f"[REASONING FAILED: {exc}]"
                    reasoning_failed.add(i)
                    continue
                reasoning_only[i] = r

        def _sibling_block(i: int) -> str:
            """Sibling block built from sibling reasoning text (Call 2 — post Call 1).
            Same confirmed-hom-only gate as _sibling_evidence_block."""
            gene = variants[i].get("Gene", "NA")
            if moi.zygosity_is_confirmed_hom(variants[i].get("Zygosity", "")):
                return ""
            siblings = [
                j for j in recessive_gene_groups.get(gene, [])
                if j != i and j not in reasoning_failed
            ]
            if not siblings:
                return ""
            parts = ["\n--- SIBLING VARIANT(S) IN SAME GENE (autosomal recessive / X-linked recessive) ---"]
            for j in siblings:
                sib = variants[j]
                parts.append(
                    f"\nSibling Variant {j + 1} — {sib.get('Gene', '?')} "
                    f"{sib.get('HGVS', sib.get('Variant', '?'))} "
                    f"(zygosity: {sib.get('Zygosity', 'NA')})\n"
                    f"Sibling reasoning:\n{reasoning_only[j]}"
                )
            return "\n".join(parts) + "\n"

        # ── Stage 2b: Second triage (INCLUDE/EXCLUDE, with sibling context) ────
        reasonings: dict[int, str] = {}
        inclusion_decisions: dict[int, str] = {}
        second_triage_justifications: dict[int, str] = {}

        for i in reasoning_failed:
            reasonings[i] = reasoning_only[i]
            inclusion_decisions[i] = "EXCLUDE"
            second_triage_justifications[i] = "reasoning error"

        triage_targets = [i for i in kept_indices if i not in reasoning_failed]

        def _second_triage_one(i: int) -> tuple[int, str, str, str]:
            sib_block = _sibling_block(i)
            combined = reasoning.run_second_triage(
                variant_context=context_slices[i],
                reasoning_text=reasoning_only[i],
                llm=self._llm,
                sibling_context_block=sib_block,
            )
            decision, justification = reasoning.parse_inclusion_decision(combined)
            return i, combined, decision, justification

        workers = min(MAX_WORKERS_LLM, len(triage_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_second_triage_one, i): i for i in triage_targets}
            for future in as_completed(futures):
                i = futures[future]
                gene = variants[i].get("Gene", "?")
                try:
                    i, combined, decision, justification = future.result()
                except Exception as exc:
                    logger.error(
                        "[Pipeline] Second triage failed for variant %d (%s): %s — marking EXCLUDE",
                        i + 1, gene, exc,
                    )
                    reasonings[i] = reasoning_only[i] + f"\n\n[SECOND TRIAGE FAILED: {exc}]"
                    inclusion_decisions[i] = "EXCLUDE"
                    second_triage_justifications[i] = f"second triage error: {exc}"
                    continue
                reasonings[i] = combined
                inclusion_decisions[i] = decision
                second_triage_justifications[i] = justification
                logger.info(
                    "[Pipeline] Variant %d (%s) second triage → %s", i + 1, gene, decision
                )

        # ── ACMG SF override: force INCLUDE for actionable variants that reached
        #     a second-triage decision (reasoning_failed ones are left EXCLUDEd —
        #     there is no reasoning text to build a conclusion/report block from). ─
        for i in actionable_indices:
            if i in reasoning_failed or i not in kept_indices:
                continue
            if inclusion_decisions.get(i) != "INCLUDE":
                logger.info(
                    "[Pipeline] Second-triage override: variant %d (%s) is an ACMG SF "
                    "actionable finding — forced INCLUDE", i + 1, variants[i].get("Gene", "?"),
                )
                second_triage_justifications[i] = (
                    f"[ACMG SF actionable exempt — original: {inclusion_decisions.get(i)}: "
                    f"{second_triage_justifications.get(i, '')}]"
                )
                inclusion_decisions[i] = "INCLUDE"

        # ── Inclusion split (direct from second_triage decisions) ────────────
        include_indices = [i for i in kept_indices if inclusion_decisions.get(i) == "INCLUDE"]
        exclude_indices = [i for i in kept_indices if inclusion_decisions.get(i) == "EXCLUDE"]

        logger.info(
            "[Pipeline] Second triage: %d included, %d excluded.",
            len(include_indices), len(exclude_indices),
        )

        # ── Stage 3.5: Gene-level cross-analysis (included variants only) ──────
        gene_groups_included: dict[str, list[int]] = defaultdict(list)
        for i in include_indices:
            gene = variants[i].get("Gene", "NA")
            if gene and gene != "NA":
                gene_groups_included[gene].append(i)

        cross_analyses: dict[str, str] = {}
        for gene, indices in gene_groups_included.items():
            if len(indices) >= 2:
                logger.info(
                    "[Pipeline] Cross-analysis for gene %s (%d variants)...",
                    gene, len(indices),
                )
                cross_analyses[gene] = cross_analysis.run(
                    gene=gene,
                    variant_contexts=[context_slices[j] for j in indices],
                    reasonings=[reasonings[j] for j in indices],
                    llm=self._llm,
                )

        # ── Stage 4: Per-variant conclusion (ALL kept variants — ACMG score is
        #     forced for every evaluated variant, not just INCLUDEd ones, so
        #     excluded variants still carry a documented ACMG classification in
        #     the report appendix for audit purposes) ──────────────────────────
        conclusion_targets = [i for i in kept_indices if i not in reasoning_failed]
        conclusions: dict[int, str] = {}
        base_acmg_points: dict[int, int | None] = {}
        def _conclude_one(i: int) -> tuple[int, str]:
            gene = variants[i].get("Gene", "NA")
            ca   = cross_analyses.get(gene) if gene != "NA" else None
            # Pertinence appended to the reasoning text (not a new run_one()
            # param) — see _pertinence_block's docstring for why.
            reasoning_with_pertinence = reasonings[i] + _pertinence_block(i)
            return i, conclusion.run_one(
                variant_context=context_slices[i],
                reasoning=reasoning_with_pertinence,
                cross_analysis=ca,
                llm=self._llm,
            )

        workers = min(MAX_WORKERS_LLM, len(conclusion_targets))
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_conclude_one, i): i for i in conclusion_targets}
            for future in as_completed(futures):
                i, conc = future.result()
                conclusions[i] = conc
                base_acmg_points[i] = _extract_acmg_points_value(conc)
                logger.info(
                    "[Pipeline] Conclusion done for variant %d (%s), base ACMG points=%s.",
                    i + 1, variants[i].get("Gene", "?"), base_acmg_points[i],
                )

        # ── Stage 4.5: MOI-specific layers (3: de novo, 4: dominant-inherited,
        #     5: recessive/compound-het, 6: X-linked) — each ADDS criteria on
        #     top of the Layer 2 base score above; scoped to include_indices
        #     (Tier 1), the same population the flat report used to cover. A
        #     variant can appear in more than one layer if its gene's MOI
        #     qualifies for more than one (e.g. AD_AR genes appear in both the
        #     dominant and recessive layers) — deliberately not deduplicated. ──
        include_set = set(include_indices)

        # Segregation classification, computed once per included variant and
        # reused by Layers 3/4/5. Layer 6 uses a separate X-linked-specific
        # classifier since XLR/XLD read differently off the same AB values.
        segregation_cache: dict[int, str] = {}
        for i in include_indices:
            ab_entry = parental_ab[i] if parental_ab else {}
            segregation_cache[i] = segregation.classify_segregation(
                ab_entry.get("proband"), ab_entry.get("mother"), ab_entry.get("father"),
            )

        gene_chrom_cache = moi.build_gene_chrom_cache(variants, kept_indices, _group_by_gene)

        # ── Layer 3: de novo (AD / AD_AR / XLD genes, AND any X-linked mode —
        #     de novo occurrence is a valid PS2/PM6 signal for XLR/XL genes
        #     too, e.g. a de novo MECP2 variant confirmed absent in both
        #     parents; the moi_xlinked layer only ever adds PP1, never PS2,
        #     so without this a confirmed-de-novo X-linked variant would get
        #     zero credit for that despite trio data confirming it) ─────────
        denovo_genes = {
            g for g, m in gene_mode_cache.items()
            if m in ("AD", "AD_AR", "XLD", "XLR", "XLD_XLR", "XL")
        }
        denovo_targets = [i for i in include_indices if variants[i].get("Gene", "NA") in denovo_genes]
        denovo_outputs: dict[int, str] = {}

        def _denovo_one(i: int) -> tuple[int, str]:
            ab_entry = parental_ab[i] if parental_ab else {}
            has_trio = segregation.has_parent_data(ab_entry)
            has_one_parent = (not has_trio) and segregation.has_any_parent_data(ab_entry)
            return i, moi_denovo.run_one(
                variant_context=context_slices[i],
                base_conclusion=conclusions[i],
                segregation=segregation_cache[i],
                has_trio=has_trio,
                has_one_parent=has_one_parent,
                llm=self._llm,
            )

        if denovo_targets:
            workers = min(MAX_WORKERS_LLM, len(denovo_targets))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_denovo_one, i): i for i in denovo_targets}
                for future in as_completed(futures):
                    i, out = future.result()
                    denovo_outputs[i] = out
                    logger.info(
                        "[Pipeline] De novo layer done for variant %d (%s).",
                        i + 1, variants[i].get("Gene", "?"),
                    )

        # ── Layer 4: dominant-inherited (AD / AD_AR / XLD only — narrower than
        #     Layer 3's de novo gene set. "Cosegregation with an affected
        #     parent" framing doesn't fit XLR/XL genes, which Layer 3 now
        #     covers for de novo purposes but Layer 4 deliberately excludes) ─
        dominant_genes = {g for g, m in gene_mode_cache.items() if m in ("AD", "AD_AR", "XLD")}
        dominant_targets = [i for i in include_indices if variants[i].get("Gene", "NA") in dominant_genes]
        dominant_outputs: dict[int, str] = {}

        def _dominant_one(i: int) -> tuple[int, str]:
            ab_entry = parental_ab[i] if parental_ab else {}
            seg = segregation_cache[i]
            proband_ab_class = segregation.classify_ab_ratio(ab_entry.get("proband"))
            if seg == "maternal":
                transmitting_parent_ab_class = segregation.classify_ab_ratio(ab_entry.get("mother"))
            elif seg == "paternal":
                transmitting_parent_ab_class = segregation.classify_ab_ratio(ab_entry.get("father"))
            else:
                transmitting_parent_ab_class = "uncertain"
            return i, moi_dominant.run_one(
                variant_context=context_slices[i],
                base_conclusion=conclusions[i],
                segregation=seg,
                proband_ab_class=proband_ab_class,
                transmitting_parent_ab_class=transmitting_parent_ab_class,
                llm=self._llm,
            )

        if dominant_targets:
            workers = min(MAX_WORKERS_LLM, len(dominant_targets))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_dominant_one, i): i for i in dominant_targets}
                for future in as_completed(futures):
                    i, out = future.result()
                    dominant_outputs[i] = out
                    logger.info(
                        "[Pipeline] Dominant layer done for variant %d (%s).",
                        i + 1, variants[i].get("Gene", "?"),
                    )

        # ── Layer 5: recessive / compound-het (AR / XLR / AD_AR / XLD_XLR
        #     genes, >=2 included variants, TRANS-gated in Python BEFORE any
        #     LLM call — a CIS pair never reaches moi_recessive.run_pair) ────
        recessive_gene_groups_included: dict[str, list[int]] = {}
        for gene, idxs in recessive_gene_groups.items():
            kept_and_included = [i for i in idxs if i in include_set]
            if len(kept_and_included) >= 2:
                recessive_gene_groups_included[gene] = kept_and_included

        recessive_outputs: dict[str, list[str]] = defaultdict(list)
        recessive_pair_targets: list[tuple[str, int, int, str]] = []  # (gene, i, j, phase)
        for gene, idxs in recessive_gene_groups_included.items():
            for i, j in itertools.combinations(idxs, 2):
                phase = segregation.classify_phase(segregation_cache[i], segregation_cache[j])
                if phase == "cis":
                    logger.info(
                        "[Pipeline] Recessive layer: gene %s variants %d/%d excluded — "
                        "CIS phase (same parent), not compound heterozygous.",
                        gene, i + 1, j + 1,
                    )
                    continue
                recessive_pair_targets.append((gene, i, j, phase))

        def _recessive_one(item: tuple[str, int, int, str]) -> tuple[str, str]:
            gene, i, j, phase = item
            gene_ca = cross_analyses.get(gene)
            var_a, var_b = variants[i], variants[j]
            label_a = f"Variant {i + 1} — {gene} ({var_a.get('HGVS', var_a.get('Variant', '?'))})"
            label_b = f"Variant {j + 1} — {gene} ({var_b.get('HGVS', var_b.get('Variant', '?'))})"
            out = moi_recessive.run_pair(
                gene=gene,
                variant_a_label=label_a,
                variant_a_context=context_slices[i],
                variant_a_base_conclusion=conclusions[i],
                variant_b_label=label_b,
                variant_b_context=context_slices[j],
                variant_b_base_conclusion=conclusions[j],
                phase=phase,
                cross_analysis=gene_ca,
                llm=self._llm,
            )
            return gene, out

        if recessive_pair_targets:
            workers = min(MAX_WORKERS_LLM, len(recessive_pair_targets))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_recessive_one, item): item for item in recessive_pair_targets}
                for future in as_completed(futures):
                    gene, out = future.result()
                    recessive_outputs[gene].append(out)
                    logger.info("[Pipeline] Recessive layer done for gene %s.", gene)

        # ── Layer 5 (solo path): confirmed-homozygous variants in a
        #     recessive-relevant gene with no compound-het partner (gene never
        #     reached the >=2-variant group above). Homozygosity alone
        #     satisfies the biallelic requirement, so these belong in the
        #     Recessive layer, not the Unclassified appendix. ─────────────────
        homozygous_solo_indices: set[int] = set()
        homozygous_solo_targets = [
            i for i in include_indices
            if gene_mode_cache.get(variants[i].get("Gene", "NA")) in ("AR", "XLR", "AD_AR", "XLD_XLR")
            and moi.zygosity_is_confirmed_hom(variants[i].get("Zygosity", ""))
            and variants[i].get("Gene", "NA") not in recessive_gene_groups_included
        ]

        def _recessive_solo_one(i: int) -> tuple[int, str, str]:
            gene = variants[i].get("Gene", "NA")
            gene_ca = cross_analyses.get(gene)
            var = variants[i]
            label = f"Variant {i + 1} — {gene} ({var.get('HGVS', var.get('Variant', '?'))})"
            ab_entry = parental_ab[i] if parental_ab else {}
            seg = segregation.classify_segregation(
                ab_entry.get("proband"), ab_entry.get("mother"), ab_entry.get("father")
            )
            out = moi_recessive.run_solo(
                gene=gene,
                variant_label=label,
                variant_context=context_slices[i],
                variant_base_conclusion=conclusions[i],
                segregation=seg,
                cross_analysis=gene_ca,
                llm=self._llm,
            )
            return i, gene, out

        if homozygous_solo_targets:
            workers = min(MAX_WORKERS_LLM, len(homozygous_solo_targets))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_recessive_solo_one, i): i for i in homozygous_solo_targets}
                for future in as_completed(futures):
                    i, gene, out = future.result()
                    recessive_outputs[gene].append(out)
                    homozygous_solo_indices.add(i)
                    logger.info(
                        "[Pipeline] Recessive layer (homozygous-solo) done for variant %d (%s).",
                        i + 1, gene,
                    )

        # ── Layer 6: X-linked (chrX-only genes — a different filter shape
        #     from Layers 3-5, chromosome-based rather than mode-based) ──────
        xlinked_targets = [
            i for i in include_indices
            if gene_chrom_cache.get(variants[i].get("Gene", "NA")) == "X"
        ]
        xlinked_outputs: dict[int, str] = {}

        def _xlinked_one(i: int) -> tuple[int, str]:
            ab_entry = parental_ab[i] if parental_ab else {}
            pattern = segregation.classify_xlinked_ab(ab_entry.get("proband"), ab_entry.get("mother"))
            return i, moi_xlinked.run_one(
                variant_context=context_slices[i],
                base_conclusion=conclusions[i],
                xlinked_pattern=pattern,
                llm=self._llm,
            )

        if xlinked_targets:
            workers = min(MAX_WORKERS_LLM, len(xlinked_targets))
            with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
                futures = {pool.submit(_xlinked_one, i): i for i in xlinked_targets}
                for future in as_completed(futures):
                    i, out = future.result()
                    xlinked_outputs[i] = out
                    logger.info(
                        "[Pipeline] X-linked layer done for variant %d (%s).",
                        i + 1, variants[i].get("Gene", "?"),
                    )

        # ── Stage 5: Cross-MOI clinical conclusion (Layer 8) ──────────────────
        logger.info("[Pipeline] Generating cross-MOI clinical conclusion...")
        layer_outputs_for_summary = {
            "De Novo":            list(denovo_outputs.values()),
            "Dominant-Inherited": list(dominant_outputs.values()),
            "Recessive":          [block for blocks in recessive_outputs.values() for block in blocks],
            "X-Linked":           list(xlinked_outputs.values()),
        }
        layer_outputs_for_summary = {k: v for k, v in layer_outputs_for_summary.items() if v}

        moi_covered_indices = (
            set(denovo_outputs) | set(dominant_outputs)
            | {i for idxs in recessive_gene_groups_included.values() for i in idxs}
            | homozygous_solo_indices
            | set(xlinked_outputs)
        )
        unclassified_indices = [i for i in include_indices if i not in moi_covered_indices]
        unclassified_conclusions = [conclusions[i] for i in unclassified_indices]

        final_summary = final_conclusion.run(
            layer_outputs=layer_outputs_for_summary,
            unclassified_conclusions=unclassified_conclusions,
            patient_phenotype=patient_phenotype,
            llm=self._llm,
        )

        # ── ACMG SF actionable variants section (runs once, after final_conclusion,
        #     over every variant flagged earlier — independent of include/exclude
        #     since flagged variants were already force-included above) ───────────
        actionable_flagged = [
            {
                "index":          i,
                "gene":           variants[i].get("Gene", "?"),
                "hgvs":           variants[i].get("HGVS", variants[i].get("Variant", "?")),
                "condition":      acmg_sf.ACMG_SF_CONDITIONS.get(variants[i].get("Gene", ""), "n/a"),
                "reason":         actionable_reasons.get(i, ""),
                "zygosity":       variants[i].get("Zygosity", "NA"),
                "classification": _extract_classification_label(conclusions.get(i, "")),
            }
            for i in sorted(actionable_indices)
            if i in conclusions
        ]
        actionable_section = actionable.run(
            flagged=actionable_flagged,
            patient_phenotype=patient_phenotype,
            llm=self._llm,
        )

        # ── Assemble final output ─────────────────────────────────────────────

        # PROCESS DETAILS — all variants + triage summary + second triage summary
        triage_summary = _build_triage_summary(
            n=n,
            kept_indices=kept_indices,
            discarded_indices=discarded_indices,
            include_indices=include_indices,
            exclude_indices=exclude_indices,
            triage_results=triage_results,
            variants=variants,
            triage_ran=triage_ran,
        )
        second_triage_summary = _build_second_triage_summary(
            kept_indices=kept_indices,
            inclusion_decisions=inclusion_decisions,
            second_triage_justifications=second_triage_justifications,
            variants=variants,
        )
        process_parts = [process_details]
        if triage_summary:
            process_parts.append(triage_summary)
        process_parts.append(second_triage_summary)
        process_section = "\n\n".join(process_parts)

        # SEGREGATION ANALYSIS — parental AB data summary (if available)
        segregation_section = _build_segregation_analysis(
            variants=variants,
            parental_ab=parental_ab,
        )

        # AUGMENTED CONTEXT — all variants (unchanged)
        full_context_display = _reconstruct_full_context(context_slices)

        # REASONING — all kept variants, in original variant order
        reasoning_parts = []
        for i in kept_indices:
            variant  = variants[i]
            gene     = variant.get("Gene", "?")
            hgvs     = variant.get("HGVS", variant.get("Variant", "?"))
            decision = inclusion_decisions.get(i, "INCLUDE")
            tier_tag = f"[{decision}]"
            reasoning_parts.append(
                f"{SEP_VARIANT}\nVARIANT {i + 1} — {gene}  {hgvs}  {tier_tag}\n"
                f"{SEP_VARIANT}\n{reasonings[i]}"
            )
        reasoning_display = "\n\n".join(reasoning_parts)

        # FINAL REPORT — one section per MOI layer (each independently ranked
        # Pathogenic → Likely Pathogenic → VUS → Likely Benign → Benign via the
        # same _classification_rank regex, which matches "ACMG points" anywhere
        # in the text so it works unchanged on each layer's own "Total ACMG
        # points: N → Classification" line), a variant appearing in multiple
        # qualifying layers shown in full in each (not deduplicated), an
        # Unclassified appendix for variants whose gene MOI never resolved to
        # any layer, then the cross-MOI Clinical Conclusion + the pre-existing
        # exclude/discard appendix (unrelated, untouched).
        moi_sections = "".join([
            _build_moi_section("DE NOVO ANALYSIS", list(denovo_outputs.values())),
            _build_moi_section("DOMINANT-INHERITED ANALYSIS", list(dominant_outputs.values())),
            _build_moi_section(
                "RECESSIVE ANALYSIS (COMPOUND-HETEROZYGOUS / CONFIRMED HOMOZYGOUS)",
                [block for blocks in recessive_outputs.values() for block in blocks],
            ),
            _build_moi_section("X-LINKED ANALYSIS", list(xlinked_outputs.values())),
            _build_moi_section("UNCLASSIFIED — NO MOI-SPECIFIC ANALYSIS", unclassified_conclusions),
        ])

        final_report = moi_sections + f"\n{SEP_VARIANT}\nCLINICAL CONCLUSION\n{SEP_VARIANT}\n\n{final_summary}\n"

        if actionable_section:
            final_report += (
                f"\n{SEP_VARIANT}\nACTIONABLE VARIANTS (ACMG SF)\n{SEP_VARIANT}\n\n{actionable_section}\n"
            )

        if triage_ran and (exclude_indices or discarded_indices):
            final_report += _build_report_appendix(
                exclude_indices=exclude_indices,
                discarded_indices=discarded_indices,
                triage_results=triage_results,
                variants=variants,
                conclusions=conclusions,
            )

        output = (
            (f"{SEP}\nCOLUMN HEADER INTERPRETATION\n{SEP}\n{header_mapping_summary}\n\n"
             if header_mapping_summary else "")
            + f"{SEP}\nPROCESS DETAILS\n{SEP}\n"
            + process_section + "\n\n"
            + (f"{SEP}\nSEGREGATION ANALYSIS\n{SEP}\n{segregation_section}\n\n" if segregation_section else "")
            + f"{SEP}\nAUGMENTED CONTEXT\n{SEP}\n"
            + full_context_display + "\n\n"
            + f"{SEP}\nREASONING\n{SEP}\n"
            + reasoning_display + "\n\n"
            + f"{SEP}\nFINAL REPORT\n{SEP}\n"
            + final_report + "\n"
        )
        return output


# ═══════════════════════════════════════════════════════════════════════════════
# OUTPUT ASSEMBLY HELPERS
# ═══════════════════════════════════════════════════════════════════════════════

def _build_triage_summary(
    n: int,
    kept_indices: list[int],
    discarded_indices: list[int],
    include_indices: list[int],
    exclude_indices: list[int],
    triage_results: dict[int, tuple[str, str]],
    variants: list[dict],
    triage_ran: bool,
) -> str:
    if not triage_ran:
        return ""

    SEP = "─" * 56
    lines = [
        f"\n{SEP}",
        "TRIAGE SUMMARY",
        SEP,
        f"Total variants  : {n}",
        f"Kept (Tier 1+2) : {len(kept_indices)}",
        f"Discarded       : {len(discarded_indices)}",
    ]

    if discarded_indices:
        lines.append("\nDiscarded variants:")
        for i in discarded_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            _, just = triage_results[i]
            lines.append(f"  Variant {i + 1} — {gene} ({hgvs}): {just}")

    # Compound-het exemptions: kept variants whose justification records the override
    exemptions = [
        i for i in kept_indices
        if triage_results[i][1].startswith("[compound-het exempt")
    ]
    if exemptions:
        lines.append("\nCompound-het exemptions (forced KEEP):")
        for i in exemptions:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            lines.append(f"  Variant {i + 1} — {gene} ({hgvs})")

    include_labels = ", ".join(f"Variant {i + 1}" for i in include_indices) or "(none)"
    exclude_labels = ", ".join(f"Variant {i + 1}" for i in exclude_indices) or "(none)"
    lines.append(f"\nIncluded (→ conclusion): {include_labels}")
    lines.append(f"Excluded (→ reasoning only): {exclude_labels}")

    return "\n".join(lines)


def _build_second_triage_summary(
    kept_indices: list[int],
    inclusion_decisions: dict[int, str],
    second_triage_justifications: dict[int, str],
    variants: list[dict],
) -> str:
    SEP = "─" * 56
    n_included = sum(1 for i in kept_indices if inclusion_decisions.get(i) == "INCLUDE")
    n_excluded  = sum(1 for i in kept_indices if inclusion_decisions.get(i) == "EXCLUDE")
    lines = [
        f"\n{SEP}",
        "SECOND TRIAGE (INCLUSION DECISIONS)",
        SEP,
        f"Evaluated : {len(kept_indices)}",
        f"Included  : {n_included}",
        f"Excluded  : {n_excluded}",
        "",
    ]
    for i in kept_indices:
        variant  = variants[i]
        gene     = variant.get("Gene", "?")
        hgvs     = variant.get("HGVS", variant.get("Variant", "?"))
        decision = inclusion_decisions.get(i, "INCLUDE")
        just     = second_triage_justifications.get(i, "")
        lines.append(f"  Variant {i + 1} — {gene}  {hgvs}  [{decision}]")
        if just:
            lines.append(f"    {just}")
    return "\n".join(lines)


_ACMG_BLOCK_RE = re.compile(r"\*\*ACMG criteria:\*\*.*?(?=\*\*Comment:\*\*|\Z)", re.DOTALL)


def _extract_acmg_block(conclusion_text: str) -> str:
    """Pull just the ACMG criteria list + points line out of a full conclusion block."""
    m = _ACMG_BLOCK_RE.search(conclusion_text)
    return m.group(0).strip() if m else ""


def _build_moi_section(title: str, blocks: list[str]) -> str:
    """One MOI-layer report section (Phase 7 of the MOI-layer restructuring).
    Blocks are ranked P → LP → VUS → LB → B by the same _classification_rank
    used for the base layer — it searches for "ACMG points" anywhere in the
    text, so it matches each layer's own "Total ACMG points: N →
    Classification" line unchanged, no separate rank function needed.
    Returns "" (omitted entirely) when there are no blocks for this layer —
    a case with no de novo findings simply has no DE NOVO ANALYSIS section,
    rather than an empty one."""
    if not blocks:
        return ""
    SEP = "─" * 56
    ranked = sorted(blocks, key=_classification_rank)
    body = "\n\n---\n\n".join(ranked)
    return f"\n{SEP}\n{title}\n{SEP}\n{body}\n"


def _build_report_appendix(
    exclude_indices: list[int],
    discarded_indices: list[int],
    triage_results: dict[int, tuple[str, str]],
    variants: list[dict],
    conclusions: dict[int, str],
) -> str:
    lines = ["\n\n---\n\n## Variants not included in conclusion"]

    if exclude_indices:
        lines.append("\n### Excluded — Reasoning available (see REASONING section)")
        for i in exclude_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            lines.append(f"- Variant {i + 1} — {gene} ({hgvs})")
            acmg_block = _extract_acmg_block(conclusions.get(i, ""))
            if acmg_block:
                lines.append(f"  {acmg_block}")

    if discarded_indices:
        lines.append("\n### Discarded at first triage")
        for i in discarded_indices:
            variant = variants[i]
            gene    = variant.get("Gene", "?")
            hgvs    = variant.get("HGVS", variant.get("Variant", "?"))
            _, just = triage_results[i]
            lines.append(f"- Variant {i + 1} — {gene} ({hgvs}): {just}")

    return "\n".join(lines)


def _build_segregation_analysis(
    variants: list[dict],
    parental_ab: list[dict] | None,
) -> str:
    """
    Build the SEGREGATION ANALYSIS section showing parental allelic balance data.
    Returns empty string if no parental data is available.
    """
    if not parental_ab:
        return ""

    SEP = "─" * 56
    lines = [
        SEP,
        "PARENTAL ALLELIC BALANCE DATA",
        SEP,
    ]

    # Detect column labels from first entry
    first_entry = parental_ab[0] if parental_ab else {}
    labels = list(first_entry.keys())

    if labels:
        header = "Variant".ljust(8) + "Gene".ljust(12)
        for label in labels:
            header += label.capitalize().rjust(12)
        lines.append(header)
        lines.append("─" * max(len(header), 40))

    for i, (variant, ab_entry) in enumerate(zip(variants, parental_ab)):
        gene = variant.get("Gene", "?")[:10]
        line = f"{i + 1:<8}{gene:<12}"
        for label in labels:
            val = ab_entry.get(label, "NA")
            line += val.rjust(12)
        lines.append(line)

    # Add interpretation summary
    n_with_parental = sum(
        1 for ab in parental_ab
        if any(k in ab for k in ("mother", "father", "extra1"))
    )
    if n_with_parental > 0:
        lines.append("")
        lines.append(f"Variants with parental data: {n_with_parental}/{len(parental_ab)}")
        lines.append("De novo candidates (AB < 0.1 in both parents):")
        for i, ab_entry in enumerate(parental_ab):
            try:
                mother_ab = float(ab_entry.get("mother", "1") or "1")
            except ValueError:
                mother_ab = 1.0
            try:
                father_ab = float(ab_entry.get("father", "1") or "1")
            except ValueError:
                father_ab = 1.0
            if mother_ab < 0.1 and father_ab < 0.1:
                variant = variants[i]
                gene = variant.get("Gene", "?")
                hgvs = variant.get("HGVS", variant.get("Variant", "?"))
                lines.append(f"  Variant {i + 1} — {gene} ({hgvs})")

    return "\n".join(lines)
