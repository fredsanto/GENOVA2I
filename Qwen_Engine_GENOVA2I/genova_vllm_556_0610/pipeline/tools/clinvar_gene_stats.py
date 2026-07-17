"""
pipeline/tools/clinvar_gene_stats.py — ClinVar gene-level P/LP consequence counts.

Counts Pathogenic/Likely pathogenic ClinVar variants per gene, split by molecular
consequence (missense vs. nonsense/frameshift), to ground PP2/BP1 in concrete
gene-level evidence instead of gene-name plausibility. Gene-scoped and cached at
class level, same pattern as GnomadConstraintTool / LitVar2SummaryTool._cgd_table.
"""

import logging
import threading

from pipeline.tools.base import NetworkTool
from pipeline.tools.websearch import _ncbi_get, DEFAULT_TIMEOUT
from pipeline.core.context import ToolContext
from pipeline.core.errors import ToolFetchError, ToolParseError

logger = logging.getLogger(__name__)

# Ratio of the larger count to the smaller at or above which a gene is
# considered "predominant" for one consequence class rather than "balanced".
PREDOMINANCE_RATIO = 2.0


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
        return f"{gene}[gene] AND (pathogenic[Properties] OR likely pathogenic[Properties])"

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
        missense_term = f'{base} AND missense variant[molecular consequence]'
        nonsense_term = (
            f'{base} AND (nonsense variant[molecular consequence] '
            f'OR frameshift variant[molecular consequence])'
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

    def run(self, variant: dict, context: ToolContext) -> str | None:
        gene = context.field("Gene")
        stats = self._get_stats(gene)
        missense = stats["missense"]
        nonsense = stats["nonsense"]

        if missense == 0 and nonsense == 0:
            verdict = "insufficient data — no P/LP missense or nonsense/frameshift variants found in ClinVar"
        elif nonsense == 0 or (missense > 0 and missense / max(nonsense, 1) >= PREDOMINANCE_RATIO):
            verdict = "missense-predominant (supports PP2, argues against BP1)"
        elif missense == 0 or (nonsense / max(missense, 1) >= PREDOMINANCE_RATIO):
            verdict = "nonsense/frameshift-predominant (supports BP1, argues against PP2)"
        else:
            verdict = "balanced — neither consequence class clearly predominates"

        return (
            f"CLINVAR GENE-LEVEL P/LP VARIANT COUNTS ({gene}):\n"
            f"P/LP missense variants   : {missense}\n"
            f"P/LP nonsense/frameshift : {nonsense}\n"
            f"-> {verdict}"
        )
