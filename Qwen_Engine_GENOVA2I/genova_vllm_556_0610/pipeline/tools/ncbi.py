"""
pipeline/tools/ncbi.py — NCBI fetch tool for the ReAct agent.

Contains:
  - NCBIFetchTool — routes PubMed / PMC / ClinVar URLs to the right E-utilities call

Shared helpers (_ncbi_get, _clean_xml_text) are imported from websearch.py.

Note: NCBIFetchTool is a sub-tool consumed by WebSearchAgentTool's internal ReActAgent.
It is NOT a pipeline-level tool. It overrides run(query: str) for ReActAgent compatibility.
"""

import re
import logging
import xml.etree.ElementTree as ET

from pipeline.tools.base import NetworkTool
from pipeline.tools.websearch import (
    _ncbi_get,
    _clean_xml_text,
    DEFAULT_TIMEOUT,
    DEFAULT_MAX_CHARS,
)

logger = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# TOOL 2 — NCBI FETCH
# ─────────────────────────────────────────────

class NCBIFetchTool(NetworkTool):
    """
    Fetches structured data from any NCBI resource using the E-utilities API.
    Covers PubMed abstracts, PMC full-text articles, and ClinVar variant records.
    Use this for ANY URL containing ncbi.nlm.nih.gov.
    Do NOT use web_fetch for NCBI URLs.
    """
    name        = "ncbi_fetch"
    description = (
        "Fetches clean, structured data from NCBI resources via the E-utilities API. "
        "Handles: PubMed abstracts (pubmed.ncbi.nlm.nih.gov), "
        "PMC full-text articles (pmc.ncbi.nlm.nih.gov), "
        "and ClinVar variant records (ncbi.nlm.nih.gov/clinvar). "
        "Always prefer this tool over web_fetch for any ncbi.nlm.nih.gov URL."
    )
    input_system = (
        "You are an NCBI URL extractor. "
        "Extract the single most relevant NCBI URL from the context. "
        "Only accept URLs from ncbi.nlm.nih.gov subdomains. "
        "NEVER include tool names, function calls, parentheses, or code syntax. "
        "Output ONLY the bare URL string. No explanation, no punctuation."
    )
    input_template = "Extract the NCBI URL from: {user_input}"

    # URL pattern matchers
    _PUBMED_RE  = re.compile(r"pubmed\.ncbi\.nlm\.nih\.gov/(\d+)")
    _PMC_RE     = re.compile(r"pmc\.ncbi\.nlm\.nih\.gov/articles/PMC(\d+)")
    _CLINVAR_RE = re.compile(r"ncbi\.nlm\.nih\.gov/clinvar/variation/(\d+)")
    _NCBI_RE    = re.compile(r"ncbi\.nlm\.nih\.gov")

    def __init__(self, max_chars: int = DEFAULT_MAX_CHARS, timeout: int = DEFAULT_TIMEOUT):
        self.max_chars = max_chars
        self.timeout   = timeout

    # ── PubMed abstract ────────────────────────────────────────────────────────

    def _fetch_pubmed(self, pmid: str) -> str:
        logger.debug("Fetching PubMed PMID=%s", pmid)

        # Metadata
        try:
            meta = _ncbi_get(
                "esummary.fcgi",
                {"db": "pubmed", "id": pmid, "retmode": "json"},
                self.timeout,
            ).json()["result"].get(pmid, {})
        except (Exception, KeyError, ValueError) as e:
            return f"PubMed metadata error for PMID {pmid}: {e}"

        title   = meta.get("title", "No title")
        journal = meta.get("fulljournalname", meta.get("source", "Unknown journal"))
        pubdate = meta.get("pubdate", "Unknown date")
        raw_authors = meta.get("authors", [])
        authors = ", ".join(a["name"] for a in raw_authors[:5] if "name" in a)
        if len(raw_authors) > 5:
            authors += " et al."

        # Full abstract
        try:
            xml = _ncbi_get(
                "efetch.fcgi",
                {"db": "pubmed", "id": pmid, "rettype": "xml", "retmode": "xml"},
                self.timeout,
            ).text
            root    = ET.fromstring(xml)
            parts   = [_clean_xml_text(ET.tostring(a, encoding="unicode"))
                       for a in root.iter("AbstractText")]
            abstract = " ".join(parts) if parts else "No abstract available."
        except Exception as e:
            abstract = f"Abstract fetch error: {e}"

        return (
            f"Title:   {title}\n"
            #f"Authors: {authors}\n"
            #f"Journal: {journal}\n"
            #f"Date:    {pubdate}\n"
            f"URL:     https://pubmed.ncbi.nlm.nih.gov/{pmid}/\n\n"
            f"{abstract[:self.max_chars]}"
        )

    # ── PMC full text ──────────────────────────────────────────────────────────

    def _fetch_pmc(self, pmc_id: str) -> str:
        logger.debug("Fetching PMC ID=%s", pmc_id)
        try:
            xml = _ncbi_get(
                "efetch.fcgi",
                {"db": "pmc", "id": pmc_id, "rettype": "xml", "retmode": "xml"},
                self.timeout,
            ).text
            root = ET.fromstring(xml)
        except Exception as e:
            return f"PMC fetch error for PMC{pmc_id}: {e}"

        # Abstract first, then body paragraphs
        parts = [_clean_xml_text(ET.tostring(a, encoding="unicode"))
                 for a in root.iter("abstract")]
        if not parts:
            parts = [_clean_xml_text(ET.tostring(p, encoding="unicode"))
                     for p in root.iter("p")]

        text = " ".join(parts).strip()
        if not text:
            return f"PMC{pmc_id}: content found but could not be extracted."

        return f"PMC{pmc_id} content:\n\n{text[:self.max_chars]}"

    # ── ClinVar variation ──────────────────────────────────────────────────────

    def _fetch_clinvar(self, variation_id: str) -> str:
        logger.debug("Fetching ClinVar variation_id=%s", variation_id)
        try:
            data   = _ncbi_get(
                "esummary.fcgi",
                {"db": "clinvar", "id": variation_id, "retmode": "json"},
                self.timeout,
            ).json()
            result = data.get("result", {}).get(variation_id, {})
        except Exception as e:
            return f"ClinVar fetch error for variation {variation_id}: {e}"

        if not result:
            return f"No ClinVar data found for variation ID {variation_id}."

        name       = result.get("title", "No title")
        germline   = result.get("germline_classification", {}).get("description", "Unknown")
        review     = result.get("germline_classification", {}).get("review_status", "")
        last_eval  = result.get("germline_classification", {}).get("last_evaluated", "")
        gene       = ", ".join(g["symbol"] for g in result.get("genes", []) if "symbol" in g)
        conditions = ", ".join(t["name"]   for t in result.get("trait_set", []) if "name" in t)

        return (
            f"Variant:    {name}\n"
            f"Gene:       {gene}\n"
            f"Conditions: {conditions}\n"
            f"Class:      {germline}\n"
            f"Review:     {review}\n"
            f"Evaluated:  {last_eval}\n"
            f"URL:        https://www.ncbi.nlm.nih.gov/clinvar/variation/{variation_id}/"
        )

    # ── Router ─────────────────────────────────────────────────────────────────

    # Also registered as legacy alias so agents calling "pubmed_fetch" still work
    aliases = ["pubmed_fetch"]

    # Bare PMID (digits only) — agent may pass just the ID without a full URL
    _BARE_PMID_RE = re.compile(r"^\d+$")

    def run(self, url: str) -> str:
        url = url.strip()

        # Handle bare PMID e.g. "39500001"
        if self._BARE_PMID_RE.match(url):
            logger.debug("NCBIFetchTool received bare PMID=%s", url)
            return self._fetch_pubmed(url)

        if not self._NCBI_RE.search(url):
            return "Error: ncbi_fetch only handles ncbi.nlm.nih.gov URLs. Use web_fetch for other sites."

        m = self._PUBMED_RE.search(url)
        if m:
            return self._fetch_pubmed(m.group(1))

        m = self._PMC_RE.search(url)
        if m:
            return self._fetch_pmc(m.group(1))

        m = self._CLINVAR_RE.search(url)
        if m:
            return self._fetch_clinvar(m.group(1))

        return (
            f"Unrecognised NCBI URL pattern: {url}\n"
            "Supported patterns:\n"
            "  pubmed.ncbi.nlm.nih.gov/<PMID>\n"
            "  pmc.ncbi.nlm.nih.gov/articles/PMC<ID>\n"
            "  ncbi.nlm.nih.gov/clinvar/variation/<ID>"
        )
