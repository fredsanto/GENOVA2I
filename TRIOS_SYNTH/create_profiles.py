#!/usr/bin/env python3
"""
create_profiles.py — Generate one synthetic trio variant profile from gnomAD v4.

Each profile = 40 variants:
  - 1 LoF variant (nonsense, frameshift, or splicing; AF < 0.1%) in a gene
    randomly picked from DOMINANT_ID_GENES (dominant intellectual-disability genes)
  - 39 missense variants (AF < 1%) in genes randomly picked from
    CONTROL_MIXED_GENES (unrelated disease-gene pool, one variant per gene)

Usage:
    python scripts/create_profiles.py -o output.csv
    python scripts/create_profiles.py -o output.csv --seed 42

Output CSV matches the pipeline's input format (Variant,Chromosome,Position,...)
plus trio columns (Allelic_balance_proband/1/2): each variant is randomly
inherited from one parent (AB=0.5), the other parent has AB=0.

gnomAD returns Ensembl transcript IDs (ENST...). The script automatically converts
these to RefSeq NM_... IDs via the MANE Select map. Use --keep-enst to skip this.
"""

from __future__ import annotations

import argparse
import csv
import gzip
import io
import logging
import random
import re
import sys
import time
from typing import Any

import requests

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger(__name__)

GNOMAD_API = "https://gnomad.broadinstitute.org/api/"
REQUEST_DELAY = 0.5

LOF_CONSEQUENCES = {"stop_gained", "nonsense", "frameshift_variant", "splice_donor_variant", "splice_acceptor_variant"}
MISSENSE_CONSEQUENCES = {"missense_variant"}

LOF_MAX_AF = 0.001   # <0.1%
MISSENSE_MAX_AF = 0.01  # <1%

FREQ_HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; GenMasterAI/1.0)",
    "Content-Type": "application/json",
}

# ── Gene pools ──────────────────────────────────────────────────────────────────

DOMINANT_ID_GENES = [
    "SYNGAP1", "KDM2A", "ARID1B", "CHD2", "SCN1A", "STXBP1", "GRIN2A",
    "KDM5C", "MEF2C", "SATB2", "TCF4", "DYRK1A", "FOXP1", "SLC6A1",
    "HNRNPU", "KCNQ2", "SMARCA2", "MED13L", "ASXL1", "SETBP1", "FOXG1",
    "GNAO1", "GABRB2", "GABRB3", "PPP2R5D", "CTNNB1", "CHD8",
]

CONTROL_MIXED_GENES = [
    "ALDH2", "HBB", "CFTR", "MYH7", "TTN", "PKD1", "LDLR", "COL1A1",
    "BRCA2", "USH2A", "MLH1", "TP53", "RYR1", "FBN1", "HEXA", "G6PD",
    "KCNQ1", "VWF", "RB1", "APOB", "NF1", "SMN1", "POLG", "TGFBR2",
    "ATM", "COL4A5", "SOD1", "PMP22", "RPE65", "TSC2", "GAA", "CACNA1A",
    "ABCA4", "RHO", "SLC26A4", "GJB2", "F8", "PAH", "MYBPC3", "DMD",
    "SCN5A", "KCNH2", "BRCA1", "MSH2", "MSH6", "PMS2", "APC", "VHL",
    "RET", "MEN1", "NF2", "COL5A1", "ELN", "FLNA", "DES", "LMNA",
]

# ── gnomAD ──────────────────────────────────────────────────────────────────────


def fetch_gnomad_variants(gene: str) -> list[dict[str, Any]]:
    query = """
    query GeneVariants($geneSymbol: String!) {
        gene(gene_symbol: $geneSymbol, reference_genome: GRCh38) {
            variants(dataset: gnomad_r4) {
                variant_id
                rsid
                hgvsc
                hgvsp
                consequence
                transcript_id
                exome { ac an af }
                genome { ac an af }
            }
        }
    }
    """
    payload = {"query": query, "variables": {"geneSymbol": gene}}
    try:
        resp = requests.post(GNOMAD_API, json=payload, headers=FREQ_HEADERS, timeout=30)
        resp.raise_for_status()
        data = resp.json()
    except Exception as e:
        logger.warning("  API request failed for %s: %s", gene, e)
        return []

    gene_data = data.get("data", {}).get("gene")
    if not gene_data:
        logger.warning("  Gene %s not found in gnomAD", gene)
        return []

    variants = gene_data.get("variants", [])
    if not variants:
        logger.warning("  No variants returned for %s", gene)
        return []

    return variants


def parse_variant_id(variant_id: str) -> tuple[str, int, str, str] | None:
    m = re.match(r"^(\w+)-(\d+)-([ACGTacgt]+)-([ACGTacgt]+)$", variant_id)
    if not m:
        return None
    chrom = m.group(1)
    if not chrom.startswith("chr"):
        chrom = f"chr{chrom}"
    return chrom, int(m.group(2)), m.group(3).upper(), m.group(4).upper()


def pick_variant(
    variants: list[dict[str, Any]],
    allowed_consequences: set[str],
    max_af: float,
) -> dict[str, Any] | None:
    """Pick one random variant whose consequence is in `allowed_consequences` and
    whose max(exome_af, genome_af) is in (0, max_af)."""
    candidates = [v for v in variants if v.get("consequence", "") in allowed_consequences]
    filtered = []
    for v in candidates:
        ex_af = (v.get("exome") or {}).get("af") or 0
        gn_af = (v.get("genome") or {}).get("af") or 0
        af = max(ex_af, gn_af) if (ex_af or gn_af) else 0
        if 0 < af < max_af:
            v["_max_af"] = af
            filtered.append(v)
    if not filtered:
        return None
    return random.choice(filtered)


def build_csv_row(v: dict[str, Any], gene: str) -> dict[str, str]:
    transmitting_parent = random.choice([1, 2])
    vid = v.get("variant_id", "")
    coords = parse_variant_id(vid)
    if coords:
        chrom, pos, ref, alt = coords
        variant_str = f"{chrom}:{pos} {ref}>{alt}"
    else:
        chrom = pos = ref = alt = "NA"
        variant_str = vid or "NA"

    rsid = v.get("rsid") or "NA"
    hgvsc = v.get("hgvsc") or ""
    hgvsp = v.get("hgvsp") or ""
    tx = v.get("transcript_id") or ""
    if hgvsp.startswith("p."):
        hgvsp = hgvsp[2:]
    if tx and hgvsc:
        hgvs = f"{tx}:{hgvsc} p.{hgvsp}" if hgvsp else f"{tx}:{hgvsc}"
    elif hgvsc:
        hgvs = f"{hgvsc} p.{hgvsp}" if hgvsp else hgvsc
    else:
        hgvs = "NA"

    conseq = v.get("consequence", "")
    if "missense" in conseq:
        vtype = "SNV"
    elif "stop_gained" in conseq:
        vtype = "nonsense"
    elif "frameshift" in conseq:
        vtype = "frameshift"
    elif "splice" in conseq:
        vtype = "splicing"
    else:
        vtype = conseq.replace("_variant", "").replace("_", " ") if conseq else "SNV"

    return {
        "Variant": variant_str,
        "Chromosome": chrom,
        "Position": str(pos) if pos != "NA" else "NA",
        "RS_ID": rsid,
        "Ref_seq": ref,
        "Var_seq": alt,
        "Type": vtype,
        "HGVS": hgvs,
        "Zygosity": "Heterozygous",
        "Gene": gene,
        "Allelic_balance_proband": "0.5",
        "Allelic_balance_1": "0.5" if transmitting_parent == 1 else "0",
        "Allelic_balance_2": "0.5" if transmitting_parent == 2 else "0",
    }


FIELDS = [
    "Variant", "Chromosome", "Position", "RS_ID", "Ref_seq", "Var_seq", "Type", "HGVS",
    "Zygosity", "Gene", "Allelic_balance_proband", "Allelic_balance_1", "Allelic_balance_2",
]


# ── MANE transcript mapping ─────────────────────────────────────────────────────

MANE_URL = "https://ftp.ncbi.nlm.nih.gov/refseq/MANE/MANE_human/current/MANE.GRCh38.v1.5.summary.txt.gz"
_MANE_ENST_TO_NM: dict[str, str] | None = None
_MANE_GENE_TO_TX: dict[str, tuple[str, str]] | None = None  # gene -> (ENST, NM_)


def load_mane() -> tuple[dict[str, str], dict[str, tuple[str, str]]]:
    global _MANE_ENST_TO_NM, _MANE_GENE_TO_TX
    if _MANE_ENST_TO_NM is not None:
        return _MANE_ENST_TO_NM, _MANE_GENE_TO_TX
    logger.info("Downloading MANE transcript map from NCBI...")
    try:
        resp = requests.get(MANE_URL, timeout=30, stream=True)
        resp.raise_for_status()
        raw = gzip.decompress(resp.content).decode("utf-8")
        reader = csv.DictReader(io.StringIO(raw), delimiter="\t")
        enst_map: dict[str, str] = {}
        gene_map: dict[str, tuple[str, str]] = {}
        for row in reader:
            enst = (row.get("Ensembl_nuc") or "").split(".")[0]
            nm = (row.get("RefSeq_nuc") or "").strip()
            gene = (row.get("symbol") or "").strip().upper()
            if enst and nm and nm.startswith("NM_"):
                if enst not in enst_map:
                    enst_map[enst] = nm
                if gene and gene not in gene_map:
                    gene_map[gene] = (enst, nm)
        _MANE_ENST_TO_NM = enst_map
        _MANE_GENE_TO_TX = gene_map
        logger.info("  Loaded %d ENST->NM_ and %d gene->MANE mappings", len(enst_map), len(gene_map))
        return enst_map, gene_map
    except Exception as e:
        logger.warning("  Failed to load MANE map: %s — keeping ENST IDs", e)
        _MANE_ENST_TO_NM = {}
        _MANE_GENE_TO_TX = {}
        return {}, {}


def apply_mane(row: dict, gene: str, mane_gene_to_tx: dict) -> None:
    mane_tx = mane_gene_to_tx.get(gene.upper())
    if not mane_tx:
        return
    mane_enst, mane_nm = mane_tx
    hgvs = row.get("HGVS", "")
    if hgvs.startswith(mane_enst):
        row["HGVS"] = hgvs.replace(mane_enst, mane_nm, 1)


# ── Profile generation ──────────────────────────────────────────────────────────


def pick_lof_row(mane_gene_to_tx: dict, rng: random.Random) -> tuple[str, dict] | None:
    """Try genes from DOMINANT_ID_GENES (shuffled) until one yields a LoF variant
    (nonsense/frameshift/splicing, AF < 0.1%). Prefers the MANE Select transcript
    when available."""
    candidates = list(DOMINANT_ID_GENES)
    rng.shuffle(candidates)
    for gene in candidates:
        variants = fetch_gnomad_variants(gene)
        time.sleep(REQUEST_DELAY)
        if not variants:
            continue
        mane_tx = mane_gene_to_tx.get(gene.upper())
        variants_for_pick = variants
        if mane_tx:
            mane_enst, _ = mane_tx
            tx_variants = [v for v in variants if (v.get("transcript_id") or "").startswith(mane_enst)]
            if tx_variants:
                variants_for_pick = tx_variants
        v = pick_variant(variants_for_pick, LOF_CONSEQUENCES, LOF_MAX_AF)
        if v is None:
            logger.warning("  %s: no LoF variant AF<%.3f%%, trying next gene", gene, LOF_MAX_AF * 100)
            continue
        row = build_csv_row(v, gene)
        apply_mane(row, gene, mane_gene_to_tx)
        logger.info("  LoF pick: %s %s AF=%.2e", gene, v.get("consequence"), v["_max_af"])
        return gene, row
    return None


def pick_missense_rows(mane_gene_to_tx: dict, rng: random.Random, quota: int, exclude: set[str]) -> list[tuple[str, dict]] | None:
    """Try genes from CONTROL_MIXED_GENES (shuffled) until `quota` of them yield a
    missense variant (AF < 1%), one variant per gene."""
    candidates = [g for g in CONTROL_MIXED_GENES if g not in exclude]
    rng.shuffle(candidates)
    picked: list[tuple[str, dict]] = []
    for gene in candidates:
        if len(picked) == quota:
            break
        variants = fetch_gnomad_variants(gene)
        time.sleep(REQUEST_DELAY)
        if not variants:
            continue
        v = pick_variant(variants, MISSENSE_CONSEQUENCES, MISSENSE_MAX_AF)
        if v is None:
            logger.warning("  %s: no missense variant AF<%.1f%%, trying next gene", gene, MISSENSE_MAX_AF * 100)
            continue
        row = build_csv_row(v, gene)
        apply_mane(row, gene, mane_gene_to_tx)
        picked.append((gene, row))
    if len(picked) < quota:
        logger.warning("  Could not fill missense quota (%d/%d)", len(picked), quota)
        return None
    return picked


def generate_profile(seed: int | None) -> tuple[str, list[dict]] | None:
    rng = random.Random(seed)
    _, mane_gene_to_tx = load_mane()

    lof = pick_lof_row(mane_gene_to_tx, rng)
    if lof is None:
        logger.error("Could not find a LoF variant in any DOMINANT_ID_GENES gene")
        return None
    pathogenic_gene, lof_row = lof

    missense = pick_missense_rows(mane_gene_to_tx, rng, quota=39, exclude={pathogenic_gene})
    if missense is None:
        return None

    all_picked = [(pathogenic_gene, lof_row)] + missense
    rng.shuffle(all_picked)
    rows = [row for _, row in all_picked]
    return pathogenic_gene, rows


def main():
    parser = argparse.ArgumentParser(description="Generate one synthetic trio variant profile from gnomAD v4")
    parser.add_argument("-o", "--output", default="synthetic_profile.csv", help="Output CSV path")
    parser.add_argument("--seed", type=int, default=None, help="Random seed for reproducibility")
    parser.add_argument("--keep-enst", action="store_true", help="Skip ENST->NM_ conversion, keep Ensembl transcript IDs")
    args = parser.parse_args()

    result = generate_profile(args.seed)
    if result is None:
        logger.error("Failed to generate profile.")
        sys.exit(1)

    pathogenic_gene, rows = result
    logger.info("Pathogenic gene: %s — writing %d variants to %s", pathogenic_gene, len(rows), args.output)

    with open(args.output, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(rows)

    logger.info("Wrote %d variants to %s", len(rows), args.output)


if __name__ == "__main__":
    main()
