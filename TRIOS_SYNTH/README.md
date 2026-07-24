# TRIOS_SYNTH — Synthetic Trio Profile Generator

Generates synthetic trio variant CSVs for testing/benchmarking the ServerQwen
pipeline (`../server_qwen.py` / `../Qwen_Engine_GENOVA2I/genova_vllm_556_0610`)
without needing real patient data. Each profile is built from real gnomAD v4
allele frequencies, so scores/frequencies look realistic, but the trio itself
(proband + 2 "parents") is fabricated.

## What a profile is

Each `profile_N.csv` = 40 variants, mimicking one exome trio:

- **1 causative LoF variant** (nonsense, frameshift, or splice; gnomAD AF <
  0.1%) in a gene randomly picked from `DOMINANT_ID_GENES` — a curated list
  of genes known to cause **autosomal-dominant intellectual disability** when
  lost. This is the variant the pipeline is expected to flag as causative.
- **39 missense "noise" variants** (gnomAD AF < 1%), one gene each, randomly
  picked from `CONTROL_MIXED_GENES` — a pool of genes for *unrelated*
  diseases (cardiac, retinal, metabolic, cancer-predisposition, etc.) with no
  connection to intellectual disability. These exist to test that the
  pipeline correctly triages them out instead of over-calling.

Every variant is randomly assigned as inherited from one parent
(`Allelic_balance_1`/`Allelic_balance_2` — one is 0.5, the other 0) with the
proband always heterozygous (`Allelic_balance_proband=0.5`). Output columns
match the pipeline's expected CSV schema exactly (see `CLAUDE.md` in the
pipeline package for the canonical field list):

```
Variant,Chromosome,Position,RS_ID,Ref_seq,Var_seq,Type,HGVS,Zygosity,Gene,
Allelic_balance_proband,Allelic_balance_1,Allelic_balance_2
```

gnomAD returns Ensembl transcript IDs (`ENST...`); the script converts these
to RefSeq `NM_...` via the NCBI MANE Select map whenever a mapping exists
(`--keep-enst` to skip this and leave ENST IDs as-is).

## Requirements

Same Python env used to run the pipeline works fine here — the script only
needs `requests` (stdlib otherwise). No GPU/vLLM/SLM involved; this only
talks to the public gnomAD GraphQL API and the NCBI MANE FTP file.

## Generating a profile

```bash
cd TRIOS_SYNTH
python create_profiles.py -o profile_11.csv
python create_profiles.py -o profile_11.csv --seed 42   # reproducible
python create_profiles.py -o profile_11.csv --keep-enst # skip NM_ conversion
```

Takes roughly 1-2 minutes (one gnomAD API call per gene tried, `REQUEST_DELAY
= 0.5s` between calls, plus retries if a gene has no qualifying variant).
Logs the picked causative gene and its consequence/AF to stdout.

## Running a profile through the pipeline

Submit like any other CSV, with a phenotype consistent with the
`DOMINANT_ID_GENES` pool (e.g. "intellectual disability"):

```bash
curl -s -X POST http://<node>:<port>/analyze \
  -F "csv_file=@TRIOS_SYNTH/profile_11.csv" \
  -F "patient_report=intellectual disability"
```

The pipeline should name the fabricated causative gene (logged by
`create_profiles.py` at generation time) as the top causative finding, and
triage/VUS-out the 39 unrelated missense variants. Compare the report's
Section 2 gene against the generator's log line to check the pipeline "got
it right."

## Adapting this to a different disease

The generator has no disease-specific logic baked in beyond the two gene
lists — swapping disease area is just editing `DOMINANT_ID_GENES` (and
optionally `CONTROL_MIXED_GENES`) in `create_profiles.py`:

1. **`DOMINANT_ID_GENES`** (line ~60) — replace with genes known to cause
   your target disease via a dominant LoF/haploinsufficiency mechanism (the
   generator specifically hunts for a nonsense/frameshift/splice variant in
   one of these — it will not find one in a gene where LoF isn't the disease
   mechanism, e.g. a gene that only causes disease via gain-of-function
   missense).
2. **`CONTROL_MIXED_GENES`** (line ~67) — the "noise" pool. Keep this
   disjoint from your new `DOMINANT_ID_GENES` list (checked automatically via
   `exclude={pathogenic_gene}` in `generate_profile()`, but don't duplicate a
   gene across both lists) and make sure it doesn't accidentally contain
   genes also linked to your target phenotype — that would make a "noise"
   variant a legitimate confounder rather than true noise.
3. **Phenotype text** used when submitting the CSV to `/analyze` — pick
   wording that actually matches what your new `DOMINANT_ID_GENES` genes are
   established to cause, mirroring how `"intellectual disability"` maps to
   the current gene list.

For a **recessive** disease instead of dominant, more than gene-list edits
are needed — `build_csv_row()` (line ~150) always emits `Zygosity=Heterozygous`
with exactly one parent carrying each variant. Generating a true recessive
case (homozygous, or compound-het in trans) requires calling `pick_variant()`
twice for the same gene and writing both parents as heterozygous carriers
(AB=0.5) with the proband homozygous (AB~1.0), or fabricating a compound-het
pair with each variant from a different parent — this script's current
`generate_profile()` doesn't do either, so treat that as new code, not a
config change.

For **X-linked** disease genes, also fine as a gene-list swap as long as the
gene itself is X-linked (`Chromosome=chrX` comes naturally from gnomAD
coordinates) — but note the generator doesn't model male/female proband sex,
so hemizygous-in-male zygosity isn't represented; every variant is written
`Heterozygous` regardless of chromosome.

## Files

| File | Purpose |
|---|---|
| `create_profiles.py` | Generator script (see above) |
| `profile_*.csv` | Generated synthetic trio profiles, ready to submit to `/analyze` |
