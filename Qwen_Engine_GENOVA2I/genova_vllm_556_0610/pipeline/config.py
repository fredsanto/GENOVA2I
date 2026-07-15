"""
pipeline/config.py — Global pipeline configuration constants.
"""

EXECUTION_MODE = "async"   # "async" | "sequential"

DEFAULT_GENOME_BUILD = "hg19"  # fallback; override per-run when input format is known

TRIAGE_ENABLED_THRESHOLD  = 12   # skip first_triage if n_variants <= this
