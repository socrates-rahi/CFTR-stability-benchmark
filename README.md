# Benchmarking FoldX, Rosetta and DDMut on CFTR nucleotide-binding-domain stability

Computational benchmark of protein-stability predictors against experimental ΔTm for CFTR NBD1/NBD2 mutations
(Bahia et al., 2021). Companion to the BBD451 B.Tech Major Project report (IIT Delhi, Nov 2026).

## What was done
- 59 reference mutations transcribed from Bahia et al. (2021), Tables 1-2 (`data/bahia_reference_59.csv`).
- **DDMut predictions generated for 43/59 mutations** via the public DDMut web server (forward + reverse),
  raw PDB accessions 2PZE chain A (NBD1) and 6UK1 chain A (NBD2, 9sol background). Raw outputs and provenance: `data/ddmut_raw/`.
- Leave-one-position-out cross-validation with bootstrap CIs over residue positions (`scripts/consensus_model.py`).
- Exploratory screening/triage analysis and a small logistic-regression triage classifier (`scripts/triage_classifier.py`).

## Key results (n = 43)
- Cross-validated Spearman rho: FoldX 0.69, Rosetta hrp 0.57, Rosetta lrp 0.53, DDMut 0.36.
- Consensus models did not beat FoldX alone. The logistic triage classifier (AUC 0.87) did not beat FoldX alone (0.90).
- DDMut forward/reverse anti-symmetry holds on CFTR (r = -0.92).

## What was NOT done / limitations
- FoldX and Rosetta values are literature values (Bahia et al.), not recomputed.
- 15 NBD2 mutations (other construct backgrounds) and Q637R were not run through DDMut.
- DDMut run on raw (unprepared) structures; no check of training-set leakage; small n; one protein family.
- This is a benchmark and analysis, not a new stability predictor.

## Reproduce
```
pip install -r requirements.txt
python3 scripts/consensus_model.py data/benchmark.csv --features foldx,rosetta_hrp,rosetta_lrp,ddmut_fwd --flip ddmut_fwd --boot 2000 --out results
python3 scripts/triage_classifier.py data/benchmark.csv
```
