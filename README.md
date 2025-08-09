# NTCP Analysis and Machine Learning Pipeline

A five-stage Python pipeline for DVH processing, dose metrics and visualization, NTCP modeling (LKB/RS),
machine-learning prediction (ANN/XGBoost), output QA checks, and clinical factor analyses.
Designed for reproducible research and publication-quality outputs (600 dpi figures, tidy tables).

**Maintainer:** K. Mondal  
**Version:** v1.0.1  
**License:** MIT

## Repository Contents
- `code1_dvh_preprocess.py` — Parse TPS DVH text exports, standardize outputs; generates `cDVH_csv/` and `dDVH_csv/` plus workbook summary.
- `code2_dvh_plot_and_summary.py` — Compute dose metrics and create cDVH/dDVH plots; writes cohort summary tables.
- `code3_ntcp_analysis_ml.py` — Compute NTCP (LKB log-logit, LKB probit, RS) and train/evaluate ML models (ANN/XGBoost).
- `code4_ntcp_output_QA_reporter.py` — QA the analysis outputs; flags inflated patient counts, unrealistic NTCPs, and overfitting/leakage symptoms; generates a DOCX report.
- `code5_ntcp_factors_analysis.py` — Merge clinical factors with NTCP outputs; perform categorical/continuous analyses and plots.

## Install
```bash
python -m venv .venv
# Windows: .venv\Scripts\activate
# Linux/Mac: source .venv/bin/activate
pip install -r requirements.txt
```

## Quick Start
1) Preprocess DVH
```bash
python code1_dvh_preprocess.py --src ./raw_DVH --dst ./processed_DVH
```
2) Dose metrics & plots
```bash
python code2_dvh_plot_and_summary.py --cdvh_dir ./processed_DVH/cDVH_csv --outdir ./analysis_out
```
3) NTCP + ML analysis
```bash
python code3_ntcp_analysis_ml.py --dDVH_dir ./processed_DVH/dDVH_csv --clinical_xlsx ./clinical_input.xlsx --outdir ./analysis_out
```
4) QA of outputs
```bash
python code4_ntcp_output_QA_reporter.py --input ./analysis_out --report_outdir ./QA_results
```
5) Clinical factors analysis
```bash
python code5_ntcp_factors_analysis.py --input_file ./clinical_input.xlsx --enhanced_output_dir ./analysis_out
```

## Citation (recommend Zenodo DOI)
Archive a GitHub release with Zenodo and cite the DOI in your manuscript.
```
Mondal, K., et al. (2025). NTCP Analysis and Machine Learning Pipeline (v1.0.1) [Computer software]. Zenodo. https://doi.org/10.5281/zenodo.xxxxxxx
```


> **Private Pre‑Submission Build (for editors/reviewers)**  
> This repository is kept **private** until manuscript acceptance. Please do not redistribute.
> See **REVIEWERS.md** for a quick, step‑by‑step runbook.


