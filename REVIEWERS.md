
# Reviewer Runbook (Private Pre‑Submission)

**Package:** NTCP Analysis and Machine Learning Pipeline (v1.0.1)  
**Maintainer:** K. Mondal et al.  
**Status:** Private pre‑submission; please do not redistribute.

## Environment
- Python 3.10+
- Install dependencies:
  ```bash
  python -m venv .venv
  # Windows: .venv\Scripts\activate
  # Linux/Mac: source .venv/bin/activate
  pip install -r requirements.txt
  ```

## Minimal Reproduction (without PHI)
> Use your institution’s anonymized/synthetic DVH examples. No patient‑identifiable data is included here.

1. **Preprocess DVH**
   ```bash
   python code1_dvh_preprocess.py --src ./raw_DVH --dst ./processed_DVH
   ```
2. **Dose Metrics & Plots**
   ```bash
   python code2_dvh_plot_and_summary.py --cdvh_dir ./processed_DVH/cDVH_csv --outdir ./analysis_out
   ```
3. **NTCP + ML**
   ```bash
   python code3_ntcp_analysis_ml.py --dDVH_dir ./processed_DVH/dDVH_csv --clinical_xlsx ./clinical_input.xlsx --outdir ./analysis_out
   ```
4. **QA Report**
   ```bash
   python code4_ntcp_output_QA_reporter.py --input ./analysis_out --report_outdir ./QA_results
   ```
5. **Clinical Factors**
   ```bash
   python code5_ntcp_factors_analysis.py --input_file ./clinical_input.xlsx --enhanced_output_dir ./analysis_out
   ```

## Notes
- Scripts are designed to run modularly. See `--help` of each script for options.
- Please report any reproducibility issues with the command used and console output.
- Licensing: MIT (see LICENSE). Repository visibility is private pre‑submission.
