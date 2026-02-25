## v1.0.1 – RS bug fix and input loading

- **Relative Seriality (RS) model**:
  - Corrected the voxel-level complication probability to include the Euler number factor as described by Källman et al.:
    \( p_i = 2^{-\exp(e \cdot \gamma (1 - D_i / D_{50}))} \).
  - This aligns the implementation with Eq. (3) in the J Med Phys 2026 manuscript and the QUANTEC-derived RS parameters in Supplementary Table 2.
  - Verified that key reported metrics in the manuscript (AUC, Brier scores, gEUD distributions) for parotid and larynx remain unchanged within numerical noise.

- **Multi-sheet clinical input support**:
  - Updated `load_patient_data` to read all sheets from `ntcp_analysis_input.xlsx` (`Parotid`, `Larynx`, `SpinalCord`) and, when needed, infer the `Organ` label from the sheet name.
  - This ensures the open-source pipeline processes all three organs exactly as described in the paper.

- **Versioning**:
  - Bumped Python package version in `pyproject.toml` to `1.0.1` to match the Zenodo metadata and manuscript title.

