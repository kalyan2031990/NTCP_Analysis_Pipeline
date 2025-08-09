
# Zenodo Embargo Plan (Reserve DOI Only)

**Goal:** Reserve a DOI now without making files public. Enable full public access after acceptance.

## Steps
1. Go to https://zenodo.org/ and create a **New upload** (not GitHub‑link; we'll upload the ZIP manually for private review).
2. In the upload form:
   - **Upload type:** Software
   - **Title:** NTCP Analysis and Machine Learning Pipeline (v1.0.1)
   - **Creators:** Mondal, K.; et al.
   - **Description:** Analysis pipeline for DVH preprocessing, dose metrics, classical NTCP models (LKB/RS), ML modules (ANN/XGBoost), QA reporting, and clinical factor analyses.
   - **Version:** v1.0.1
   - **License:** MIT
   - **Keywords:** radiotherapy, NTCP, DVH, LKB, Relative Seriality, machine learning
   - **Access right:** **Embargoed**
   - **Embargoed until:** YYYY‑MM‑DD (e.g., set to 12 months ahead, or the expected publication date)
3. Click **Reserve DOI** to obtain a DOI without publishing the record.
4. Optionally upload this repository ZIP (`NTCP_Analysis_Pipeline_private_review_v1.0.1.zip`) as the embargoed file. (Files remain hidden until you lift the embargo.)
5. Add **Related identifiers** (when available):
   - **isSupplementTo:** DOI of the accepted article (to be added post‑acceptance).
6. On acceptance:
   - Switch **Access right** to **Open** and **Publish** the record to activate the DOI and make files publicly available.

## Notes
- The DOI will be valid and citable once published (after lifting embargo). During review, only the metadata (not files) may be visible, depending on your Zenodo settings.
- If your journal asks for a DOI at submission, include the **reserved DOI** in the manuscript; clarify that access is embargoed during peer review.
