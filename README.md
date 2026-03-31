# CT-to-MRI-PETQ-UQ

> **MRI-Free Brain PET Quantification** via Diffusion-Based CT-to-MRI Synthesis with Uncertainty Quantification

---

## Overview

Standard amyloid PET quantification (Centiloid scale) requires **paired MRI** for brain parcellation — but ~30% of patients cannot undergo MRI. This research develops **SURE-CL**, a framework that synthesizes MRI from CT using a Brownian Bridge Diffusion Model (BBDM) and provides statistically certified uncertainty estimates.

```
CT scan
  └─▶ [VQ-VAE Encoder]
        └─▶ [BBDM: Brownian Bridge Diffusion]  ← Choo et al. 2024
               └─▶ Synthetic MRI + Pixel-wise Uncertainty Map
                      └─▶ [SynthSeg Parcellation]
                             └─▶ Centiloid + Conformal Prediction Interval
```

**Target:** ICC > 0.97 vs. MRI-derived Centiloid | 90% coverage guarantee | 8–12% flag rate

---

## Repository Contents

| File | Description |
|------|-------------|
| [`Knowledge_Base_MRI_Free_Brain_PET.md`](Knowledge_Base_MRI_Free_Brain_PET.md) | 150+ papers: CT-to-MRI synthesis, Centiloid, AD neuroimaging, BBDM, Mamba |
| [`Research_Proposal_SURE_CL.md`](Research_Proposal_SURE_CL.md) | Full proposal (~11,000 words): Background, Aims 1–3, Methods, Architecture |
| [`UQ_for_BBDM_papers/UQ_for_BBDM_Knowledge_Base.md`](UQ_for_BBDM_papers/UQ_for_BBDM_Knowledge_Base.md) | 48 papers: BayesDiff, Conformal Prediction, Heteroscedastic UQ, SDE theory |

---

## Key Methods

| Component | Method | Paper |
|-----------|--------|-------|
| CT→MRI Synthesis | Brownian Bridge Diffusion (BBDM) | Li et al., CVPR 2023 |
| 3D Architecture | VQ-VAE + Mamba SSM | Choo et al. 2024 |
| Aleatoric UQ | Heteroscedastic head (log σ²) | Kendall & Gal, NeurIPS 2017 |
| Epistemic UQ | K=20 stochastic samples | BayesDiff, ICLR 2024 |
| Calibration | Conformalized Quantile Regression | Romano et al., NeurIPS 2019 |
| Risk Control | Risk-Controlling Prediction Sets | Bates et al., JACM 2023 |
| Parcellation | SynthSeg (contrast-agnostic) | Billot et al., MedIA 2023 |

---

## jungwoooseo 개인 레포와의 연결

이 레포는 [@jungwoooseo](https://github.com/jungwoooseo) 의 연구를 Transconnectome 조직 레포로 관리합니다.  
개인 작업 레포: [jungwoooseo/CT-to-MRI-PETQ-UQ](https://github.com/jungwoooseo/CT-to-MRI-PETQ-UQ)

---

*Last updated: 2026-03-31*

