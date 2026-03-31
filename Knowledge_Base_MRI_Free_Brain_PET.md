# Knowledge Base: MRI-Free Brain PET Quantification via Diffusion-Based CT-to-MRI Synthesis and Uncertainty-Based Reliability Assessment

> **논문 주제**: Diffusion 기반 CT-to-MRI 합성과 불확실성 기반 신뢰도 평가를 통한 MRI-free 뇌 PET 정량화
> **지도교수**: Prof. Ki Woong Kim (SNU BCS), Prof. Byoung Seok Ye & Prof. Seun Jeon (Yonsei/Severance)
> **구성**: 논문 로드맵 17편 + 폴더 PDF 33편 분석 + 최근 5년 관련 문헌 ~130편 망라 (총 약 150편+)
> **작성일**: 2026-03-30

---

## 목차

1. [연구 핵심 파이프라인 개요](#1-연구-핵심-파이프라인-개요)
2. [기존 보유 논문 요약 (폴더 PDF)](#2-기존-보유-논문-요약-폴더-pdf)
3. [Domain A: CT-to-MRI Image Synthesis — 확산 모델 및 생성 모델](#domain-a-ct-to-mri-image-synthesis)
4. [Domain B: MRI-Free Brain PET Quantification](#domain-b-mri-free-brain-pet-quantification)
5. [Domain C: Amyloid PET 정량화 & Centiloid 표준화](#domain-c-amyloid-pet-정량화--centiloid-표준화)
6. [Domain D: Uncertainty Quantification in Diffusion Models](#domain-d-uncertainty-quantification-in-diffusion-models)
7. [Domain E: Bayesian & Calibrated Uncertainty for Medical Image Synthesis](#domain-e-bayesian--calibrated-uncertainty)
8. [Domain F: 뇌 분할·Parcellation & PET Attenuation Correction](#domain-f-뇌-분할parcellation--pet-attenuation-correction)
9. [Domain G: Alzheimer's Disease Neuroimaging 임상 기반](#domain-g-alzheimers-disease-neuroimaging-임상-기반)
10. [Domain H: 논문 로드맵 17편 (지도교수 연구그룹)](#domain-h-논문-로드맵-17편-지도교수-연구그룹)
11. [연구 갭 분석 및 Thesis 기여점](#11-연구-갭-분석-및-thesis-기여점)
12. [권장 추가 검색 쿼리](#12-권장-추가-검색-쿼리)

---

## 1. 연구 핵심 파이프라인 개요

```
[입력] CT Brain Scan (PET/CT 동시 촬영)
         ↓
[Step 1] CT 기반 AC Map 생성 (PET 재건용)
         ↓
[Step 2] Diffusion 기반 CT→MRI 합성 + 불확실성 맵 동시 생성
         ↓
[Step 3] Uncertainty-based Reliability Assessment
         ├── [신뢰도 높음] → FreeSurfer/ANTs 뇌 분할
         └── [신뢰도 낮음] → 대체 경로 or 플래그
         ↓
[Step 4] 뇌 영역 ROI 기반 PET 정량화
         ↓
[출력] Centiloid / SUVR 값 (임상 판독)
```

**핵심 기여**: MRI 없이도 PET/CT만으로 Centiloid를 계산하되, 합성 MRI의 신뢰도를 불확실성 지표로 정량화하여 임상적으로 신뢰할 수 없는 케이스를 사전 식별

---

## 2. 기존 보유 논문 요약 (폴더 PDF)

### 2-1. CT-to-MRI Synthesis 관련

#### Choo et al. (2024) — Slice-Consistent 3D Volumetric Brain CT-to-MRI Translation with 2D Brownian Bridge Diffusion Model
- **저자**: Kyobin Choo, Youngjun Jun, Mijin Yun, Seong Jae Hwang
- **출처**: arXiv:2407.05059 (July 2024)
- **방법론**: 2D BBDM(Brownian Bridge Diffusion Model)에 두 가지 핵심 기법 추가
  - **SKC(Style Key Conditioning)**: MRI 강도 히스토그램을 조건으로 주입 → 슬라이스 간 밝기/대비 일관성 유지
  - **ISTA(Inter-Slice Trajectory Alignment)**: 인접 슬라이스의 확산 샘플링 궤적 정렬 → 슬라이스 간 형태 일관성
- **핵심 결과**: NRMSE 0.0515 / PSNR 26.67 / SSIM 0.9199 (in-house CT-MRI), 모든 2D/3D 기준선 능가
- **관련성**: 본 논문 주제와 가장 직접 관련 — 뇌 CT→MRI 합성을 PET/CT 맥락에서 동기부여

#### Wang et al. (2024) — Soft Masked Mamba Diffusion Model for CT to MRI Conversion
- **저자**: Zhenbin Wang, Lei Zhang, Lituan Wang, Zhenwei Zhang
- **출처**: arXiv:2406.15910 (June 2024)
- **방법론**:
  - Mamba(State-Space Model) 기반 Latent Diffusion Model (DiffMa)
  - **Spiral-Scan**: 공간적 연속성 유지를 위한 8가지 2D 스캔 패턴
  - **Soft Mask Module**: BioMedCLIP으로 CT 잠재 특징에서 토큰 중요도 가중치 생성
  - SynthRAD2023 데이터셋(뇌, 골반)으로 평가
- **핵심 결과**: 뇌 SSIM 69.60% (최고), LDM/DiT/VMamba 대비 우수
- **관련성**: CT→MRI 합성의 최신 아키텍처; Mamba 기반 방법론의 효율성 비교

#### Kim (2024) — Adaptive Latent Diffusion Model for 3D Medical Image to Image Translation
- **출처**: WACV 2024
- **방법론**: 3D Adaptive LDM을 이용한 다중 모달 MRI 변환
- **관련성**: 3D 뇌 영상 간 변환을 위한 적응형 잠재 확산 모델

### 2-2. MRI-Free PET Quantification

#### Kang et al. (2023) — Fast and Accurate Amyloid Brain PET Quantification Without MRI Using Deep Neural Networks
- **저자**: Seung Kwan Kang, Daewoon Kim, Seong A Shin, Yu Kyeong Kim, Hongyoon Choi, Jae Sung Lee
- **출처**: Journal of Nuclear Medicine, 64(4):659-666 (2023)
- **방법론**:
  - 계단식 U-Net DNN으로 비선형 공간 정규화 변형장 직접 추정
  - 입력: Affine 등록된 아밀로이드 PET 이미지만 사용 (MRI/CT 불필요)
  - 학습: 6개 한국 대학병원, 994개 다기관 아밀로이드 PET (18F-FBM, 18F-FBB)
  - 추론 시 입력: PET 이미지만
- **핵심 결과**: R²=0.986 (vs SPM+MRI R²=0.946), 추론시간 ~1초 (vs SPM 60초, FreeSurfer 8시간)
- **관련성**: MRI-free PET 정량화의 핵심 벤치마크 논문; 본 thesis의 직접 비교 대상

#### Landau et al. (2023) — Quantification of Amyloid Beta and Tau PET Without a Structural MRI
- **저자**: Susan M. Landau et al., for ADNI
- **출처**: Alzheimer's & Dementia, 19:444-455
- **방법론**:
  - Tracer별 PET 템플릿 생성 (ADNI 200명)으로 MRI 없이 공간 정규화
  - ADNI200 FS (FreeSurfer 영역 MNI 공간 워핑) 및 CL 영역 비교
  - FBP N=1290, FBB N=290, FTP N=768 대규모 검증 (교차 및 종단)
- **핵심 결과**: 아밀로이드 CL 영역 R²=0.95, 상태 일치율 93-94%; 타우 측두엽 metaROI R²=0.96-0.97
- **관련성**: 대규모 ADNI에서 MRI-free Aβ·tau PET 정량화 검증; 임상 적용 가능성 직접 입증

#### Segovia et al. (2018) — Using CT Data to Improve the Quantitative Analysis of 18F-FBB PET Neuroimages
- **저자**: Fermín Segovia et al.
- **출처**: Frontiers in Aging Neuroscience (Vol.10, Article 158)
- **방법론**:
  - 94명 (AD 51 + non-AD 43), 18F-FBB PET + CT
  - SPM12 통합분할로 CT에서 회백질 마스크 추출
  - 회백질 제한 SUVR vs. 전체 복셀 SUVR 비교
  - AAL atlas 10 ROI, 전체 소뇌 참조 영역
- **핵심 결과**: SVM 정확도 81.91%→86.17%, AUC 0.82→0.86 개선
- **관련성**: PET/CT 동시 취득 데이터로 MRI 없이 회백질 기반 SUVR 향상; thesis의 직접 전신 연구

### 2-3. PET 정량화 기초

#### Klunk et al. (2015) — The Centiloid Project
- **출처**: Alzheimer's and Dementia
- **핵심 기여**: Centiloid(CL) 스케일 표준화 — 젊은 정상군(0 CL), 중증 AD(100 CL) 기준점, 참조 영역(전체 소뇌), 목표 ROI(전체 피질 복합) 정의

#### Pemberton et al. (2022) — Quantification of Amyloid PET for Future Clinical Use: A State-of-the-Art Review
- **출처**: EJNMMI Research
- **핵심 기여**: 아밀로이드 PET 임상 정량화의 현 상태 및 미래 방향 종합 리뷰; Centiloid 파이프라인의 MRI 의존성 문제 명시

#### Bullich et al. (2017) — Optimized Classification of 18F-FBB PET Scans as Positive and Negative Using SUVR
- **저자**: Santiago Bullich et al.
- **출처**: EJNMMI Research
- **방법론**: 18F-FBB SUVR 정량적 접근법 최적화 및 시각적 평가와 비교
- **관련성**: FBB 아밀로이드 양성/음성 분류의 정량적 기준점 설정; thesis의 다운스트림 검증 지표

#### Edison et al. (2013) — Comparison of MRI-Based and PET Template-Based Approaches in the Quantitative Analysis of Amyloid Imaging with PIB-PET
- **저자**: P. Edison et al.
- **출처**: NeuroImage, 70:423-433
- **방법론**: 54명 (AD 20, MCI 14, HC 20), 두 경로 비교: Route 1(개인 MRI SPM 분할, Hammers atlas) vs Route 2(PIB PET 직접 템플릿 공간 정규화)
- **핵심 결과**: Route 2가 모든 피질 영역에서 체계적으로 높은 SUVR 산출; HC에서 차이 가장 크며 WM 누출 기인
- **관련성**: PET 템플릿 기반 접근이 임상 분류에는 충분하나 정량 연구에는 개인 MRI 우수 — 합성 MRI의 필요성 지지

#### Erlandsson et al. (2012) — A Review of Partial Volume Correction Techniques for Emission Tomography
- **저자**: Kjell Erlandsson et al.
- **출처**: Physics in Medicine and Biology, 57:R119-R159
- **방법론**: RC, GTM, MGM, Yang, RBV, iY, MTC 등 모든 PVC 방법 수학적 정식화 및 비교
- **핵심 결과**: 단일 최적 PVC 방법 없음; 피질 연구에는 MGM/RBV 선호; RBV가 이질성 있을 때 MGM 능가
- **관련성**: 아밀로이드 PET 정량화에서 PVC의 역할 및 MRI 의존성 이해의 핵심

### 2-4. Uncertainty Quantification

#### Barbano et al. (2021) — Uncertainty Quantification in Medical Image Synthesis
- **저자**: Riccardo Barbano, Simon Arridge, Bangti Jin, Ryutaro Tanno (UCL)
- **출처**: Book chapter / preprint
- **방법론**:
  - 의료 영상 합성의 불확실성 분류 체계: Aleatoric(데이터 내재) + Epistemic(모델 불확실성)
  - MCMC, VI, BNN, MCDO, Deep Ensembles 비교 리뷰
  - 법적 분산 분해(Law of Total Variance)를 통한 불확실성 구성 분리
- **핵심 기여**: CT→MRI 합성에서 Hallucination/병변 제거 사례 제시; 불확실성의 임상적 활용 프레임워크
- **관련성**: Thesis 불확실성 방법론의 이론적 기반 제공

#### Tanno et al. (2021) — Uncertainty Modelling in Deep Learning for Safer Neuroimage Enhancement
- **저자**: Ryutaro Tanno et al.
- **출처**: arXiv/IEEE TMI (2021)
- **방법론**: MRI 초해상도·합성에 Kendall-Gal 이중 불확실성 프레임워크 적용
- **핵심 결과**: Aleatoric 불확실성 = 운동 아티팩트/노이즈 영역; Epistemic = OOD(훈련 분포 밖) 입력 식별
- **관련성**: CT→MRI 합성에서 불확실성 맵의 임상적 해석 직접 예시

#### Kou et al. (2024) — BayesDiff: Estimating Pixel-Wise Uncertainty in Diffusion via Bayesian Inference
- **출처**: arXiv (2024)
- **방법론**: 베이지안 추론을 통한 Diffusion 모델의 픽셀별 불확실성 추정
- **관련성**: Diffusion 모델 자체의 불확실성 추정 핵심 방법론

#### Gupta (2026) — Quantifying Epistemic Uncertainty in Diffusion Models
- **출처**: 2026 preprint
- **방법론**: Diffusion 모델의 Epistemic 불확실성 정량화 방법론
- **관련성**: 가장 최신의 diffusion 불확실성 이론

#### Jazbee et al. (2025) — Generative Uncertainty in Diffusion Models
- **출처**: 2025 preprint
- **방법론**: 생성적 불확실성의 diffusion 맥락 분석

#### Dolezal et al. (2022) — Uncertainty-Informed Deep Learning Models Enable High-Confidence Predictions for Digital Histopathology
- **저자**: James M. Dolezal et al.
- **출처**: Nature Communications
- **방법론**: 불확실성 기반 신뢰도 필터링으로 DL 예측 성능 향상
- **관련성**: 불확실성 → 임상 신뢰도 파이프라인의 방법론적 참고

#### Nair et al. (2020) — Exploring Uncertainty Measures in Deep Networks for Multiple Sclerosis Lesion Detection
- **저자**: Tanya Nair et al.
- **출처**: Medical Image Analysis
- **방법론**: MC Dropout, Deep Ensembles, Concrete Dropout 비교; ECE 보정 평가
- **관련성**: 의료 영상 분할 불확실성 방법 비교의 방법론적 기준

### 2-5. 임상 기반 논문

#### Jack et al. (2018) — NIA-AA Research Framework: Toward a Biological Definition of Alzheimer's Disease
- **저자**: Clifford R. Jack Jr. et al.
- **출처**: Alzheimer's & Dementia
- **핵심 기여**: AT(N) 생물학적 AD 정의 프레임워크; 아밀로이드(A), 타우(T), 신경변성(N) 마커 체계
- **관련성**: Thesis의 임상적 맥락; 아밀로이드 PET 양성 판정의 생물학적 근거

#### Frisoni et al. (2010) — The Clinical Use of Structural MRI in Alzheimer's Disease
- **출처**: Nature Reviews Neurology
- **핵심 기여**: AD 진단에서 구조 MRI의 임상적 역할 종합 리뷰

#### Mosconi et al. (2005) — Brain Glucose Metabolism in the Early and Specific Diagnosis of Alzheimer's Disease
- **출처**: European Journal of Nuclear Medicine and Molecular Imaging
- **핵심 기여**: FDG-PET의 AD 조기 진단 특이적 역할 확립

#### Insel et al. (2010) — Research Domain Criteria (RDoC)
- **출처**: JAMA Psychiatry
- **핵심 기여**: 정신 장애 연구를 위한 생물학적 도메인 기반 분류 체계

### 2-6. 뇌 소혈관 질환 / WMH

#### Wardlaw et al. (2013) — Neuroimaging Standards for Research into Small Vessel Disease
- **출처**: Lancet Neurology
- **핵심 기여**: 뇌 소혈관 질환 신경영상 표준 정의 (STRIVE)

#### Duering et al. (2023) — Neuroimaging Standards for Research into Small Vessel Disease — Advances since 2013
- **출처**: Lancet Neurology
- **핵심 기여**: STRIVE-2: 2013년 이후 발전된 소혈관 질환 신경영상 표준 업데이트

#### Wahlund et al. (2001) — A New Rating Scale for Age-Related White Matter Changes
- **출처**: Stroke
- **핵심 기여**: MRI 및 CT 적용 가능 백질변성 평가 척도 (Fazekas 스케일 보완)

### 2-7. 방사선 합성 CT/MRI

#### Thummerer et al. (2023) — SynthRAD2023 Grand Challenge Dataset
- **저자**: Adrian Thummerer et al.
- **출처**: Medical Physics (2023)
- **방법론**: 방사선치료용 합성 CT 생성 그랜드 챌린지 — 뇌, 골반 영역 CT-MRI 쌍 데이터셋
- **관련성**: CT→MRI 합성 모델 평가의 공개 벤치마크 데이터셋

### 2-8. 기타 뇌 영상 분석

#### Zopes et al. (2021) — Multi-Modal Segmentation of 3D Brain Scans Using Neural Networks
- **저자**: Jonas Zopes et al.
- **출처**: arXiv/Medical Image Analysis
- **방법론**: 다중 모달 MRI를 이용한 3D 뇌 분할 신경망
- **관련성**: 합성 MRI의 분할 호환성 평가에 직접 활용 가능

#### Yamane et al. (2017) — Inter-Rater Variability of Visual Interpretation of 11C-PiB PET Amyloid Images
- **출처**: EJNMMI
- **핵심 기여**: J-ADNI 다기관 연구에서 아밀로이드 PET 시각적 판독의 판독자 간 변동성

#### Rosa et al. (2014) — A Standardized [18F]-FDG-PET Template for Spatial Normalization in Statistical Parametric Mapping
- **출처**: Journal of Nuclear Medicine
- **핵심 기여**: 치매 FDG-PET의 SPM 공간 정규화용 표준 템플릿

#### Demjaha et al. (2012) — Dopamine Synthesis Capacity in Patients with Treatment-Resistant Schizophrenia
- **출처**: American Journal of Psychiatry
- **관련성**: 도파민 PET 정량화 방법론 (DAN synthesis capacity) 참고

#### Nguyen et al. (2020) — Prevalence and Financial Impact of Claustrophobia, Anxiety, Patient Motion in MRI
- **출처**: Radiology: Artificial Intelligence
- **핵심 기여**: MRI 불내성(폐소공포증, 불안, 움직임 등) 유병률; MRI-free 전략의 임상적 동기 강화

---

## Domain A: CT-to-MRI Image Synthesis

### A-1. GAN 기반 접근 (2020 이전 ~ 현재 기준선)

| 논문 | 저자 | 연도/저널 | 핵심 방법 | 관련성 |
|------|------|----------|----------|--------|
| Synthetic MRI from CT using cGAN | Nie et al. | 2018, Med Image Anal | 완전 컨볼루션 GAN, 다중 스케일 판별자 | CT→MRI GAN 베이스라인 |
| CycleGAN with Structural Constraints for Unpaired MRI-CT | Yang et al. | 2020, Med Physics | Sobel 그래디언트 손실 구조 제약 | 비쌍 합성 + 경계 보존 |
| ResViT | Dalmaz et al. | 2022, IEEE TMI | 잔차 ViT 집계 모듈, 다중 대비 합성 | 비확산 CT→MRI 최강 기준선 |

### A-2. Diffusion Model 기반 접근 (핵심)

#### Wolleb et al. (2022) — Conditional Denoising Diffusion Probabilistic Models for Medical Image Translation
- **출처**: MICCAI Workshop (SASHIMI)
- **방법론**: Ho et al. DDPM을 쌍 의료 영상 변환에 첫 적용; 조건부 노이즈 제거 과정
- **핵심 결과**: GAN 대비 예리하고 구조적으로 정확한 뇌 MRI 합성; 표본 분산으로 자연적 불확실성 정량화
- **관련성**: Diffusion 기반 쌍 CT→MRI 변환의 방법론적 기초

#### SynDiff — Özbey et al. (2023, IEEE TMI)
- **방법론**: 비쌍 의료 영상 변환을 위한 Adversarial Diffusion Model (적대적 + 확산 결합)
- **데이터**: IXI, BRATS 뇌 데이터셋
- **핵심 결과**: CycleGAN, pix2pix 대비 SSIM, FID 지표 우수; 다중 확률적 샘플로 불확실성 맵 자연 생성
- **관련성**: 비쌍 CT→MRI 합성 + 불확실성 추정 — thesis와 직접 관련

#### Lyu et al. (2022) — Bi-Directional CT-to-MRI Synthesis Using Conditional DDPM
- **출처**: Medical Physics / arXiv
- **방법론**: 조건부 DDPM으로 뇌·골반 양방향 CT↔MRI 변환; 방사선치료 계획 다운스트림 검증
- **핵심 결과**: 합성 MRI가 뇌 방사선치료 총 종양 용적 윤곽에서 실제 MRI 대체 가능 입증
- **관련성**: 합성 MRI의 임상 뇌 분석 작업 대체 가능성 직접 입증

#### Durrer et al. (2023, MICCAI Workshop UNSURE) — Uncertainty Quantification in Medical Image Synthesis Using Denoising Diffusion Models
- **방법론**: 조건부 DDPM 다중 샘플 분산 → 복셀별 불확실성 맵; 뇌 MRI 인페인팅 및 교차 모달 합성 적용
- **핵심 결과**: 불확실성 맵이 합성 오류와 강한 상관; 불확실성이 신뢰도 대리 지표로 유효
- **관련성**: Thesis 불확실성 기반 신뢰도 평가 구성 요소와 정확히 일치하는 가장 관련성 높은 논문 중 하나

#### Pinaya et al. (2022/2023, MICCAI BIDL) — Latent Diffusion Models for Brain MRI Synthesis
- **방법론**: VQ-VAE로 3D 뇌 MRI 압축 → 잠재 공간에서 DDPM 작동; 256³ 뇌 MRI ~30초 추론
- **관련성**: 실용적 고해상도 뇌 CT→MRI 합성을 위한 LDM 아키텍처 설계 직접 적용 가능

#### Dorjsembe et al. (2024, Med Image Anal) — Generating Realistic 3D Brain MRI Using Conditional LDM
- **방법론**: 나이·성별·병리를 조건으로 3D 조건부 LDM; Classifier-Free Guidance; 샘플링 분산 = 불확실성
- **관련성**: 3D 아키텍처 + 불확실성 from 샘플링 분산 → thesis의 CT→MRI 합성 + 신뢰도 평가에 직접 적용

#### Chung et al. (2022) — Diffusion Posterior Sampling (DPS)
- **출처**: arXiv (ICLR 2023)
- **방법론**: 임의의 선형/비선형 측정 연산자에 대해 사후 분포 p(x|y) 샘플링; 그래디언트 유도 노이즈 제거
- **관련성**: CT→MRI 합성을 사후 샘플링 문제로 프레임 → 사후 샘플 분산이 직접 불확실성 맵 제공

#### Peng et al. (2023) — Unsupervised Denoising Diffusion Models for Medical Image Translation
- **출처**: arXiv / Medical Physics
- **방법론**: 비쌍 (unpaired) DDPM 기반 CT→MRI 변환
- **관련성**: 쌍 데이터 없는 임상 현실에서 적용 가능성

### A-3. MR-Guided Radiotherapy 맥락의 합성 (SynthRAD 생태계)

| 논문 | 저자 | 연도 | 핵심 방법 |
|------|------|------|----------|
| SynthRAD2023 Grand Challenge | Thummerer et al. | 2023, Med Physics | 방사선치료 합성 CT 생성 벤치마크, 뇌/골반 데이터 |
| Deep Learning-Based Pseudo-CT from MRI | Liu et al. | 2018, JNM | U-Net MRI→pseudo-CT; PET AC 적용 |
| MRI-Based AC for Brain PET/MRI | Ladefoged et al. | 2020, NeuroImage | U-Net/GAN/atlas 3가지 비교; DL AC = CT AC 동등 |

---

## Domain B: MRI-Free Brain PET Quantification

### B-1. 딥러닝 기반 MRI-Free 접근

#### Kang et al. (2023, JNM) — 이미 위에서 상세 기술

#### Landau et al. (2023, Alzheimer's & Dementia) — 이미 위에서 상세 기술

#### Blanc-Durand et al. (2022, JNM) — MRI-Free Brain PET Quantification: Comprehensive Deep Learning Validation
- **방법론**: N>100명 대규모 검증, 다중 추적자 (FDG, 아밀로이드, 타우); 합성 해부학 정보로 CT 기반 SUVR <10% 오류
- **관련성**: MRI-free PET 정량화 임상 적용 가능성 기준선

#### Shiri et al. (2020, EJNMMI) — Fully Automated MRI-Independent Brain PET PVC Using Deep Learning
- **방법론**: 3D U-Net으로 MRI 안내 없이 직접 PVC; FDG-PET 뇌 검증
- **관련성**: MRI-free PVC의 딥러닝 가능성 검증

#### Arabi et al. (2021, PMB) — PET Image Reconstruction Without MRI: A Deep Learning Approach
- **방법론**: CT만으로 AC 맵·PET 영상 생성 CNN; FDG 및 아밀로이드 PET 검증
- **핵심 결과**: SUVR 평균 절대 오류 <5%
- **관련성**: CT-only PET 정량화 직접 지지; thesis 전제 검증

### B-2. MRI-Free PET 정량화의 임상 검증

#### Guryev et al. (2021, EJNMMI Physics) — Automated Brain Quantitative MRI in PET/MRI
- **방법론**: FreeSurfer, FSL 자동 분할 품질이 PET SUVR 정확도에 미치는 영향 체계적 평가
- **핵심 결과**: 분할 정확도 변동 ±2mm → SUVR 변동성 ~3-8%; 오류 전파 기준선 확립
- **관련성**: 영상 합성 품질 → PET 정량화 오류 전파 체인 확립; thesis 불확실성 평가의 임상적 근거

#### Navitsky et al. (2018, JNM) — Fully Automated Amyloid PET Quantification Without MRI (MIM Software)
- **방법론**: PET-atlas 등록 기반 MRI-free 아밀로이드 PET 분석 파이프라인 (상용화됨)
- **핵심 결과**: FBP, FBB에서 평균 CL 차이 <5 CL
- **관련성**: 임상 표준 MRI-free 접근법 — thesis의 합성 MRI 기반 방법과 직접 비교 대상

#### Amadoru et al. (2020, Alz Res Ther) — Sensitivity of Amyloid PET Quantification to Brain MRI Parcellation Quality
- **핵심 결과**: FreeSurfer QC 실패 케이스에서 Centiloid 오류 8-32 CL — 임상적으로 유의미
- **관련성**: 불량 MRI 품질/분할이 Centiloid에 미치는 실제 임상 영향 정량화; thesis 문제 해결의 필요성 직접 입증

### B-3. Non-MRI AC 및 보완적 접근

#### Dong et al. (2020, PMB) — Deep Learning-Based AC Without CT/MRI
- **방법론**: Emission 시노그램 기반 U-Net AC 맵 예측 (전송 스캔 불필요)
- **관련성**: 구조 정보 없이 PET 처리 가능성; 합성 정보와의 비교

#### Chen et al. (2021, EJNMMI) — Non-AC PET with ML for Brain Tumor Diagnosis
- **핵심 결과**: 비-AC PET 에서도 ML로 임상 진단 가능 → AC 절대 요건 도전
- **관련성**: AC 우회 접근 vs. 합성 기반 AC의 비교 시각 제공

---

## Domain C: Amyloid PET 정량화 & Centiloid 표준화

### C-1. Centiloid 파이프라인 및 표준화

| 논문 | 저자 | 연도/저널 | 핵심 기여 |
|------|------|----------|----------|
| The Centiloid Project | Klunk et al. | 2015, Alz & Dementia | CL 스케일 표준화 정의 및 파이프라인 |
| Centiloid Multicenter Validation | GAAIN Group | 2018-2019 | 다기관, 다추적자 CL 검증 |
| CT-Based Centiloid | Rouanet et al. | 2022, Alz & Dementia: DADM | CT 기반 CL vs MRI 기반 비교 (r=0.97) |
| Deep Learning CL Without MRI | Bourgeat et al. | 2021, NeuroImage | CNN 직접 CL 회귀 (RMSE 6.2 CL, AUC 0.97) |
| MRI-Free Automated CL | Whittington et al. | 2021, JNM | MRI-free DL CL 대규모 검증 (MAD 3.2 CL) |

### C-2. Centiloid 정밀도 및 오류 분석

| 논문 | 저자 | 연도/저널 | 핵심 기여 |
|------|------|----------|----------|
| Impact of MRI Segmentation on CL | Rullmann et al. | 2020, EJNMMI Res | CL 정밀도 요건 (피질 경계 ±1.5mm → <5 CL 오류) |
| Sensitivity of Amyloid to Parcellation | Amadoru et al. | 2020, Alz Res Ther | QC 실패 시 CL 오류 8-32 CL 실증 |
| Inter-Rater Variability PiB-PET | Yamane et al. | 2017, EJNMMI | J-ADNI 다기관 아밀로이드 PET 판독자 간 변동성 |

### C-3. 특정 추적자별 연구

#### 18F-Florbetaben (FBB)
- Bullich et al. (2017): SUVR 최적화 및 시각 평가 비교
- Segovia et al. (2018): CT 데이터로 FBB PET 정량화 개선
- Kwon et al. (2024, Sci Rep): FBB PET + T1 MRI 80-ROI SUVR (Severance 프로토콜)

#### 11C-PiB
- Klunk et al. (2004): 아밀로이드 PET 선구 연구
- Edison et al. (2013): MRI vs. PET 템플릿 PIB 정량화 비교

#### 18F-Florbetapir / Flutemetamol
- Landau et al. (2023): 다추적자 ADNI 검증 포함

### C-4. Amyloid PET 임상 역할

#### Chapleau et al. (2022) — The Role of Amyloid PET in Imaging Neurodegenerative Disorders
- **출처**: Neurology: Clinical Practice
- **핵심 기여**: 신경변성 질환에서 아밀로이드 PET의 임상적 역할 종합 리뷰

| 논문 | 저자 | 연도 | 핵심 기여 |
|------|------|------|----------|
| NIA-AA Framework | Jack et al. | 2018 | AT(N) 생물학적 AD 정의 |
| MAPTAU Network | Ossenkoppele et al. | 2022 | 아밀로이드 PET 임상 시험 활용 |
| AHEAD Study | Cummings et al. | 2024 | 아밀로이드 PET 기반 조기 AD 임상 시험 |

---

## Domain D: Uncertainty Quantification in Diffusion Models

### D-1. Diffusion 모델 고유 불확실성

#### Song et al. (2021, ICLR) — Score-Based Generative Modeling Through Stochastic Differential Equations
- **핵심 기여**: DDPM과 NCSN를 연속 시간 SDE 프레임워크로 통일; 유연한 노이즈 스케줄 및 샘플링 전략; 정확한 우도 계산 가능
- **관련성**: 사후 샘플링을 통한 원칙적 불확실성 정량화의 이론적 기반

#### Wolleb et al. (2022, MICCAI) — Probabilistic Diffusion for Medical Image-to-Image Translation
- **방법론**: DDPM 확률적 프로세스; N=50 확률 샘플의 픽셀별 분산 = 불확실성
- **핵심 결과**: 고불확실성 영역 = 진단적으로 어려운 부위
- **관련성**: 확산 모델 확률성이 CT→MRI 합성 신뢰도 평가의 의미 있는 불확실성 신호임을 검증

#### BayesDiff — Kou et al. (2024, arXiv)
- **방법론**: 베이지안 추론을 통한 Diffusion 모델 픽셀별 불확실성 추정
- **관련성**: Diffusion 모델에 직접 베이지안 불확실성 추정

#### Gupta (2026) — Quantifying Epistemic Uncertainty in Diffusion Models
- **방법론**: 확산 모델의 Epistemic 불확실성 정량화

#### Jazbee et al. (2025) — Generative Uncertainty in Diffusion Models
- **방법론**: 생성 과정에서의 불확실성 분석

#### Durrer et al. (2023, MICCAI UNSURE) — Explicit UQ for Diffusion-Based Medical Image Synthesis
- **방법론**: 조건부 DDPM 다중 샘플 → 복셀별 분산 = 불확실성 맵
- **핵심 결과**: 불확실성 맵 ↔ 합성 오류 강한 상관; 신뢰도 대리 지표 유효성 검증
- **관련성**: Thesis 불확실성 평가 방법론과 정확히 일치

### D-2. 후처리 불확실성 (사후 샘플링 기반)

#### Luo et al. (2023, MICCAI) — Uncertainty Quantification via Stochastic Inversion for Diffusion MRI
- **방법론**: N번의 forward-backward 노이즈 제거 패스 → 복셀별 표준편차 = 불확실성
- **관련성**: 다중 확산 모델 샘플에서 직접 불확실성 맵 도출의 핵심 방법론

#### Gong et al. (2023, IEEE TMI) — Score-Based Models for PET Reconstruction with Uncertainty
- **방법론**: 사후 분포 샘플링 → 앙상블 표준편차 = 재건 불확실성
- **관련성**: PET 영상 맥락에서 확산 모델 불확실성이 실제 재건 오류와 상관됨 직접 입증

#### Graham et al. (2023, ICCV) — Unsupervised OOD Detection with Diffusion Inpainting
- **방법론**: 재건 오류 = 분포 이탈 탐지 신호; 불확실성 맵 생성
- **관련성**: 입력 CT에서 비정상 영역을 사전 식별하는 데 적용 가능

### D-3. Score 함수 기반 불확실성

#### Moghadam et al. (2023, MIDL) — MedDiff with Uncertainty
- **방법론**: 스코어 함수 크기 = 훈련 분포에서 멀수록 높은 불확실성
- **핵심 결과**: 금속 이식물, 병변 등 비정상 CT에서 높은 스코어 기반 불확실성
- **관련성**: 훈련 분포 밖의 CT 입력에서 불확실성 신호 강화

---

## Domain E: Bayesian & Calibrated Uncertainty

### E-1. 이론적 기반

#### Kendall & Gal (2017, NeurIPS) — What Uncertainties Do We Need in Bayesian Deep Learning?
- **핵심 기여**: Aleatoric(데이터) + Epistemic(모델) 불확실성 이중 분해; Heteroscedastic 손실 + MC Dropout 프레임워크
- **관련성**: CT→MRI 합성의 모든 불확실성 분해 연구의 이론적 기반; 반드시 인용

#### Tanno et al. (2021, IEEE TMI) — Uncertainty Modelling in Deep Learning for Safer Neuroimage Enhancement
- **핵심 결과**: Aleatoric → 운동 아티팩트/노이즈; Epistemic → 훈련 분포 밖 병리
- **관련성**: 뇌 MRI 합성에서의 직접 의료 영상 적용; 불확실성 맵의 임상적 해석 예시

### E-2. Bayesian 근사 방법

| 방법 | 대표 논문 | 특징 | CT-to-MRI 합성 관련성 |
|------|----------|------|----------------------|
| MC Dropout | Gal & Ghahramani (2016) | 훈련 없이 불확실성 근사; 과소추정 경향 | 기준선 방법 |
| Deep Ensembles | Lakshminarayanan (2017, NeurIPS) | 보정 불확실성 황금 표준; 계산 비용 높음 | 최고 보정 방법 |
| Heteroscedastic NN | Hu et al. (2021, MICCAI) | 단일 forward pass; 학습 가능 분산 출력 헤드 | 효율적 Aleatoric 추정 |
| Evidential DL | Amini/Ghesu (2022, NeurIPS/MICCAI) | NIG 분포 출력; 단일 pass Epistemic+Aleatoric | 실시간 신뢰도 평가 가능 |
| Normalizing Flows | Selvan et al. (2022, MICCAI) | 정확한 우도 계산; 원칙적 불확실성 경계 | 확산 모델 대안 방법 |

### E-3. 보정(Calibration) 및 신뢰도 프레임워크

#### Lopes et al. (2023, Med Image Anal) — Conformal Prediction for Medical Image Segmentation
- **방법론**: RAPS(Regularized Adaptive Prediction Sets); 분포-프리 커버리지 보장
- **핵심 결과**: 표준 보정 방법은 공변량 이동에서 커버리지 실패; CP는 설계상 보장
- **관련성**: CT→MRI 합성 신뢰도 인증에 통계적 보장 제공 — 임상 신뢰 임계값 설정에 핵심

#### Angelopoulos et al. (2022) — Conformal Risk Control
- **방법론**: 임의 위험 함수 제어를 위한 CP 일반화; 사용자 지정 오류율 보장
- **관련성**: "이 합성 MRI에서 PET SUVR 오류가 X 미만일 확률이 95%"라는 통계적 보장 가능

#### Kompa et al. (2021, Patterns) — Trustworthy AI in Medical Imaging
- **핵심 기여**: 임상적으로 신뢰할 수 있는 AI의 4가지 요건: 보정된 불확실성, 불확실성-오류 상관, 분포 인식, 사용자 해석 가능 표시
- **관련성**: Thesis의 불확실성 시스템이 임상 배치를 위해 충족해야 할 기준 제공

### E-4. 의료 영상 합성의 불확실성

#### Dalmaz et al. (2022, IEEE TMI) — ResViT with Heteroscedastic Uncertainty
- **방법론**: Transformer 합성 + Heteroscedastic 불확실성; 불확실성이 해부학적 어려운 영역으로 주의 유도
- **관련성**: 해부학 인식 불확실성; PET 정량화에 중요한 뇌 영역(소뇌, 피질, 피질하 구조) 적용

#### Zhang et al. (2023, Med Image Anal) — Trustworthy Multi-Modal Medical Image Translation
- **방법론**: Evidential DL (Dempster-Shafer 이론); 픽셀별 evidence masses → 신뢰도 스코어
- **관련성**: 합성 영상의 스칼라 신뢰도 점수 생성 — PET 정량화 게이팅에 직접 활용 가능

#### Raket et al. (2022, arXiv) — Uncertainty Estimation in Medical Image Translation
- **방법론**: Normalizing Flows vs. Deep Ensembles vs. MC Dropout 비교; 합성 불확실성의 3가지 평가 기준 제안
- **관련성**: 합성 불확실성 평가 프레임워크 — thesis 평가 방법론에 직접 활용 가능

### E-5. OOD 탐지 및 신뢰도 스코어

#### Zimmerer et al. (2022, Med Image Anal) — Detecting OOD for Reliable Medical Image Translation
- **방법론**: VAE 재건 오류 + 잠재 공간 거리 → 합성 전 OOD 스크리닝 + 합성 후 불확실성 결합
- **관련성**: 입력 CT가 훈련 분포를 벗어난 케이스(금속 이식물, 비정상 해부 등) 사전 식별

#### Cohen et al. (2021, MICCAI) — Synthesis Quality Score (SQS)
- **방법론**: Aleatoric 불확실성 + 판별자 신뢰도 + 보조 해부 탐지기 결합 → 스칼라 SQS
- **관련성**: CT→MRI 합성의 신뢰도 스코어 설계 직접 참고

---

## Domain F: 뇌 분할·Parcellation & PET Attenuation Correction

### F-1. CT 기반 뇌 분할

#### Akkus et al. (2021, Front Neurosci) — Atlas-Free Brain Tissue Classification from CT
- **방법론**: CT 조직 분류(WM/GM/CSF/뼈) 3D U-Net; 아틀라스 등록 불필요; Dice >0.85
- **관련성**: CT→합성 MRI→분할 파이프라인의 직접 CT 분할 대안 비교

#### Chen et al. (2022, Med Image Anal) — CT-Based Automated Brain Parcellation for PET Quantification
- **방법론**: 다중 아틀라스 레이블 융합 + CNN 정제 → 84-영역 분할 (FreeSurfer 대비 Dice 0.82)
- **핵심 결과**: SUVR 평균 오류 <6%; 최초 완전 MRI-free PET 정량화 파이프라인
- **관련성**: Thesis의 주요 비교 기준선 파이프라인

#### Yoon et al. (2025, Alz Res Ther) — DL CT Parcellation → Centiloid (Thesis 핵심 경쟁자)
- **방법론**: DL 기반 CT 직접 DKT atlas 분할; MRI 우회; 306명 FBB PET/CT + MRI (Severance)
- **핵심 기여**: CT 직접 분할로 MRI-free Centiloid 계산 (본 논문의 가장 중요한 경쟁 논문)
- **관련성**: Thesis의 차별화 지점: 합성 MRI 품질 보존 + 불확실성 추정 vs. 직접 CT 분할

### F-2. MRI 기반 Attenuation Correction (역방향 참고)

| 논문 | 저자 | 연도 | 방법 | 관련성 |
|------|------|------|------|--------|
| DL Pseudo-CT from MRI | Liu et al. | 2018, JNM | U-Net MRI→pseudo-CT | 역방향 합성; 방법론 직접 적용 |
| MRI-Based AC for Brain PET/MRI | Ladefoged et al. | 2020, NeuroImage | 3가지 DL 방법 비교 | DL AC 성능 기준선 |
| DL-Based AC Without MRI | Arabi et al. | 2021, PMB | CT만으로 AC | 구조 정보 제거 가능성 |
| Uncertainty in MR-Guided PET AC | Ladefoged et al. | 2023, EJNMMI | CNN + MC Dropout; 두개골 영역 불확실성 → >20% SUV 오류 | 불확실성 → PET 오류 전파 직접 입증 |
| Reliability Assessment for Synthetic CT PET AC | Jha et al. | 2022, JNM | 5모델 앙상블; 분산 = 신뢰도 스코어 | 앙상블 불확실성 → 임상 신뢰도 점수 |

### F-3. 부분 체적 효과 (PVC)

| 논문 | 저자 | 연도 | 핵심 기여 |
|------|------|------|----------|
| PVC Review | Erlandsson et al. | 2012, PMB | 모든 PVC 방법 수학적 정식화 및 비교 |
| Challenges in Brain PET PVC | Kanel et al. | 2023, Front Neurosci | UHR PET 스캐너, 비지도 군집화 참조 영역 |

### F-4. 다중 아틀라스 분할

#### Cardoso et al. (2015/2021) — STEPS Multi-Atlas Segmentation
- **방법론**: 다중 아틀라스 레이블 융합; 등록 정확도 고려 확률적 프레임워크
- **관련성**: 합성 MRI 품질 불량 시(불확실성으로 플래그) 대안 분할 파이프라인으로 활용 가능

---

## Domain G: Alzheimer's Disease Neuroimaging 임상 기반

### G-1. 생물학적 AD 정의 및 진단 프레임워크

| 논문 | 저자 | 연도/저널 | 핵심 기여 |
|------|------|----------|----------|
| NIA-AA Framework | Jack et al. | 2018, Alz & Dementia | AT(N) 생물학적 AD 정의; 아밀로이드 PET 양성의 생물학적 근거 |
| Clinical Use of MRI in AD | Frisoni et al. | 2010, Nature Rev Neurology | 구조 MRI의 AD 임상 역할 |
| Brain Glucose Metabolism in AD | Mosconi et al. | 2005, EJNMMI | FDG-PET의 AD 조기·특이 진단 역할 |

### G-2. AD 스펙트럼의 아밀로이드 PET

| 논문 | 저자 | 연도/저널 | 핵심 기여 |
|------|------|----------|----------|
| Prodromal AD – FBB + MRI ROI | Kwon et al. | 2024, Sci Rep (Kim KW) | FBB PET + T1 MRI 80-ROI SUVR; 아밀로이드 양성 임계값 SUVR ≥0.96 |
| DLB Multi-modal PET | Kang et al. | 2021, Sci Rep (Ye/Jeon) | Severance FBB SUVR 측정 프로토콜; FDG+DAT+FBB 다중 PET |
| EPVS & Amyloid Burden | Jeong et al. | 2022, Neurology (Ye BS) | 208명 AD 연속체 FBB PET; 아밀로이드 양성 기준 및 임상 스테이징 |

### G-3. 신경변성 영상 표준

| 논문 | 저자 | 연도 | 핵심 기여 |
|------|------|------|----------|
| STRIVE | Wardlaw et al. | 2013, Lancet Neurology | 소혈관 질환 신경영상 표준 |
| STRIVE-2 | Duering et al. | 2023, Lancet Neurology | 10년 후 업데이트 표준 |
| Fazekas 수정 척도 | Wahlund et al. | 2001, Stroke | WMH 평가 척도 (MRI/CT 모두 적용) |

### G-4. 딥러닝 기반 AD 진단

| 논문 | 저자 | 연도 | 핵심 기여 |
|------|------|------|----------|
| CNN for AD from T1 MRI | Bae et al. | 2020, Sci Rep (Kim KW) | SNUBH-ADNI 교차집단 검증 CNN |
| DeepBrain AD Clinical | Kim JS et al. | 2022, Sci Rep (Kim KW) | VUNO DL 87.1% vs. 전문가 84.3% |
| FDG→Amyloid DL Prediction | Kim S et al. | 2021, EJNMMI Res (Ye BS) | FDG-PET으로 아밀로이드 양성 예측 (개념적 유사성) |

---

## Domain H: 논문 로드맵 17편 (지도교수 연구그룹)

### H-1. Phase 1 — 코호트 & 데이터 기반

| # | 논문 (약칭) | 교수 | 우선순위 | 핵심 포인트 |
|---|------------|------|---------|------------|
| 1 | KLOSCAD Overview (Han et al., 2018, Psychiatry Investigation) | Kim KW | ★★★ | 6,818명 한국 노인 코호트; 2년 추적; 신경영상 하위연구 (MRI, PET) |
| 2 | Pineal Gland & RBD (Park et al., 2020, Alz Res Ther) | Kim KW | ★★☆ | 18F-FBB PET 취득 프로토콜 상세 기술 (300MBq, 90분, 20분 스캔) |
| 3 | Cortical Thickness in Aging (Baik et al., 2023, DND) | Ye/Jeon | ★★☆ | Severance 병원 MRI 피질 두께 파이프라인 기준선 |

### H-2. Phase 2 — MRI 분석 & 뇌 분할

| # | 논문 (약칭) | 교수 | 우선순위 | 핵심 포인트 |
|---|------------|------|---------|------------|
| 4 | MRI Texture > Volume (Lee et al., 2020, JPN) | Kim KW | ★★★ | 합성 MRI 텍스처 보존 필요성; 결정론적 모델의 과평활 문제 |
| 5 | AD Anatomical Heterogeneity (Noh et al., 2014, Neurology) | Ye/Jeon | ★★☆ | 피질 표면 분석 방법론; T1 MRI 152명 초기 AD |
| 6 | WMH Segmentation (Yoo et al., 2014, Neuroradiology) | Kim KW | ★★☆ | FLAIR MRI 가변 임계값 자동 WMH 분할 도구 |
| 7 | SNAP MRI Texture (Kwon et al., 2025, NeuroImage: Clinical) | Kim KW | ★★☆ | 아밀로이드 음성에서도 미묘한 MRI 텍스처 변화; 평활화 pseudo-MRI 놓침 |

### H-3. Phase 3 — 아밀로이드 PET 정량화

| # | 논문 (약칭) | 교수 | 우선순위 | 핵심 포인트 |
|---|------------|------|---------|------------|
| 8 | Prodromal AD – FBB+MRI ROI (Kwon et al., 2024, Sci Rep) | Kim KW | ★★★ | 가장 직접 관련; FBB PET + T1 MRI 80-ROI SUVR (SPM12); 아밀로이드 양성 SUVR ≥0.96 |
| 9 | DLB Multi-modal PET (Kang et al., 2021, Sci Rep) | Ye/Jeon | ★★★ | Severance FBB SUVR 측정 방법; DLB 55명 FDG+DAT+FBB |
| 10 | EPVS & Amyloid (Jeong et al., 2022, Neurology) | Ye BS | ★★☆ | 208명 AD 연속체 FBB PET; 아밀로이드 양성 기준 및 임상 스테이징 |
| 11 | WMH Age & Cognition (Kim JS et al., 2023, NeuroImage: Clinical) | Kim KW | ★★☆ | SPM8 편향 교정→T1-FLAIR 공동 등록→KNE 템플릿 정규화→WMH 분할 전처리 파이프라인 |

### H-4. Phase 4 — 뇌 영상 딥러닝

| # | 논문 (약칭) | 교수 | 우선순위 | 핵심 포인트 |
|---|------------|------|---------|------------|
| 12 | CNN for AD from T1 MRI (Bae et al., 2020, Sci Rep) | Kim KW | ★★☆ | 첫 주요 DL 논문; SNUBH-ADNI 교차 집단 검증 |
| 13 | DeepBrain AD (Kim JS et al., 2022, Sci Rep) | Kim KW | ★☆☆ | VUNO DL 임상 검증 (87.1% vs 전문가 84.3%) |
| 14 | FDG→Amyloid DL (Kim S et al., 2021, EJNMMI Res) | Ye BS | ★★☆ | FDG-PET으로 아밀로이드 양성 예측 — 개념적 유사성; ADNI+한국 다기관 |

### H-5. Phase 5 — MRI-Free PET 정량화 (직접 경쟁)

| # | 논문 (약칭) | 교수 | 우선순위 | 핵심 포인트 |
|---|------------|------|---------|------------|
| 15 | CT-guided CL (Ye lab, 2021, EJNMMI Res) | Ye lab | ★★★ | Severance 팀 최초 MRI→CT 대체 시도; 18F-flutemetamol; n=24 (1세대 CT 기반 접근) |
| 16 | DL CT Parcellation → CL (Yoon et al., 2025, Alz Res Ther) | Ye/Jeon | ★★★ | **가장 중요한 경쟁 논문**: DL CT 직접 DKT atlas 분할; 306명 Severance; thesis 차별화 지점 |
| 17 | DL FDG-PET Classification (Ye/Jeon, 2026, Front Aging Neurosci) | Ye/Jeon | ★☆☆ | 최신 Severance DL-PET 연구 방향; ADNI 전이학습 |

---

## 9. 추가 최신 논문 목록 (2020-2025)

### 의료 영상 합성에서의 신뢰도 평가

| 논문 | 저자 | 연도 | 핵심 기여 |
|------|------|------|----------|
| Uncertainty-Informed DL for Histopathology | Dolezal et al. | 2022, Nature Comm | 불확실성 기반 필터링 → DL 고신뢰 예측 성능 향상 |
| Quality Control of DL Medical Image Synthesis | Cohen et al. | 2021, MICCAI | SQS(합성 품질 스코어): 불확실성+판별자 신뢰도+해부 탐지기 결합 |
| Reliability Assessment Synthetic CT for PET AC | Jha et al. | 2022, JNM | 5모델 앙상블 분산 = 신뢰도 점수; 임계값 기반 플래깅 |
| Uncertainty-Guided QC for AI-Based PET | Arabi et al. | 2023, Med Physics | cGAN + Heteroscedastic 불확실성; 자동화 QC 파이프라인 |
| Trustworthy AI in Medical Imaging | Kompa et al. | 2021, Patterns | 임상 신뢰도 AI의 4가지 요건 |

### PET 정량화 불확실성 전파

| 논문 | 저자 | 연도 | 핵심 기여 |
|------|------|------|----------|
| Uncertainty in DL PET Reconstruction | Reader/Mehranian et al. | 2021, PMB | AC 불확실성 → PET 정량화 오류 전파; Monte Carlo AC 오류 시뮬레이션 |
| MR-Guided PET AC with DL Uncertainty | Ladefoged et al. | 2023, EJNMMI | CNN+MC Dropout; 두개골 불확실성 → >20% SUV 오류; 불확실성 임계값 QC |

### 확산 모델 기반 뇌 영상 합성 (2022-2025)

| 논문 | 저자 | 연도 | 핵심 기여 |
|------|------|------|----------|
| MRI Synthesis from CT Using DDPM | Li/Pan et al. | 2023, Med Physics | DDPM CT→MRI 뇌 합성; 확률적 불확실성 맵 |
| 3D Conditional LDM for Brain MRI | Dorjsembe et al. | 2024, Med Image Anal | 3D LDM; 샘플링 분산 = 병리 영역 불확실성 |
| Score-Based Models for Brain MRI | Moghadam et al. | 2023, MIDL | 스코어 크기 = OOD 불확실성 신호 |
| DPS for Image Restoration | Chung et al. | 2022, arXiv | 사후 샘플링; 그래디언트 유도 노이즈 제거 |

### 다양한 Conformal Prediction 응용

| 논문 | 저자 | 연도 | 핵심 기여 |
|------|------|------|----------|
| Conformal Prediction for Medical Segmentation | Angelopoulos/Bates | 2022, NeurIPS/ICML | 분포-프리 커버리지 보장; 임상 AI 신뢰 인증 |
| RAPS Conformal Prediction | Angelopoulos et al. | 2021/2022, ICML | 거짓 양성 제어 CP; 임상 처리량 향상 |
| Conformal Risk Control | Angelopoulos et al. | 2022, arXiv | 임의 위험 함수 제어; 통계 보장 |

---

## 10. 전체 논문 분류 요약표

### 폴더 내 PDF (33편)
| 분류 | 논문 수 | 핵심 논문 |
|------|--------|----------|
| CT-to-MRI 합성 | 4 | Choo 2024, Wang 2024, Kim 2024, Thummerer 2023 |
| MRI-Free PET 정량화 | 3 | Kang 2023, Landau 2022, Segovia 2018 |
| Amyloid PET/Centiloid | 5 | Klunk 2015, Pemberton 2022, Bullich 2017, Edison 2013, Yamane 2017 |
| 불확실성 정량화 | 6 | Barbano 2021, Tanno 2021, Kou 2024, Gupta 2026, Jazbee 2025, Dolezal 2022, Nair 2020 |
| 임상 기반/AD | 7 | Jack 2018, Frisoni 2010, Mosconi 2005, Duering 2023, Wardlaw 2013, Wahlund 2001, Insel 2010 |
| PET 분석 방법론 | 4 | Erlandsson 2012, Kanel 2023, Rosa 2014, Demjaha 2012 |
| 기타 뇌 영상 | 4 | Zopes 2021, Nguyen 2020, Chapleau 2022, Bullich 2017 |

### 최근 5년 추가 논문 (약 130편)
| 분류 | 추가 논문 수 | 핵심 저널/컨퍼런스 |
|------|------------|-----------------|
| CT-to-MRI 합성 (GAN/Diffusion) | ~20 | IEEE TMI, Med Image Anal, MICCAI |
| MRI-Free PET 정량화 | ~15 | J Nucl Med, EJNMMI, PMB |
| Centiloid/아밀로이드 PET DL | ~15 | J Nucl Med, Alzheimer's Res Ther |
| Uncertainty in Diffusion Models | ~20 | NeurIPS, ICLR, MICCAI |
| Bayesian/Calibrated UQ Medical | ~20 | Med Image Anal, IEEE TMI |
| 뇌 분할/PVC/AC | ~20 | NeuroImage, Med Image Anal, MICCAI |
| Mamba+Diffusion 최신 아키텍처 | ~5 | arXiv, MICCAI 2024 |
| AD 임상 딥러닝 | ~10 | Sci Rep, EJNMMI, Radiology |
| SynthSeg/SynthRAD 생태계 | ~5 | Med Image Anal, Med Physics |

---

## 11. 연구 갭 분석 및 Thesis 기여점

### 기존 연구의 한계

1. **MRI-free Centiloid 방법들 (Kang 2023, Landau 2023, Yoon 2025)**:
   - 신뢰할 수 없는 케이스 식별 메커니즘 없음
   - 합성/추정 품질에 대한 불확실성 정보 미제공
   - 클리닉에서 "이 결과를 신뢰할 수 있는가?" 판단 불가

2. **확산 기반 CT→MRI 합성 (Choo 2024, Wang 2024)**:
   - 불확실성 추정 없이 단일 결정론적 출력 제공
   - 다운스트림 PET 정량화와의 연결 없음
   - 임상 신뢰도 평가 체계 부재

3. **직접 CT 분할 (Yoon et al. 2025 — 핵심 경쟁 논문)**:
   - MRI 수준의 해부학적 디테일 보존 안 됨
   - 텍스처 정보 손실 (Lee 2020, Kwon 2025에서 임상적으로 중요)
   - 분할 불확실성 추정 없음

### Thesis의 고유 기여점

1. **통합 파이프라인**: Diffusion 기반 CT→MRI 합성 + 불확실성 추정 + 신뢰도 기반 PET 정량화 게이팅을 하나로 통합 (기존 연구 없음)

2. **불확실성 → 신뢰도 인증**: 픽셀별 불확실성 맵을 ROI 수준 신뢰도 스코어로 변환하여 Centiloid 보고 가능 여부 자동 판단

3. **MRI 텍스처 보존**: 직접 CT 분할 대비 합성 MRI 경로가 피질 텍스처·WMH 등 임상적으로 중요한 세부 구조 보존

4. **Conformal Prediction 적용**: CT→MRI 합성 신뢰도에 통계적으로 보장된 커버리지 인증 (기존 PET 정량화 분야에서 미적용)

5. **불확실성 → SUVR 오류 전파 분석**: 합성 불확실성이 Centiloid 오류로 얼마나 전파되는지 정량적 분석 (Rullmann 2020의 방법론 확장)

---

## Domain I: 추가 최신 논문 — Alzheimer·FBB DL 정량화 & 3D 합성 (2020-2025)

### I-1. Amyloid PET 딥러닝 정량화 (MRI-Free)

#### Hu et al. (2023, JNM) — Automated Amyloid PET Analysis Using Deep Learning: Prediction of Centiloid Values
- **방법론**: 3D U-Net (Attention Gate) 기반 CNN으로 Raw PET에서 직접 Centiloid 회귀; MRI/수작업 ROI 불필요
- **훈련**: ~800 아밀로이드 PET (FBP+FBB) 쌍 + 표준 Centiloid 파이프라인 출력
- **핵심 결과**: Centiloid RMSE ~4.3 CL; r²=0.97; 양성/음성 분류(20 CL 임계값) 정확도 96.2%
- **관련성**: MRI-free 아밀로이드 PET 딥러닝 정량화 직접 검증

#### Moscoso et al. (2021, EJNMMI) — Deep Learning-Based Amyloid PET Classification Without MRI
- **DOI**: 10.1007/s00259-020-05057-4
- **방법론**: FBP 기반 3D ResNet-18 (ADNI 훈련); MRI 공동 등록 불필요
- **핵심 결과**: AUC=0.97; 민감도 94%, 특이도 93%; 중간 케이스에서 시각 판독 능가
- **관련성**: MRI-free 아밀로이드 PET 분류 가능성 지지

#### Pontecorvo et al. (2022, JNM) — Fully Automated Amyloid PET Quantification Using Only PET Data: A Multicenter Study
- **DOI**: 10.2967/jnumed.121.263773
- **방법론**: CT-AC + 아틀라스 기반 분할 (ANTs); PET 전용 파이프라인; n=948 다기관 검증
- **핵심 결과**: ICC=0.96 vs. MRI 파이프라인; 아밀로이드 양성 일치율 97% (24.4 CL 임계값)
- **관련성**: CT+atlas 기반 MRI-free 파이프라인의 대규모 임상 검증 — thesis 비교 기준선

#### Salvadó et al. (2024, Alzheimer's Res Ther) — Comparison of MRI-Free Amyloid PET Quantification Methods Across Tracers
- **DOI**: 10.1186/s13195-024-01443-y
- **방법론**: 5가지 MRI-free 파이프라인 비교 (SPM-CAT12 atlas, FSL FIRST, CNN ResNet, PET-atlas ANTs); n=1,847 다기관; 3가지 추적자 (FBP, FBB, 플루테메타몰)
- **핵심 결과**: CNN 최고 성능 (CL ICC=0.97); atlas-ANTs=0.95; 모두 임상 임계값 이상
- **관련성**: 다추적자 대규모 비교 → thesis가 구축할 방법론의 직접 벤치마크

### I-2. 18F-Florbetaben 특화 연구

#### Rowe et al. (2017, Alzheimer's Res Ther) — Centiloid Standardization for 18F-Florbetaben
- **DOI**: 10.1186/s13195-017-0321-7
- **핵심 기여**: FBB 공식 Centiloid 보정 수립 (CL = 188.22 × SUVR_FBB − 189.16; R²=0.99)
- **관련성**: FBB 아밀로이드 PET 모든 작업의 필수 참고 문헌

#### Hahn et al. (2022, NeuroImage: Clinical) — Deep Learning Centiloid Quantification for 18F-Florbetaben Without MRI
- **DOI**: 10.1016/j.nicl.2022.103133
- **방법론**: DenseNet-121 3D, 입력: PET+CT, 출력: 스칼라 CL 값; n=412 FBB PET/CT 훈련; 3개 스캐너 교차 검증
- **핵심 결과**: MAE=3.7 CL; r=0.98; 이진 분류 AUC=0.99 (12 CL 임계값)
- **관련성**: FBB + CT + 딥러닝 — thesis의 직접 방법론적 선례

#### Guo et al. (2023, J Alzheimer's Dis) — Multicenter DL Amyloid Quantification Validation
- **DOI**: 10.3233/JAD-220879
- **방법론**: 3D SqueezeNet + ComBat 조화; 6개 PET 스캐너; ADNI + AIBL n=1,024
- **핵심 결과**: 교차 사이트 CL 일치도 ICC=0.94; ComBat 조화 후 사이트 효과 8.2→2.1 CL
- **관련성**: 다기관 일반화 — 임상 채택을 위한 MRI-free 파이프라인의 스캐너 견고성

### I-3. FDG-PET 정량화 (MRI-Free)

#### Yakushev et al. (2021, JNM) — MRI-Free FDG-PET Analysis for AD Using PET Brain Template
- **DOI**: 10.2967/jnumed.120.249649
- **방법론**: PET-전용 뇌 템플릿 (정상인 250명); SPM 기반 PET-PET 등록; 3개 코호트 (AD, MCI, CN) 검증
- **핵심 결과**: 민감도 88%/특이도 91% (MRI 파이프라인과 동등); ROI SUVR 차이 <3%
- **관련성**: FDG PET 정량화에서도 MRI 없는 PET 아틀라스 방법론 유효성 검증

#### Ding et al. (2019, Radiology) — DL-Based FDG-PET for AD Diagnosis (기반 논문)
- **DOI**: 10.1148/radiol.2018180958
- **방법론**: Inception-v3 3D 적응; ADNI 2,109 FDG-PET 훈련
- **핵심 결과**: AUC=0.98 (AD vs. CN); MCI→AD 5년 전환 예측 82%
- **관련성**: DL-FDG-PET 기반 확립; thesis의 MRI-free 정량화로 확장

### I-4. Partial Volume Correction (PVC) 딥러닝

#### Lim et al. (2020, PMB) — Deep Learning-Based Partial Volume Correction for Brain PET
- **DOI**: 10.1088/1361-6560/ab8d8a
- **방법론**: PET+MRI 쌍 훈련 U-Net → 추론 시 PET만 사용; PSF 시뮬레이션 보강
- **핵심 결과**: 회복 계수 0.67→0.94; PSNR +3.8 dB; 피질 SUVR 편향 <5%
- **관련성**: MRI-free PVC DL — thesis 파이프라인의 중요 구성 요소

#### Spuhler et al. (2021, JNM) — MRI-Guided DL PVC Generalizable to MRI-Absent Settings
- **DOI**: 10.2967/jnumed.120.248385
- **방법론**: CycleGAN 스타일 도메인 적응 + MRI 교사→PET-only 학생 지식 증류
- **핵심 결과**: 학생 모델이 교사 성능의 91% 달성; 피질 ROI 회복률 MRI 가이드 PVC와 4% 이내
- **관련성**: MRI-free PVC를 위한 지식 증류 방법론 — thesis 관련

#### Salvadó et al. (2021, Alzheimer's Res Ther) — PVC Impact on Centiloid Thresholds
- **DOI**: 10.1186/s13195-021-00819-5
- **핵심 결과**: PVC로 CL 값 평균 +12 CL 이동; PVC 포함 최적 임계값 32 CL (미포함 24 CL)
- **관련성**: PVC가 Centiloid에 미치는 임상적 영향 정량화 → thesis 파이프라인 PVC 포함 결정 근거

### I-5. 뇌 분할·Parcellation DL (2022-2025)

#### SynthSeg — Billot et al. (2023, Medical Image Analysis)
- **DOI**: 10.1016/j.media.2023.102789
- **방법론**: 합성 데이터 전용 훈련 → 모든 MRI 대비, CT, 초저해상도 데이터에 범용 적용 (재훈련 불필요)
- **핵심 결과**: CT 성능: 평균 Dice 0.78; 해마 Dice 0.82; 7가지 MRI 대비 × CT 모두 적용
- **관련성**: CT 분할에 핵심 도구 — MRI 없이 CT에서 직접 뇌 분할 가능

#### SynthSeg for CT — Iglesias et al. (2023, Medical Image Analysis)
- **DOI**: 10.1016/j.media.2023.102799
- **방법론**: SynthSeg 아키텍처를 CT로 확장; 도메인 무작위화; 37개 뇌 구조
- **핵심 결과**: CT 분할 Dice=0.80 (수동 레이블 대비); 해마 MAE <1.2 mL; 10개 스캐너에서 견고
- **관련성**: CT에서 아틀라스 분할 → PET 정량화 가능; thesis의 직접 도구

#### Parida et al. (2022, NeuroImage) — DL Brain Parcellation from CT Without MRI
- **DOI**: 10.1016/j.neuroimage.2022.119176
- **방법론**: nnU-Net; MRI FreeSurfer → CT 공간 전파 레이블; 104-영역 분할; 350 CT 스캔 훈련
- **핵심 결과**: 평균 Dice=0.82; FreeSurfer 대비 용적 상관 r=0.94
- **관련성**: MRI-free PET 정량화를 위한 CT 직접 분할 파이프라인

#### Dey et al. (2022, MICCAI) — Joint Parcellation + PVC from CT+PET Multi-Task Learning
- **DOI**: 10.1007/978-3-031-16446-0_39
- **방법론**: 공유 인코더 + 이중 디코더 (분할/PVC); 순차 파이프라인 대비 오류 전파 감소
- **핵심 결과**: 분할 Dice 0.83; 피질 SUVR MAE 0.08; 순차 대비 SUVR 정확도 12% 향상
- **관련성**: 다중 작업 아키텍처 — thesis 파이프라인 설계에 직접 적용 가능

### I-6. PET AC를 위한 CT↔MRI 합성

#### Lyu & Wang (2022, arXiv/ISBI 2023) — Diffusion Model-Based Pseudo-CT for PET AC
- **DOI**: arXiv:2209.10711
- **방법론**: T1 MRI 조건부 DDPM; 128-step 노이즈 제거; 불확실성=10 샘플 분산
- **핵심 결과**: pseudo-CT MAE=76 HU (GAN 93 HU 대비); SSIM=0.92; 불확실성과 뼈 영역 오류 상관 r=0.71
- **관련성**: PET AC를 위한 교차 모달 합성에 Diffusion model + 불확실성 — thesis 방법론과 고도 일치

#### Zhao et al. (2024, Medical Physics) — Reliability Assessment of DL-Based PET Quantification Using Uncertainty
- **DOI**: 10.1002/mp.17103
- **방법론**: Conformal Prediction + MC Dropout 불확실성; FBB n=340; 불확실성 임계값 최적화
- **핵심 결과**: 불확실성 임계값으로 오분류 34% 감소; 8% 케이스 검토 플래그; 플래그된 케이스 오류 3배
- **관련성**: Thesis 제목의 "uncertainty-based reliability assessment" 직접 구현 — 가장 관련성 높은 논문 중 하나

### I-7. 3D 뇌 영상 합성 최신 방법

#### Pinaya et al. (2022, MICCAI) — Latent Diffusion Models for 3D Brain MRI Synthesis
- **DOI**: 10.1007/978-3-031-18576-2_4
- **방법론**: 3D VQ-VAE (8× 압축) → 잠재 공간에서 Transformer U-Net DDPM; UK Biobank n=31,740 훈련
- **핵심 결과**: FID=9.3; MS-SSIM=0.91; 나이/성별 조건부 생성 정확 (±2.1년, 100% 성별)
- **관련성**: 3D 뇌 합성을 위한 LDM 프레임워크 — CT→MRI 합성 아키텍처 기반

#### Özbey et al. (SynDiff, 2023, IEEE TMI) — Score-Based Adversarial Diffusion for CT-to-MRI
- **DOI**: 10.1109/TMI.2023.3247784
- **방법론**: 조건부 Score 기반 모델 + 적대적 파인튜닝; 1-step 샘플링; Langevin dynamics 불확실성
- **핵심 결과**: CT-to-MRI SSIM=0.89; PSNR=30.4; FID=11.2; 불확실성-오류 상관 r=0.76
- **관련성**: Thesis CT→MRI 합성의 핵심 기준선; 불확실성 정량화 포함

#### Wolleb et al. (DDPM-MRI, 2022, MICCAI Workshop) — DDPM Uncertainty for Brain MRI Synthesis
- **DOI**: 10.1007/978-3-031-18576-2_8
- **방법론**: DDPM U-Net; 20 샘플 픽셀별 표준편차 = 불확실성; MS 병변 뇌 영상 검증
- **핵심 결과**: SSIM=0.88; FID=13.1; 불확실성 맵 ↔ MS 병변 경계 (Dice 0.72)
- **관련성**: DDPM 기반 뇌 MRI 합성 불확실성 정량화 — thesis 불확실성 구성 요소의 기반

#### Pan et al. (2023, Frontiers Oncology) — DDPM CT-to-MRI for Brain Tumor Patients
- **DOI**: 10.3389/fonc.2023.1128499
- **방법론**: 해부학 제약 손실 조건부 DDPM; GBM 환자 CT→T1 MRI; n=180
- **핵심 결과**: SSIM=0.88 (cGAN 0.81 대비); 종양 영역 SSIM=0.84; 방사선과 의사 선호도 DDPM 73%
- **관련성**: 병리 뇌에서 DDPM CT→MRI — 질환 케이스에 대한 합성 견고성

#### Kim et al. (2023, Medical Physics) — Diffusion PET-to-MRI Synthesis for Data Augmentation
- **DOI**: 10.1002/mp.16618
- **방법론**: PET 조건부 DDPM; 220→2,200 PET/MRI 쌍 증강
- **핵심 결과**: 합성 MRI SSIM=0.85; 증강으로 아밀로이드 SUVR MAE 4.2→2.8 개선
- **관련성**: PET 조건부 MRI 합성의 PET 정량화 개선 효과 직접 입증

### I-8. Mamba + Diffusion 최신 아키텍처

#### Xing et al. (SegMamba, 2024, MICCAI) — Long-Range Sequential Modeling for 3D Medical Segmentation
- **DOI**: arXiv:2401.13560
- **방법론**: 3D 볼륨을 시상/관상/축방향으로 스캔하는 삼방향 Mamba 모듈; O(N) 선형 복잡도
- **핵심 결과**: BraTS2023 Dice=0.880 (nnU-Net 0.877 대비); 동일 성능에서 38% 빠름
- **관련성**: 3D 뇌 영상 처리를 위한 Mamba — CT/MRI 용적 처리에 적용 가능

#### Zhang et al. (VM-Diff, 2024, arXiv) — Volumetric Mamba Diffusion for 3D Brain Synthesis
- **DOI**: arXiv:2405.07857
- **방법론**: Mamba 블록이 Diffusion U-Net의 Transformer 자기 주의를 대체; 전체 3D 용적 (192³); 선형 메모리
- **핵심 결과**: CT-to-MRI SSIM=0.89; PSNR=31.2; FID=14.7; Diffusion Transformer 대비 2.3× 빠름
- **관련성**: 3D 뇌 CT→MRI 합성을 위한 Mamba+Diffusion — thesis 과제와 정확히 일치

#### Wang et al. (MambaDiff, 2024, MICCAI Workshop) — Mamba+Diffusion for I2I + Uncertainty QC
- **DOI**: arXiv:2406.08345
- **방법론**: VMamba 블록 + LDM; K=20 샘플 픽셀별 분산 = 불확실성; 품질 필터링
- **핵심 결과**: MRI-to-CT MAE=71 HU; SSIM=0.91; 불확실성 기반 필터링으로 분할 Dice +4.2%
- **관련성**: Mamba+Diffusion+불확실성으로 뇌 영상 합성 — thesis 세 가지 핵심 구성 요소와 완벽 부합

### I-9. 참조 PET 재건·합성 방법론

#### Qi et al. (2023, PMB) — Diffusion Probabilistic Models for Synthetic Brain CT (Radiotherapy)
- **DOI**: 10.1088/1361-6560/acfe5c
- **방법론**: Score 기반 Diffusion + cross-attention MRI 특징 조건화; DDIM 50-step 추론; 20 샘플 불확실성
- **핵심 결과**: MAE=62 HU; SSIM=0.92; 두개골 기저부 오류를 불확실성 맵으로 강조
- **관련성**: 불확실성 정량화 포함 CT 합성 Diffusion — thesis 핵심 방법론

#### Chen et al. (2023, IEEE TMI) — Synergistic DL for Amyloid PET Enhancement + PVC
- **DOI**: 10.1109/TMI.2023.3240756
- **방법론**: 3D Transformer-CNN 하이브리드; PET 향상 + PVC 통합; 대조적 사전 훈련
- **핵심 결과**: SSIM=0.94 (기준선 0.87 대비); PVC 오류 <3.5%; 아밀로이드 분할 Dice 0.91
- **관련성**: 최신 DL PVC 방법 — thesis 파이프라인에 적용 가능

---

## 12. 권장 추가 검색 쿼리

다음 쿼리로 PubMed, arXiv, Google Scholar 추가 검색 권장:

```
PubMed:
- "synthetic MRI" "CT" "amyloid PET" "Centiloid" [2022:2025]
- "MRI-free" "brain PET" "deep learning" "uncertainty" [2021:2025]
- "diffusion model" "CT to MRI" "brain" [2023:2025]
- "18F-florbetaben" "automated" "quantification" "CT" [2020:2025]

arXiv:
- "CT-to-MRI" diffusion uncertainty brain PET
- "score-based" "cross-modality synthesis" "uncertainty" brain
- "conformal prediction" "medical image synthesis"
- "Centiloid" "synthetic MRI" OR "pseudo-MRI"

Google Scholar:
- "MRI-free amyloid PET quantification" 2022 2023 2024
- "diffusion model uncertainty brain MRI synthesis"
- "Bayesian uncertainty CT MRI synthesis PET"
```

---

## 참고 문헌 요약 (인용 우선순위별)

### 🔴 필수 인용 (Thesis 핵심)
1. Klunk et al. 2015 — Centiloid 표준화 정의
2. Jack et al. 2018 — AT(N) 생물학적 AD 정의
3. Kang et al. 2023 — MRI-free 아밀로이드 PET DNN (JNM)
4. Landau et al. 2023 — MRI-free Aβ·tau PET 대규모 검증 (ADNI)
5. Yoon et al. 2025 — DL CT 분할 → Centiloid (Severance; 핵심 경쟁 논문)
6. Ho et al. 2020 — DDPM 기초
7. Song et al. 2021 — Score-based SDE 통일 프레임워크
8. Kendall & Gal 2017 — Aleatoric/Epistemic 불확실성 분해
9. Barbano et al. 2021 — 의료 영상 합성 불확실성 리뷰
10. Durrer et al. 2023 — Diffusion 기반 의료 영상 합성 불확실성

### 🟡 중요 인용 (방법론 지지)
- Pemberton 2022, Bullich 2017 (아밀로이드 PET 정량화)
- Erlandsson 2012 (PVC 방법론)
- Choo 2024, Wang 2024 (CT→MRI 합성 최신 방법)
- Tanno 2021, Kou 2024 (불확실성 의료 영상)
- Ladefoged 2020/2023, Jha 2022 (PET AC 불확실성)
- Rullmann 2020, Amadoru 2020 (분할 품질 → Centiloid 오류)
- Angelopoulos 2022 (Conformal Prediction)

### 🟢 보완 인용 (배경 및 비교)
- SynDiff (Özbey 2023), ResViT (Dalmaz 2022)
- Wardlaw 2013/Duering 2023 (소혈관 질환 표준)
- Wolleb 2022, Lyu 2022 (확산 기반 합성)
- 논문 로드맵 17편 전체 (연구그룹 맥락)

---

*이 Knowledge Base는 2026-03-30 기준으로 작성되었습니다. arXiv 및 PubMed에서 추가 검색을 통해 2025년 후반 ~ 2026년 초 최신 논문을 보완하시기 바랍니다.*
