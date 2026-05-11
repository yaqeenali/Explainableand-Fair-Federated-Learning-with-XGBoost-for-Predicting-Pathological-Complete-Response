# Explainable and Fair Federated Learning with XGBoost
## Predicting Pathological Complete Response in Breast Cancer using Multi-Center DCE-MRI Data

[![Paper](https://img.shields.io/badge/SPIE_Medical_Imaging_2026-Published-blue)](https://doi.org/10.1117/12.3085564)
[![Python](https://img.shields.io/badge/Python-3.8%2B-green)](https://www.python.org/)
[![NVIDIA FLARE](https://img.shields.io/badge/NVIDIA_FLARE-2.4%2B-76B900)](https://nvflare.readthedocs.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0%2B-orange)](https://xgboost.readthedocs.io/)
[![License](https://img.shields.io/badge/License-MIT-yellow)](LICENSE)

> **Explainable and Fair Federated Learning with XGBoost for Predicting Pathological Complete Response in Breast Cancer using Multi-Center DCE-MRI Data**
> Yaqeen Ali, Julia Müller, Andreas Weinmann, Johannes Gregori
> *SPIE Medical Imaging 2026: Computer-Aided Diagnosis — Proc. SPIE Vol. 13926, 139260Q*
> DOI: [10.1117/12.3085564](https://doi.org/10.1117/12.3085564)

---

## Overview

This repository implements a **privacy-preserving, explainable, and fair federated learning (FL)** framework for predicting pathological complete response (pCR) to neoadjuvant systemic therapy in breast cancer. The framework combines:

- **Federated Learning** via NVIDIA FLARE (4-client simulation across DUKE, ISPY1, ISPY2, NACT)
- **XGBoost** with bagging aggregation — no raw data leaves each institution
- **Radiomic features** extracted with PyRadiomics from DCE-MRI tumor volumes
- **SHAP** explainability to identify clinically meaningful predictors
- **Fairness analysis** across age, menopausal status, scanner vendor, and clinical site

---

## Key Results

### Performance Comparison (Table 1 from paper)

| Test Site | Local (Bal.Acc / AUC) | **FL Model** (Bal.Acc / AUC) | Centralized (Bal.Acc / AUC) |
|---|---|---|---|
| Duke | 0.53 / 0.53 | **0.68 / 0.68** | 0.68 / 0.68 |
| ISPY1 | 0.64 / 0.64 | **0.69 / 0.69** | 0.68 / 0.68 |
| ISPY2 | 0.53 / 0.53 | **0.63 / 0.63** | 0.61 / 0.61 |
| NACT | 0.50 / 0.50 | 0.48 / 0.48 | 0.41 / 0.41 |
| **All Sites** | — | **0.65 / 0.64** | **0.65 / 0.70** |

### Fairness Comparison

| Model | Fairness Score |
|---|---|
| Centralized | 0.53 |
| **Federated (ours)** | **0.62** |

> FL acts as a natural regularizer — higher fairness score with competitive accuracy.

### Top SHAP Predictors (Figure 2 from paper)

| Rank | Feature | Direction |
|---|---|---|
| 1 | Hormone Receptor (HR) | High HR → non-pCR |
| 2 | HER2 Status | High HER2 → pCR |
| 3 | Maximum 3D Tumor Diameter | — |
| 4 | Tumor Sphericity | — |
| 5 | Tumor Texture Complexity | — |

---

## Repository Structure

```
.
├── federated/
│   ├── server/
│   │   └── fl_server.py              # NVIDIA FLARE server — ScatterAndGather controller
│   └── clients/
│       ├── fl_client.py              # FLARE client executor — local XGBoost training
│       └── site_configs/             # Per-site data path configs (Duke, ISPY1, ISPY2, NACT)
│           ├── duke.yaml
│           ├── ispy1.yaml
│           ├── ispy2.yaml
│           └── nact.yaml
│
├── feature_engineering/
│   ├── radiomic_extraction.py        # PyRadiomics feature extraction from 3D tumor ROIs
│   ├── feature_selection.py          # Spearman + mutual information feature selection
│   └── feature_fusion.py            # Fuse top-6 imaging + 5 clinical features
│
├── models/
│   ├── xgboost_centralized.py        # Centralized XGBoost baseline
│   ├── xgboost_local.py             # Per-site local XGBoost models
│   └── xgboost_federated.py         # Federated XGBoost with XGBBaggingAggregator
│
├── evaluation/
│   ├── metrics.py                    # AUC, balanced accuracy, sensitivity, specificity
│   ├── fairness.py                   # Fairness score (Equalized Odds-based, Eq. 1-3)
│   └── explainability.py            # SHAP summary plots and feature importance
│
├── configs/
│   └── config.yaml                   # All hyperparameters (eta=0.1, max_depth=3, etc.)
│
├── notebooks/
│   ├── 01_data_exploration.ipynb    # MAMA-MIA dataset statistics and class distribution
│   ├── 02_feature_engineering.ipynb  # Radiomic extraction walkthrough
│   ├── 03_federated_training.ipynb   # FL training curves across rounds
│   ├── 04_results_comparison.ipynb  # Table 1 reproduction
│   └── 05_shap_fairness.ipynb       # SHAP plots + fairness subgroup analysis
│
├── figures/
│   ├── workflow.png                  # Figure 1 from paper
│   └── shap_summary.png             # Figure 2 from paper
│
├── paper/
│   └── Ali_et_al_SPIE2026_FL_XGBoost_pCR.pdf
│
├── requirements.txt
└── README.md
```

---

## Pipeline

```
MAMA-MIA Dataset (1,506 DCE-MRI cases, 4 collections)
        │
        ▼
(A) Tumor Segmentation
    Training:  Expert-delineated masks (MAMA-MIA ground truth)
    Inference: Pre-trained nnU-Net (mean Dice = 0.76)
    Preprocessing: Z-score normalisation + isotropic resampling (1 mm³)
        │
        ▼
(B) Radiomic Feature Extraction (PyRadiomics)
    Shape + First-Order + Texture (GLCM, GLRLM, GLSZM, ...)
        │
        ▼
(C) Feature Selection + Fusion
    Top-6 imaging features  +  5 clinical variables
    (ER, PR, HER2, HR, tumour subtype)
    → Multimodal feature vector per patient
        │
        ▼
(D) Federated Learning — NVIDIA FLARE
    ┌──────────────┐    ScatterAndGather    ┌──────────────┐
    │ FLARE Server │ ◄──────────────────── │ Client: DUKE │
    │ XGBBagging   │ ──────────────────► │ Client: ISPY1│
    │ Aggregator   │ ◄──────────────────── │ Client: ISPY2│
    └──────────────┘                       │ Client: NACT │
                                           └──────────────┘
        │
        ▼
(E) Evaluation
    Performance:      AUC, Balanced Accuracy
    Explainability:   SHAP summary plot
    Fairness:         Equalized Odds across age, site, scanner, menopausal status
```

---

## Dataset

**MAMA-MIA** — Large-scale multicenter breast cancer DCE-MRI benchmark

| Collection | Role in FL | Size |
|---|---|---|
| Duke-Breast-Cancer-MRI | Client 1 | 922 cases |
| I-SPY 1 | Client 2 | 98 cases |
| I-SPY 2 | Client 3 | 386 cases |
| NACT-Breast-MRI | Client 4 | 100 cases |
| **Total** | | **1,506 cases** |

- Download: [The Cancer Imaging Archive (TCIA)](https://www.cancerimagingarchive.net/)
- Also available via [Synapse](https://www.synapse.org/)
- Reference: Garrucho et al., *Scientific Data* 12(1), 453 (2025)

---

## Installation

```bash
git clone https://github.com/yaqeenali/Explainableand-Fair-Federated-Learning-with-XGBoost-for-Predicting-Pathological-Complete-Response.git
cd Explainableand-Fair-Federated-Learning-with-XGBoost-for-Predicting-Pathological-Complete-Response
pip install -r requirements.txt
```

### Requirements

```
nvflare>=2.4.0
xgboost>=2.0
pyradiomics>=3.1
shap>=0.44
scikit-learn>=1.3
pandas>=2.0
numpy>=1.24
SimpleITK>=2.3
matplotlib>=3.7
seaborn>=0.12
pyyaml>=6.0
tqdm>=4.65
jupyter>=1.0
```

---

## Usage

### Step 1 — Extract radiomic features
```bash
python feature_engineering/radiomic_extraction.py \
    --input_dir  /data/mama-mia/nifti \
    --mask_dir   /data/mama-mia/masks \
    --output_csv /data/mama-mia/radiomics.csv
```

### Step 2 — Feature selection and fusion
```bash
python feature_engineering/feature_selection.py \
    --radiomics_csv /data/mama-mia/radiomics.csv \
    --clinical_csv  /data/mama-mia/clinical.csv \
    --output_csv    /data/mama-mia/features_fused.csv \
    --top_k 6
```

### Step 3a — Centralized baseline
```bash
python models/xgboost_centralized.py \
    --features_csv /data/mama-mia/features_fused.csv \
    --output_dir   results/centralized
```

### Step 3b — Local baselines (per site)
```bash
python models/xgboost_local.py \
    --features_csv /data/mama-mia/features_fused.csv \
    --output_dir   results/local
```

### Step 3c — Federated Learning (NVIDIA FLARE)
```bash
# Start the FL server
python federated/server/fl_server.py --config configs/config.yaml

# Start each client (in separate terminals or processes)
python federated/clients/fl_client.py --site duke  --config federated/clients/site_configs/duke.yaml
python federated/clients/fl_client.py --site ispy1 --config federated/clients/site_configs/ispy1.yaml
python federated/clients/fl_client.py --site ispy2 --config federated/clients/site_configs/ispy2.yaml
python federated/clients/fl_client.py --site nact  --config federated/clients/site_configs/nact.yaml
```

### Step 4 — Evaluation, SHAP, and Fairness
```bash
# Performance metrics
python evaluation/metrics.py --model_dir results/federated --output_dir results/

# SHAP explainability
python evaluation/explainability.py --model_path results/federated/global_model.json \
                                     --features_csv /data/mama-mia/features_fused.csv

# Fairness analysis
python evaluation/fairness.py --predictions_csv results/federated/predictions.csv \
                               --subgroup_csv    /data/mama-mia/clinical.csv
```

---

## Fairness Score Formula

The fairness metric used in this study (Equations 1–3 from the paper):

$$\text{Disparity}_v = \left(\max(\text{TPR}_v) - \min(\text{TPR}_v)\right) + \left(\max(\text{FPR}_v) - \min(\text{FPR}_v)\right), \quad \forall v \in V$$

$$\overline{\text{Disparity}} = \frac{1}{|V|} \sum_{v \in V} \text{Disparity}_v$$

$$\text{Fairness Score} = 1 - \overline{\text{Disparity}}$$

Where $V$ = {age, menopausal status, scanner vendor, clinical site}. Higher scores → more equitable performance.

---

## Citation

If you use this code, please cite:

```bibtex
@inproceedings{ali2026fl,
  title     = {Explainable and Fair Federated Learning with XGBoost for Predicting
               Pathological Complete Response in Breast Cancer using
               Multi-Center DCE-MRI Data},
  author    = {Ali, Yaqeen and M{\"u}ller, Julia and Weinmann, Andreas and Gregori, Johannes},
  booktitle = {Medical Imaging 2026: Computer-Aided Diagnosis},
  volume    = {13926},
  pages     = {139260Q},
  year      = {2026},
  publisher = {SPIE},
  doi       = {10.1117/12.3085564}
}
```

Also consider citing the related papers:

```bibtex
@inproceedings{ali2025tnbc,
  title     = {Leveraging MRI Radiomics and Machine Learning for Accurate
               Differentiation of Triple-Negative Breast Cancer Subtype},
  author    = {Ali, Yaqeen and Gregori, Johannes and Tareke, Tewele W. and
               Lalande, Alain and Meriaudeau, Fabrice},
  booktitle = {2025 IEEE 38th International Symposium on Computer-Based
               Medical Systems (CBMS)},
  pages     = {922--928},
  year      = {2025},
  doi       = {10.1109/CBMS65348.2025.00185}
}

@inproceedings{ali2026pcr,
  title     = {Predicting Pathological Complete Response in Breast Cancer
               Using a Dual 3D ResNet-Transformer Architecture with
               Multimodal Data Fusion},
  author    = {Ali, Yaqeen and M{\"u}ller, Julia and Tareke, Tewele W. and
               Lalande, Alain and Meriaudeau, Fabrice and Gregori, Johannes},
  booktitle = {Medical Imaging 2026: Computer-Aided Diagnosis},
  volume    = {13926},
  pages     = {139260T},
  year      = {2026},
  doi       = {10.1117/12.3087618}
}
```

---

## Related Repositories

| Repo | Paper | Method |
|---|---|---|
| [tnbc-mri-radiomics](https://github.com/yaqeenali/tnbc-mri-radiomics) | IEEE CBMS 2025 | Radiomics + EasyEnsemble, 87 patients |
| [pcr-3dresnet-transformer](https://github.com/yaqeenali/pcr-3dresnet-transformer) | SPIE 2026 | 3D ResNet + Transformer, MAMA-MIA |
| **This repo** | SPIE 2026 | Federated XGBoost + SHAP + Fairness |

---

## Funding

This study was supported by the **Marie Skłodowska-Curie Doctoral Network** (HORIZON-MSCA-2021-DN-01-01) under Grant Agreement No. 101073222, and by **BMFTR** (Federal Ministry of Research, Technology and Space) project number 01KD25015 (MICRATE).

---

## License

This project is licensed under the MIT License — see [LICENSE](LICENSE) for details.
