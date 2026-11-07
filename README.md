# 🧠 Explainable AI for Brain Tumor Diagnosis via Multi-Modal CNN-ViT Fusion & Differentiable XAI Constraints

[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch)](https://pytorch.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Domain: Neuroimaging](https://img.shields.io/badge/Domain-Neuroimaging%20%26%20XAI-1B365D)]()

## 📌 Project Overview
Standard Deep Learning models and Vision Transformers (ViTs) often act as "black boxes" in neuro-oncological image analysis, frequently exhibiting off-target attention (e.g., focusing on ventricles or skull boundaries rather than tumor margins). 

This research introduces a novel, trustworthy multi-modal architecture for classifying 4-class brain MRI modalities (**Glioma**, **Meningioma**, **Pituitary Tumor**, and **No Tumor**). Our framework combines global-local visual feature learning with radiomic profiling and introduces a novel **training-time differentiable spatial entropy loss** that actively constrains model attention onto true anatomical pathology.

---

## ✨ Key Research Contributions & Architecture

1. **Multi-Modal Hybrid Architecture:** Integrates a local CNN feature extraction stream (**EfficientNet-B3**) with a global Vision Transformer stream (**google/vit-base-patch16-224-in21k**).
2. **Cross-Attention Spatial Fusion:** Fuses global ViT tokens (Query) with dense local CNN feature maps (Key/Value) using scaled dot-product cross-attention.
3. **Tabular Radiomic Profiling:** Incorporates 32 extracted radiomic features (GLCM texture, Local Binary Patterns, and morphological shape properties) via a dedicated Multi-Layer Perceptron (MLP) stream.
4. **Differentiable XAI & Self-Supervised Constraint Loss:** Enforces training-time spatial entropy minimization on Grad-CAM feature maps combined with an Otsu-derived self-supervised pseudo-mask Dice loss to eliminate spurious background attention.

---

## 👥 Project Team & Responsibilities

| Team Member | Role | Core Technical Responsibilities |
| :--- | :--- | :--- |
| **Sandy Antonius** | Team Lead & System Architect | Hybrid Architecture Design (`model.py`), Training Loop (`train.py`), Cross-Attention Fusion, Methods & Discussion Writing. |
| **Mariam** | XAI & Evaluation Lead | Differentiable Grad-CAM Loss, Spatial Entropy Loss, Self-Supervised Pseudo-Mask (`Phase 4 Lead`), Evaluation Scripts. |
| **Youstina** | Software Engineering & Pipeline | PyTorch Dataset Classes (`dataset_brain.py`), Dataset Statistics Computation (`compute_stats_brain.py`), Radiomic Loaders. |
| **Hana** | Data Science & EDA | Exploratory Data Analysis, Scanner Drift & Intensity Histograms, 32 Radiomic Feature Extractions, McNemar's Significance Testing. |
| **Ahmed Ehab** | Literature & Theory | FT-ViT & Survey Analysis, Gap Analysis Document, Introduction & Related Work Drafting. |
| **Ahmed Mohammed** | Documentation & Quality Control | Visual Asset Management, WandB Metric Logging, Methods Subsections, Paper Conclusion & Submission Checklists. |

---

## 📅 Research Phases & Project Roadmap

- **Phase 0:** Reading, Clinical Grounding & Architectural Derivations (Weeks 1–3)
- **Phase 1:** Exploratory Data Analysis & Radiomic Extraction (Weeks 1–2 Concurrent)
- **Phase 2:** Baseline Single-Stream EfficientNet-B3 Benchmarking (Weeks 4–5)
- **Phase 3:** Hybrid CNN-ViT Cross-Attention Architecture Assembly (Weeks 6–7)
- **Phase 4:** Differentiable XAI Entropy Loss & Pseudo-Mask Training (Weeks 8–10)
- **Phase 5:** Optuna Hyperparameter Search & 5-Fold Systematic Ablations (Weeks 11–13)
- **Phase 6:** Manuscript Synthesis & Journal Submission to *CBM / AIIM* (Weeks 14–16)

---

## 📂 Repository Structure

```text
├── docs/                      # Dataset statistics, split indices, baseline CAM logs
├── src/
│   ├── dataset_brain.py        # PyTorch Dataset class & radiomic integration
│   ├── compute_stats_brain.py  # Dataset-specific mean/std normalization
│   ├── model_brain.py          # Hybrid CNN-ViT, Cross-Attention, and Radiomic MLP
│   ├── train_brain.py          # Training loop with composite XAI multi-task loss
│   ├── evaluate.py             # Inference, confusion matrix, and quantitative metrics
│   └── gradcam_baseline.py     # Differentiable Grad-CAM & spatial entropy modules
├── data/                       # Kaggle Brain Tumor MRI Dataset (masoudnickparvar)
├── README.md                   # Project overview & documentation
└── requirements.txt            # Python dependencies
