# AdaptFoundation 1st lab

**Adaptation of Foundation Models for Cortical Folding Analysis**

This repository contains the implementation of Phase 1 of AdaptFoundation, a project focused on evaluating and adapting foundation models (DINOv2, CLIP) for 3D neuroanatomical data analysis, specifically cortical sulcal skeletons. 

## Project Overview

This lab aims to:
- Evaluate the effectiveness of 2D foundation models for 3D neuroimaging tasks
- Develop modular 3D→2D adaptation strategies
- Compare performance on neuroanatomical classification tasks
- Provide extensible pipeline for rapid experimentation

## Repository Structure

```
adaptfoundation/
├── data/
│   ├── loaders.py              # Data loading utilities
│   └── preprocessing.py        # Data preprocessing functions
├── models/
│   ├── slicing.py             # 3D→2D slicing strategies
│   ├── feature_extraction.py  # Foundation model feature extraction
│   └── aggregation.py         # Multi-axis feature aggregation
├── classification/
│   ├── linear_probing.py      # Linear probing implementations
│   └── metrics.py             # Evaluation metrics
├── pipelines/
│   └── skeleton_pipeline.py   # Complete processing pipeline
├── configs/
│   └── default_config.yaml    # Default configuration
├── utils/
│   ├── visualization.py       # Data visualization utilities
│   └── validation.py          # Validation functions
├── exploration.ipynb          # Exploratory analysis notebook
├── requirements.txt           # Python dependencies
└── README.md                  # This file
```

## Installation

### 1. Clone the repository
```bash
git clone https://gitlab.telecom-paris.fr/telecom-neurospin/adaptfoundation.git
cd adaptfoundation
```

### 2. Create virtual environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install dependencies
```bash
pip install -r requirements.txt
```

### 4. Install package in development mode
```bash
pip install -e .
```

## Quick Start

### Data Exploration
```python
from utils.visualization import explore_hcp_ofc_dataset

# Explore HCP OFC dataset
explore_hcp_ofc_dataset("crops/2mm/S.Or")
```

### Basic Pipeline Usage
```python
from pipelines.skeleton_pipeline import Pipeline3DTo2D

# Initialize pipeline
pipeline = Pipeline3DTo2D(config_path="configs/default_config.yaml")

# Run complete evaluation
results = pipeline.run_evaluation(
    data_path="path/to/data",
    split_type="cross_validation"
)
```

## Dataset Structure

The lab works with HCP OFC (Human Connectome Project - Orbitofrontal Cortex) dataset:

- **Format**: Cortical sulcal skeletons in .npy format
- **Structure**: Binary 3D volumes (0=background, 1=skeleton surface)
- **Tasks**: 4-class classification on Left_OFC labels
- **Splits**: Stratified cross-validation splits provided

### Expected Data Organization
```
crops/2mm/S.Or/
├── Lskeleton.npy              # All skeleton crops
├── Lskeleton_subject.csv      # Subject mapping
├── hcp_OFC_labels.csv         # Classification labels
└── splits/
    ├── train_val_split_0.csv  # CV split 0
    ├── train_val_split_1.csv  # CV split 1
    ├── ...
    └── test_split.csv         # Final test split
```

## Pipeline Overview

The AdaptFoundation pipeline implements:

1. **3D→2D Adaptation**: Multi-axis slicing (axial, coronal, sagittal)
2. **Feature Extraction**: DINOv2-384D feature extraction from 2D slices
3. **Aggregation**: Average pooling per axis → concatenation (3×384=1152D)
4. **Optional PCA**: Dimensionality reduction
5. **Linear Probing**: Logistic Regression/Linear SVM/KNN classification

## Evaluation Metrics

- **Classification**: ROC-AUC (One vs All for multi-class)
- **Validation**: 5-fold stratified cross-validation
- **Final Evaluation**: Hold-out test set

## Configuration

Pipeline behavior is controlled via YAML configuration files. See `configs/default_config.yaml` for available options.
