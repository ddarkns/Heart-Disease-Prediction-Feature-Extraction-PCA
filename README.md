# Heart-Disease-Prediction-Feature-Extraction-PCA
A comprehensive machine learning project demonstrating Principal Component Analysis (PCA) for feature extraction and dimensionality reduction in heart disease classification.

# Heart Disease Prediction with PCA Feature Extraction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

A comprehensive machine learning project that demonstrates feature extraction and dimensionality reduction using Principal Component Analysis (PCA) for heart disease classification.

## 📋 Table of Contents
- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)

## 🎯 Overview

This project explores how **Principal Component Analysis (PCA)** can be used to reduce the dimensionality of clinical data while maintaining predictive power for heart disease classification. We compare model performance using all original features versus PCA-reduced components.

**Key Objectives:**
- Preprocess and clean clinical heart disease data
- Apply PCA for feature extraction and dimensionality reduction
- Compare Random Forest classifier performance with vs without PCA
- Visualize high-dimensional data in 2D space
- Analyze trade-offs between model accuracy and feature reduction

## 📊 Dataset

The project uses the **Heart Failure Prediction Dataset** containing 918 patient records with clinical features.

**Target Variable:**
- `HeartDisease`: Binary classification (0 = No Heart Disease, 1 = Heart Disease)

**Clinical Features:**
- **Numerical**: Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak
- **Categorical**: Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope

## 🔧 Installation

### Prerequisites
- Python 3.7+
- pip package manager

### Setup Instructions

```bash
# 1. Clone the repository
git clone https://github.com/yourusername/Heart-Disease-Prediction-Feature-Extraction-PCA.git
cd Heart-Disease-Prediction-Feature-Extraction-PCA

# 2. Create virtual environment (recommended)
python -m venv heart_env
source heart_env/bin/activate  # On Windows: heart_env\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt
