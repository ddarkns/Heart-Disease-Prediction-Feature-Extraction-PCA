# Heart-Disease-Prediction-Feature-Extraction-PCA
A comprehensive machine learning project demonstrating Principal Component Analysis (PCA) for feature extraction and dimensionality reduction in heart disease classification.

---

# ❤️ Heart Disease Prediction with PCA Feature Extraction

![Python](https://img.shields.io/badge/Python-3.7%2B-blue)
![Machine Learning](https://img.shields.io/badge/Machine-Learning-orange)
![Scikit-learn](https://img.shields.io/badge/Scikit--Learn-1.0%2B-green)
![Jupyter](https://img.shields.io/badge/Jupyter-Notebook-orange)

A comprehensive machine learning project that demonstrates feature extraction and dimensionality reduction using Principal Component Analysis (PCA) for heart disease classification.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Dataset](#dataset)
- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Methodology](#methodology)
- [Results](#results)
- [Project Structure](#project-structure)
- [Future Enhancements](#future-enhancements)
- [Author](#author)
- [Acknowledgments](#acknowledgments)

---

## 🎯 Overview

This project explores how **Principal Component Analysis (PCA)** can be used to reduce the dimensionality of clinical data while maintaining predictive power for heart disease classification.

**Key Objectives:**
- Preprocess and clean clinical heart disease data
- Apply PCA for feature extraction and dimensionality reduction
- Compare Random Forest classifier performance with vs without PCA
- Visualize high-dimensional data in 2D space
- Analyze trade-offs between model accuracy and feature reduction

---

## 📊 Dataset

The project uses the **Heart Failure Prediction Dataset** containing **918 patient records** with clinical features.

**Target Variable:**
- `HeartDisease` — Binary classification (0 = No Heart Disease, 1 = Heart Disease)

**Clinical Features:**
- **Numerical:** Age, RestingBP, Cholesterol, FastingBS, MaxHR, Oldpeak  
- **Categorical:** Sex, ChestPainType, RestingECG, ExerciseAngina, ST_Slope

---

## 🔧 Installation

### ✅ Prerequisites
- Python 3.7+
- pip package manager

### 📦 Dependencies (requirements.txt)

    pandas>=1.3.0
    scikit-learn>=1.0.0
    matplotlib>=3.5.0
    numpy>=1.21.0
    jupyter>=1.0.0
    seaborn>=0.11.0

Install dependencies:

    pip install -r requirements.txt

---

## 🚀 Usage

Run the Notebook:

    jupyter notebook PCA_tutorial_digits.ipynb

---

## 📓 Notebook Workflow (Sequential Steps)

1. **Data Loading & Exploration** — Understand dataset structure  
2. **Data Preprocessing** — Handle categorical variables and outliers  
3. **Feature Engineering** — One-hot encoding & standardization  
4. **PCA Implementation** — Dimensionality reduction  
5. **Model Training** — Random Forest classifier  
6. **Evaluation & Visualization** — Performance comparison & plots  

---

## 🔬 Methodology

### 1. Data Preprocessing
- One-hot encoding for categorical variables (`ChestPainType`, `RestingECG`)
- Outlier detection using **Z-score method** (±3 standard deviations)
- Standardization using **StandardScaler** (required for PCA)

### 2. Principal Component Analysis (PCA)

    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X_scaled)

### 3. Model Training & Evaluation
- Algorithm: **Random Forest Classifier**
- Train/Test Split: **80/20**
- Metrics: Accuracy, Precision, Recall, F1-Score

---
##  📈  Results & Outputs:
### 🔍 Finding the Optimal Number of Components
<img width="1068" height="752" alt="image" src="https://github.com/user-attachments/assets/32fabbc7-5b69-401c-9efa-cd271ddbc7b6" />

### 🎯 PC1 vs PC2: Patient Clustering Visualization
<img width="1129" height="724" alt="image" src="https://github.com/user-attachments/assets/52d9c6ae-1333-49c4-a9fb-805a5b3bf176" />

### 🏆 Top Contributing Features Analysis
<img width="1176" height="700" alt="image" src="https://github.com/user-attachments/assets/170aecf1-64ee-4520-8873-fef11edaa593" />




---

## 📁 Project Structure

    Heart-Disease-Prediction-Feature-Extraction-PCA/
    │
    ├── main.ipynb      # Main analysis notebook
    ├── requirements.txt               # Python dependencies
    ├── README.md                      # Project documentation
    └── data/
        └── heart.csv                  # Dataset file

---


## 🔮 Future Enhancements
- Try t-SNE and UMAP for dimensionality reduction
- Test alternate classifiers (XGBoost, SVM, Neural Networks)
- Add cross-validation and hyperparameter tuning
- Deploy as a web application
- Add SHAP values for interpretability

---

## 👨‍💻 Author
GitHub: `@ddarkns`

---

## 🙏 Acknowledgments
- Dataset: Heart Failure Prediction Dataset (Kaggle)
- Scikit-learn for machine learning tools  
- Matplotlib/Seaborn for data visualization
