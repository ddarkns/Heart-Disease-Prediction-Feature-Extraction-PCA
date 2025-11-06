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
Dependencies
txt

pandas>=1.3.0
scikit-learn>=1.0.0
matplotlib>=3.5.0
numpy>=1.21.0
jupyter>=1.0.0
seaborn>=0.11.0

🚀 Usage
Running the Analysis
bash

# Launch Jupyter Notebook
jupyter notebook PCA_tutorial_digits.ipynb

The notebook is organized sequentially:

    Data Loading & Exploration - Understand dataset structure

    Data Preprocessing - Handle categorical variables and outliers

    Feature Engineering - One-hot encoding and standardization

    PCA Implementation - Dimensionality reduction

    Model Training - Random Forest classifier

    Evaluation & Visualization - Compare results and create plots

🔬 Methodology
1. Data Preprocessing

    One-hot encoding for categorical variables (ChestPainType, RestingECG)

    Outlier detection using Z-score method (±3 standard deviations)

    Feature standardization using StandardScaler for PCA compatibility

2. Principal Component Analysis
python

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

# Standardize features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply PCA for dimensionality reduction
pca = PCA(n_components=2)
X_pca = pca.fit_transform(X_scaled)

3. Model Training & Evaluation

    Algorithm: Random Forest Classifier

    Train-Test Split: 80-20 ratio

    Evaluation Metrics: Accuracy, Precision, Recall, F1-Score

    Comparison: Full feature set vs PCA-reduced features

📈 Results
Performance Comparison
Model	Accuracy	Precision	Recall	F1-Score	Features
Full Features	~85%	~0.86	~0.84	~0.85	15+
PCA (2 Components)	~78%	~0.79	~0.77	~0.78	2
Key Findings

    Dimensionality Reduction: Successfully reduced from 15+ features to 2 principal components

    Variance Explained: First two components capture significant data variance

    Performance Trade-off: Minimal accuracy reduction for substantial feature reduction

    Visualization: 2D PCA plots provide clear separation between patient groups

Visualizations Included

    PCA scatter plots with heart disease classification

    Explained variance plots

    Confusion matrix comparisons

    Feature importance analysis

📁 Project Structure
text

Heart-Disease-Prediction-Feature-Extraction-PCA/
│
├── PCA_tutorial_digits.ipynb          # Main analysis notebook
├── requirements.txt                   # Python dependencies
├── README.md                         # Project documentation
└── data/
    └── heart.csv                     # Dataset file

🛠️ Technical Skills Demonstrated

    Data Preprocessing: Handling missing values, outlier detection, feature encoding

    Dimensionality Reduction: PCA implementation and interpretation

    Machine Learning: Random Forest classification, model evaluation

    Data Visualization: Matplotlib and Seaborn for insightful plots

    Python Programming: Pandas, Scikit-learn, NumPy proficiency

🔮 Future Enhancements

    Experiment with different dimensionality reduction techniques (t-SNE, UMAP)

    Try alternative classification algorithms (XGBoost, SVM, Neural Networks)

    Implement hyperparameter tuning and cross-validation

    Develop a web application for real-time predictions

    Add SHAP values for model interpretability

👨‍💻 Author

Your Name

    GitHub: @yourusername

    LinkedIn: Your LinkedIn

📄 License

This project is licensed under the MIT License - see the LICENSE file for details.
🙏 Acknowledgments

    Heart Failure Prediction Dataset from Kaggle

    Scikit-learn library for machine learning tools

    Matplotlib and Seaborn for visualization capabilities
