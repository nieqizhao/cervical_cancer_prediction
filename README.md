# cervical_cancer_prediction
This repository contains code and data processing pipelines for building predictive models to detect cervical cancer based on clinical and behavioral features.

## Problem Statement
Cervical cancer is a preventable disease if detected early. We aim to build interpretable and accurate models that can assist in screening and risk stratification.

## Features and Model building
- Data preprocessing and imputation in preprocess.py
- Feature engineering and selection in preprocess.py
- Multiple ML models (Logistic Regression, Naïve Bayes, SVM, KNN, Decision Tree, Random Forest, XGBoost, LightGBM, AdaBoost, MLP and CatBoost.) in modeling.py
- Grid Search for hyperparameters and Model evaluation (AUC, confusion matrix, PR curve) in modeling.py
- SHAP explainability in shap.py

## Dataset
We use the Fifth Affiliated Hospital of Sun Yat-sen University, which contains demographic, behavioral, and medical history data.

## Installation
```bash
git clone https://github.com/nieqizhao/cervical-cancer-prediction.git
cd cervical-cancer-prediction
pip install -r requirements.txt

