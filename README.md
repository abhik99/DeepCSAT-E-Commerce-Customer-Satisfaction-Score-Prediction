# 🛒 DeepCSAT: E-Commerce Customer Satisfaction Score Prediction

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=for-the-badge&logo=python&logoColor=white)](https://python.org)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](https://tensorflow.org)
[![Keras](https://img.shields.io/badge/Keras-Deep%20Learning-D00000?style=for-the-badge&logo=keras&logoColor=white)](https://keras.io)
[![Streamlit](https://img.shields.io/badge/Streamlit-Live%20App-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://streamlit.io)
[![Scikit-Learn](https://img.shields.io/badge/Scikit--Learn-Machine%20Learning-F7931E?style=for-the-badge&logo=scikit-learn&logoColor=white)](https://scikit-learn.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg?style=for-the-badge)](LICENSE)

An end-to-end predictive deep learning system designed to forecast Customer Satisfaction (CSAT) scores (1–5) in real time using 85,907 customer interactions. Combines operational latency metrics, TF-IDF natural language processing on customer remarks, synthetic oversampling (SMOTE), and a 6-layer Deep Artificial Neural Network (ANN) served via an interactive Streamlit UI.

---

## 📌 Table of Contents
- [Project Overview](#-project-overview)
- [System Architecture](#-system-architecture)
- [Dataset & Operational Features](#-dataset--operational-features)
- [Deep Learning Architecture](#-deep-learning-architecture)
- [Comparative Benchmark](#-comparative-benchmark)
- [Streamlit Web Application](#-streamlit-web-application)
- [Repository Structure](#-repository-structure)
- [Installation & Setup](#-installation--setup)
- [Author & Connect](#-author--connect)

---

## 🚀 Project Overview
Customer Satisfaction (CSAT) directly dictates customer retention, repeat purchases, and churn risk in large-scale e-commerce operations.

**DeepCSAT** solves key challenges in customer experience intelligence:
- **High-Sparsity Operational Data:** Addresses severe data sparsity across handling durations, agent connection times, and customer tenure.
- **Multimodal Feature Space:** Blends numerical timestamps, categorical routing tags, and free-form customer text feedback.
- **Severe Class Imbalance:** Counteracts high concentration of 5-star ratings against dissatisfied customers (ratings 1, 2, and 3) using SMOTE and class-weight balancing.
- **Production Deployment:** Serialized model (`csat_model.h5`), feature transformer (`features.pkl`), and `scaler.pkl` integrated into a real-time Streamlit dashboard (`app.py`).

---

## 🏗️ System Architecture

```text
[ 85,907 Customer Interaction Logs & Support Records ]
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│             Data Cleaning & Preprocessing              │
│   • Median & Mode imputation for sparse timestamps     │
│   • Cyclical date-time feature engineering             │
│   • Channel & category one-hot encoding                │
│   • TF-IDF feature extraction on customer feedback     │
└────────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│             Imbalance Mitigation Pipeline              │
│   • Synthetic Minority Over-sampling Technique (SMOTE) │
│   • Stratified 3-Fold Cross-Validation                 │
│   • Dynamic learning rate reduction & Early Stopping   │
└────────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│       6-Layer Deep Artificial Neural Network (ANN)     │
│   • Dense (256) -> BatchNorm -> Dropout(0.3) -> L2     │
│   • Dense (128) -> BatchNorm -> Dropout(0.3) -> L2     │
│   • Dense (64)  -> BatchNorm -> Dropout(0.2)           │
│   • Softmax 5-Class Output (CSAT 1 to 5)               │
└────────────────────────────────────────────────────────┘
                       │
                       ▼
┌────────────────────────────────────────────────────────┐
│             Interactive Streamlit Dashboard            │
│   Enables support leads to simulate customer inputs    │
│   and forecast immediate satisfaction ratings          │
└────────────────────────────────────────────────────────┘
```

---

## 📊 Dataset & Operational Features
- **Total Records:** 85,907 customer interactions.
- **Target Variable:** CSAT rating (1 = Very Dissatisfied, 2 = Dissatisfied, 3 = Neutral, 4 = Satisfied, 5 = Highly Satisfied).
- **Operational Metrics:** Connection wait times, agent response latency, order category, customer channel, handling duration.
- **NLP Features:** TF-IDF n-gram vectors extracted from customer sentiment remarks.

---

## 🧠 Comparative Benchmark

| Model | Architecture / Algorithm | Accuracy / F1-Macro | Strengths |
|---|---|---|---|
| **DeepCSAT ANN** | **6-Layer Deep Neural Net (Keras)** | **Top Generalization** | **Best minority class recall & feature interaction** |
| LightGBM | Gradient Boosted Decision Trees | High Efficiency | Fast training on sparse categorical splits |
| XGBoost | Extreme Gradient Boosting | Strong Baseline | Resilient to non-linear operational outliers |
| Random Forest | 100 Trees Ensemble | Balanced F1 | Robust against feature collinearity |
| Decision Tree | Single CART Tree | Baseline Interpretability | Clear threshold rules |
| Logistic Regression | Multinomial Softmax | Benchmark Baseline | High bias, fast convergence |

---

## 📁 Repository Structure
```text
├── Ecommerce_CSAT.ipynb             # Research notebook (EDA, SMOTE, ANN training)
├── app.py                            # Streamlit web application
├── csat_model.h5                     # Serialized Keras Deep ANN weights
├── features.pkl                      # Fitted feature pipeline and transformers
├── scaler.pkl                        # Fitted StandardScaler object
├── requirements.txt                  # Python dependencies
└── models/                           # Comparative baseline models
    ├── decision_tree_model.pkl
    ├── lightgbm_model.pkl
    ├── logistic_regression_model.pkl
    ├── random_forest_model.pkl
    └── xgboost_model.pkl
```

---

## 🛠️ Installation & Setup

```bash
# 1. Clone repository
git clone https://github.com/abhik99/DeepCSAT-E-Commerce-Customer-Satisfaction-Score-Prediction.git
cd DeepCSAT-E-Commerce-Customer-Satisfaction-Score-Prediction

# 2. Setup virtual environment
python -m venv venv
venv\Scripts\activate  # On Windows

# 3. Install dependencies
pip install -r requirements.txt

# 4. Launch Streamlit UI
streamlit run app.py
```

---

## 👨‍💻 Author & Connect

**Abhishek Kumar (Abhi)**  
*Data Scientist & Analytics Engineer*

- 🌐 **Portfolio:** [abhicodes.online](https://abhicodes.online)
- 💼 **LinkedIn:** [linkedin.com/in/abhik99](https://linkedin.com/in/abhik99)
- 🐙 **GitHub:** [@abhik99](https://github.com/abhik99)
- 📧 **Email:** [akdevil27boy@gmail.com](mailto:akdevil27boy@gmail.com)

---
*Distributed under the MIT License.*
