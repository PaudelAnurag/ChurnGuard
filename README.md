# 🚀 ChurnGuard — Customer Churn Prediction System

> End-to-end machine learning system to predict telecom customer churn in real time  
> ⚡ FastAPI backend · 🎨 Vanilla HTML frontend · 🤖 14 ML models trained

---

## 📌 Overview

ChurnGuard is a full-stack machine learning system that predicts whether a telecom customer will churn.

It includes:
- 14 trained ML models
- FastAPI REST API for inference
- Real-time frontend dashboard
- Full preprocessing + SMOTE pipeline

---

## 📂 Project Structure
Customer Churn/
│
├── data/
│ ├── raw/
│ │ └── customer_churn.csv
│ └── processed/
│ └── Processed_customer_churn.csv
│
├── notebooks/
│ └── Customer_Churn.ipynb
│
├── reports/
│ ├── models/
│ │ ├── gradient_boosting.pkl
│ │ ├── lightgbm.pkl
│ │ ├── xgboost.pkl
│ │ └── ...
│ └── images/
│
├── app.py
├── index.html
└── README.md


---

## 📊 Dataset

**Kaggle Telco Customer Churn Dataset**

- 7,043 rows
- 20 features + target variable

| Feature | Type | Description |
|--------|------|-------------|
| gender | Binary | Male / Female |
| SeniorCitizen | Binary | 0 / 1 |
| Partner | Binary | Yes / No |
| Dependents | Binary | Yes / No |
| tenure | Numeric | Months with company |
| PhoneService | Binary | Yes / No |
| MultipleLines | Categorical | No / Yes / No phone service |
| InternetService | Categorical | DSL / Fiber optic / No |
| OnlineSecurity | Categorical | No / Yes / No internet service |
| OnlineBackup | Categorical | No / Yes / No internet service |
| DeviceProtection | Categorical | No / Yes / No internet service |
| TechSupport | Categorical | No / Yes / No internet service |
| StreamingTV | Categorical | No / Yes / No internet service |
| StreamingMovies | Categorical | No / Yes / No internet service |
| Contract | Ordinal | Month-to-month / One year / Two year |
| PaperlessBilling | Binary | Yes / No |
| PaymentMethod | Categorical | Payment type |
| MonthlyCharges | Numeric | Monthly bill |
| TotalCharges | Numeric | Total charges |
| Churn | Target | 0 = No, 1 = Yes |

---

## ⚙️ Preprocessing Pipeline

Raw Data
↓
Encoding (Label Encoding + OneHot Encoding)
↓
Scaling (StandardScaler)
↓
SMOTE (training only)
↓
Model Training


---

## 🤖 Model Performance Comparison

Sorted by ROC-AUC score:

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|------|---------|-----------|--------|----|--------|
| Logistic Regression | 0.7793 | 0.5644 | 0.7380 | 0.6396 | 0.8469 |
| Gradient Boosting | 0.7651 | 0.5410 | 0.7594 | 0.6318 | 0.8475 |
| AdaBoost | 0.7736 | 0.5577 | 0.7112 | 0.6251 | 0.8387 |
| LightGBM | 0.7388 | 0.5049 | 0.8342 | 0.6290 | 0.8347 |
| SVM | 0.7537 | 0.5242 | 0.7834 | 0.6281 | 0.8264 |
| Random Forest | 0.7168 | 0.4806 | 0.8289 | 0.6084 | 0.8218 |
| XGBoost | 0.7317 | 0.4966 | 0.7781 | 0.6062 | 0.8179 |
| Naive Bayes | 0.6856 | 0.4493 | 0.8182 | 0.5801 | 0.8117 |
| Bagging | 0.7026 | 0.4661 | 0.8262 | 0.5959 | 0.8117 |
| Extra Trees | 0.6941 | 0.4568 | 0.8048 | 0.5828 | 0.7956 |
| KNN | 0.7076 | 0.4688 | 0.7620 | 0.5804 | 0.7939 |
| Decision Tree | 0.7260 | 0.4855 | 0.5374 | 0.5102 | 0.6656 |

---

## 🧠 Best Model — Gradient Boosting

precision recall f1-score support

No Churn 0.87 0.81 0.84 1035
Churn 0.56 0.68 0.61 374

accuracy 0.77 1409
macro avg 0.72 0.74 0.73 1409
weighted avg 0.79 0.77 0.78 1409


---

## 🌐 API Reference

### GET `/models`

```json
{
  "models": ["gradient_boosting", "lightgbm", "xgboost"]
}

```

### POST `/predict`

```json
{
  "model_name": "gradient_boosting",
  "gender": 1,
  "SeniorCitizen": 0,
  "Partner": 1,
  "Dependents": 0,
  "tenure": 24,
  "PhoneService": 1,
  "MultipleLines": "No",
  "InternetService": "Fiber optic",
  "OnlineSecurity": "No",
  "OnlineBackup": "No",
  "DeviceProtection": "No",
  "TechSupport": "No",
  "StreamingTV": "No",
  "StreamingMovies": "No",
  "Contract": 0,
  "PaperlessBilling": 1,
  "PaymentMethod": "Electronic check",
  "MonthlyCharges": 70.35,
  "TotalCharges": 1680.25
}

```



## 🧪 Tech Stack
Python (pandas, scikit-learn, imbalanced-learn)
FastAPI
XGBoost / LightGBM / Sklearn models
HTML + Vanilla JS
Bootstrap UI

```
