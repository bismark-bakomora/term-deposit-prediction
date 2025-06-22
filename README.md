# Term Deposit Prediction

## Overview

This project addresses a classification task to predict whether a client will subscribe to a term deposit based on their demographic attributes and previous marketing interactions. The solution combines exploratory data analysis (EDA), supervised machine learning, and API deployment to deliver a production-ready predictive system.

---

## Dataset
 
- **File Used**: `bank-additional-full.csv`  
- **Target Variable**: `y` (binary: "yes" or "no")

---

## Methodology

### 1. Data Preprocessing
- Handled categorical variables via **one-hot encoding**
- Converted target labels to binary (1 = yes, 0 = no)
- Applied **SMOTE** to address class imbalance
- Saved processed feature names for model compatibility during inference

### 2. Modeling
- Algorithm: **Random Forest Classifier**
- Evaluation Metrics:  
  - Accuracy  
  - Precision  
  - Recall  
  - F1 Score  
  - ROC AUC

### 3. Results

| Metric         | Score    |
|----------------|----------|
| Accuracy       | 95.96%   |
| Precision      | 95.91%   |
| Recall         | 96.01%   |
| F1 Score       | 95.96%   |
| ROC AUC Score  | 95.96%   |

### 4. Key Predictors
- `duration` – Length of last contact
- `poutcome` – Outcome of previous campaign
- `contact` – Contact communication type
- `month` – Contact month
- `emp.var.rate` – Employment variation rate

---

## Insights

- Clients with longer call durations and successful prior engagements (`poutcome`) are more likely to subscribe.
- Campaigns during certain months (e.g., **May**) have notably lower success rates.
- Over-contacting reduces conversion likelihood.

---

## Deployment

- **Backend**: Flask API deployed on **Render**
- **Frontend**: HTML form hosted on **Vercel**, calling the Render API

---
