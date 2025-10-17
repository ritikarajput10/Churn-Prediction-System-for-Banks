# Bank Churn Prediction

This project predicts whether a bank customer will churn using machine learning. It includes a complete pipeline from data preprocessing and model training to deployment with a Flask web app.

---

## Project Overview

* Predicts customer churn based on features like age, tenure, credit score, active membership, and number of products.
* Handles imbalanced data using **SMOTE**.
* Gradient Boosting Classifier was used as the best-performing model.
* Provides real-time predictions through a **Flask web application**.

---

## Features

* Data preprocessing and feature engineering
* Model training and evaluation
* Flask app for predictions

---

## Technologies Used

* Python, Pandas, Scikit-learn
* Flask for web deployment
* MLflow for model tracking
* SMOTE for balancing the dataset

---

## Installation

1. Create and activate a Conda environment:

```bash
conda create --name churn_prediction python=3.10
conda activate churn_prediction
```

2. Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Application

```bash
python app.py
```

---

## Model Performance

* **Accuracy:** 86%
* **Precision:** 66%
* **Recall:** 62%
* **F1-score:** 60%

---
