# 💳 Loan Default Prediction using Machine Learning

A complete end-to-end Machine Learning pipeline for predicting whether a customer is likely to default on a loan.

This project demonstrates the entire machine learning workflow, from raw data preprocessing and feature engineering to model training, evaluation, explainability, and prediction using multiple classification algorithms.

---

## 📌 Project Overview

Financial institutions face significant risks from loan defaults. Early identification of high-risk customers enables banks to make better lending decisions and reduce financial losses.

This project builds several machine learning classification models to predict loan default risk using customer financial information.

---

## 🎯 Objectives

- Explore and understand the dataset
- Clean and preprocess raw data
- Handle missing values
- Detect and cap outliers
- Perform feature engineering
- Train multiple machine learning models
- Compare model performance
- Explain model predictions
- Predict loan default for new customers

---

## 📂 Project Structure

```
loan-default-prediction/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── images/
│
├── models/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_data_cleaning.ipynb
│   ├── 03_feature_engineering.ipynb
│   ├── 04_model_training.ipynb
│   ├── 05_model_explainability.ipynb
│   └── 06_prediction.ipynb
│
├── src/
│   ├── config.py
│   ├── preprocessing.py
│   ├── feature_engineering.py
│   ├── visualization.py
│   ├── train.py
│   ├── evaluate.py
│   ├── predict.py
│   └── utils.py
│
├── requirements.txt
├── README.md
└── main.py
```

---

## 📊 Dataset

The dataset contains customer financial information used to predict loan default.

### Target Variable

- **SeriousDlqin2yrs**
  - **1** → Customer defaulted
  - **0** → Customer did not default

### Features

- RevolvingUtilizationOfUnsecuredLines
- Age
- NumberOfTime30-59DaysPastDueNotWorse
- DebtRatio
- MonthlyIncome
- NumberOfOpenCreditLinesAndLoans
- NumberOfTimes90DaysLate
- NumberRealEstateLoansOrLines
- NumberOfTime60-89DaysPastDueNotWorse
- NumberOfDependents

---

# 🛠 Data Preprocessing

The preprocessing pipeline includes:

- Loading raw dataset
- Removing unnecessary index column
- Handling missing values
- Outlier treatment using IQR
- Feature scaling using StandardScaler
- Train/Test split

---

# 📈 Exploratory Data Analysis

The following visualizations were created:

- Target Distribution
- Age Distribution
- Monthly Income Distribution
- Correlation Heatmap
- Feature Boxplots
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve
- Feature Importance

---

# 🤖 Machine Learning Models

Four classification algorithms were implemented and compared:

| Model | Status |
|--------|--------|
| Logistic Regression | ✅ |
| Decision Tree | ✅ |
| Random Forest | ✅ |
| XGBoost | ✅ |

---

# 📏 Evaluation Metrics

The models were evaluated using:

- Accuracy
- Precision
- Recall
- F1 Score

---

# 📊 Model Performance

| Model | Accuracy | Precision | Recall | F1 Score |
|--------|----------|-----------|--------|----------|
| Logistic Regression | 0.9344 | 0.2800 | 0.0036 | 0.0071 |
| Decision Tree | 0.8828 | 0.1576 | 0.1835 | 0.1696 |
| Random Forest | 0.9321 | 0.3571 | 0.0511 | 0.0894 |
| XGBoost | 0.9340 | 0.4355 | 0.0414 | 0.0756 |

> **Note:** Since the dataset is highly imbalanced, Accuracy alone is not sufficient to evaluate model performance. Recall and F1 Score provide more meaningful insights for identifying high-risk customers.

---

# 🔍 Model Explainability

Model interpretation includes:

- Feature Importance
- Confusion Matrix
- ROC Curve
- Precision-Recall Curve

---

# 🚀 Prediction

The project supports prediction for new customer records.

Example workflow:

1. Load trained model
2. Prepare customer data
3. Scale features
4. Predict default probability
5. Display prediction result

---

# 💻 Technologies Used

- Python
- Pandas
- NumPy
- Matplotlib
- Seaborn
- Scikit-learn
- XGBoost
- Jupyter Notebook

---

# 📦 Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/loan-default-prediction.git

cd loan-default-prediction
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# ▶️ Running the Project

Open Jupyter Notebook:

```bash
jupyter notebook
```

Run the notebooks in the following order:

```
01_data_understanding.ipynb

↓

02_data_cleaning.ipynb

↓

03_feature_engineering.ipynb

↓

04_model_training.ipynb

↓

05_model_explainability.ipynb

↓

06_prediction.ipynb
```

---

# 📚 Skills Demonstrated

- Data Cleaning
- Exploratory Data Analysis
- Feature Engineering
- Data Visualization
- Machine Learning
- Model Evaluation
- Model Explainability
- Predictive Analytics
- Python Programming

---

# 📄 License

This project is intended for educational and portfolio purposes.

---

# 👩‍💻 Author

**Maryam Ahmadova**

Backend Developer | Machine Learning Enthusiast

- GitHub: https://github.com/MariamAxmed
- LinkedIn: https://www.linkedin.com/in/maryam-ahmadova/