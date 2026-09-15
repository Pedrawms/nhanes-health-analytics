#  Personal Health Analytics

An end-to-end **Data Analysis + Machine Learning + Streamlit** project using publicly available **NHANES 2021–2023** data.

The project explores demographic, anthropometric, laboratory, and smoking-related features and builds a machine learning model to estimate the likelihood of **elevated blood pressure**.

> **Disclaimer:** This project is intended for educational and research purposes only. It is not a medical diagnostic tool and should not replace professional medical advice.

---

##  Project Overview

This project was built to demonstrate a complete machine learning workflow:

**Raw Data → Data Cleaning → Exploratory Analysis → Feature Engineering → Machine Learning → Model Evaluation → Deployment**

The final model is integrated into an interactive Streamlit application where users can enter personal and health-related measurements and receive a model-based risk classification.

---

##  Objective

The main objective is to build an interpretable machine learning model that estimates whether an individual is more likely to have **elevated blood pressure** based on selected demographic, anthropometric, laboratory, and smoking-related features.

The project focuses on the technical machine learning workflow rather than clinical diagnosis.

---

##  Dataset

The project uses publicly available data from the:

**National Health and Nutrition Examination Survey (NHANES), 2021–2023**

Selected NHANES datasets include:

* Demographics (`DEMO_L`)
* Body Measures (`BMX_L`)
* Blood Pressure (`BPXO_L`)
* Vitamin D (`VID_L`)
* HDL Cholesterol (`HDL_L`)
* Triglycerides (`TRIGLY_L`)
* Smoking (`SMQ_L`)

The participant identifier `SEQN` was used to merge the datasets.

### Data Privacy

Raw NHANES files are not included in the GitHub repository.

The project uses publicly available, de-identified survey data.

---

##  Data Preparation

The data preparation process included:

* Loading SAS Transport (`.XPT`) files with Pandas
* Selecting relevant variables
* Renaming variables for readability
* Filtering the dataset to adults
* Handling special NHANES missing-value codes
* Investigating missingness patterns
* Merging multiple datasets using `SEQN`
* Creating aggregated blood pressure measurements
* Constructing the target variable
* Checking duplicate participant IDs
* Investigating extreme values and potential outliers

### Blood Pressure Feature Engineering

Multiple blood pressure measurements were available for each participant.

Average systolic and diastolic blood pressure were calculated from available measurements.

The target was defined as:

```text
high_bp = 1
if systolic BP >= 130 OR diastolic BP >= 80
```

Otherwise:

```text
high_bp = 0
```

Participants without available blood pressure measurements were excluded from model training.

---

##  Features

The final model uses the following features:

### Numerical Features

* Age
* Household size
* BMI
* Waist circumference
* Weight
* Height
* HDL cholesterol
* Vitamin D

### Categorical Features

* Gender
* Race / Ethnicity
* Education level
* Marital status
* Lifetime smoking history

The **Family Income-to-Poverty Ratio** was initially considered but was removed from the final model to simplify the user-facing application and improve interpretability for general users.

---

##  Preventing Data Leakage

Blood pressure measurements were used to construct the target variable.

Therefore, blood pressure measurements were **not used as model features**.

This prevents the model from directly observing the information it is being asked to predict.

The following variables were therefore excluded from the final feature set:

* Systolic blood pressure
* Diastolic blood pressure
* Individual blood pressure readings

---

##  Machine Learning Pipeline

The project compares multiple machine learning approaches:

* Logistic Regression
* Random Forest
* Gradient Boosting

The final model was selected based on predictive performance, interpretability, and suitability for deployment.

### Preprocessing

Numerical features:

1. Median imputation
2. Standard scaling

Categorical features:

1. Most-frequent imputation
2. One-hot encoding

All preprocessing steps are included inside the Scikit-learn pipeline to prevent preprocessing leakage.

---

##  Final Model

The final model is:

**Logistic Regression**

Configuration:

```text
C = 3
max_iter = 1000
```

The classification threshold was selected using out-of-fold predictions on the training set rather than optimizing the test set.

### Decision Threshold

```text
Threshold = 0.29
```

A model score:

```text
>= 0.29
```

is classified as a higher likelihood of elevated blood pressure.

---

##  Model Performance

Performance on the held-out test set:

| Metric    | Score |
| --------- | ----: |
| Accuracy  | 0.600 |
| Precision | 0.502 |
| Recall    | 0.872 |
| F1 Score  | 0.637 |
| ROC-AUC   | 0.677 |

### Confusion Matrix

```text
                 Predicted
                0        1
Actual 0      305      427
Actual 1       63      430
```

The model was intentionally evaluated using multiple metrics rather than accuracy alone.

Because the final threshold prioritizes identifying more positive cases, the model achieves relatively high recall at the cost of lower specificity/precision.

---

##  Threshold Selection

Instead of relying on the default classification threshold of `0.50`, out-of-fold predictions were generated on the training set.

Different thresholds were evaluated using F1 score.

The final threshold was selected as:

```text
0.29
```

This threshold was then applied to the untouched test set.

This approach avoids selecting the threshold directly on the test data.

---

##  Streamlit Application

The trained model is deployed through a Streamlit application.

The application allows users to enter:

* Age
* Gender
* Race / Ethnicity
* Education
* Marital status
* Household size
* Weight
* Height
* Waist circumference
* HDL cholesterol
* Vitamin D
* Lifetime smoking history

BMI is automatically calculated from height and weight.

The application then displays:

* Model Score
* Decision Threshold
* Classification Result
* Input Summary
* Model Interpretation
* Educational Disclaimer

---

##  Project Structure

```text
nhanes-health-analytics/
│
├── data/
│   ├── raw/
│   └── processed/
│
├── notebooks/
│   └── 01_data_understanding.ipynb
│
├── src/
│
├── app/
│   └── app.py
│
├── models/
│   ├── logistic_regression_final.joblib
│   └── model_config.json
│
├── README.md
├── requirements.txt
└── .gitignore
```

---

##  Installation

Clone the repository:

```bash
git clone <https://github.com/Pedrawms/nhanes-health-analytics>
cd nhanes-health-analytics
```

Create a virtual environment:

### Windows

```bash
python -m venv .venv
.venv\Scripts\activate
```

### macOS / Linux

```bash
python3 -m venv .venv
source .venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

##  Run the Application

From the project root:

```bash
streamlit run app/app.py
```

The application will open in your browser.

---

##  Main Technologies

* Python
* Pandas
* NumPy
* Scikit-learn
* Streamlit
* Joblib
* Jupyter Notebook

---

##  Key Machine Learning Concepts Demonstrated

This project demonstrates practical understanding of:

* Data cleaning
* Missing data analysis
* Feature engineering
* Categorical encoding
* Numerical scaling
* Train/test splitting
* Stratified sampling
* Cross-validation
* Out-of-fold predictions
* Hyperparameter tuning
* Classification threshold optimization
* ROC-AUC
* Precision
* Recall
* F1 score
* Confusion matrix
* Data leakage prevention
* Model serialization
* Machine learning deployment

---

##  Limitations

This project has several important limitations.

### 1. Not a clinical model

The model has not been clinically validated and should not be used for diagnosis or medical decision-making.

### 2. Survey data

NHANES is a complex survey dataset with sampling weights, strata, and primary sampling units. The initial machine learning workflow does not incorporate the full complex survey design into model training.

### 3. Model performance

The ROC-AUC of approximately `0.677` indicates moderate discrimination rather than clinical-grade predictive performance.

### 4. Threshold-dependent results

The classification result depends on the selected threshold of `0.29`.

Changing the threshold changes the balance between recall, precision, sensitivity, and specificity.

### 5. Missing laboratory measurements

Some laboratory variables have substantial missingness. The machine learning pipeline handles missing values through imputation.

---

##  Future Improvements

Possible future extensions include:

* Incorporating NHANES survey weights into model development
* Calibration analysis
* Precision-Recall curves
* ROC curves
* Feature importance visualization
* SHAP-based interpretability
* Model comparison dashboard
* More advanced models
* External validation
* Improved handling of missing laboratory measurements
* Automated data preprocessing pipeline
* Streamlit deployment

---

##  Author

**Pedram**

Computer Engineering Student
Interested in Data Analysis, Machine Learning, and Applied AI.

---

##  License

This project is intended for educational and portfolio purposes.
