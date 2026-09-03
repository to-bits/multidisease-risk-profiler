# 🏥 Multi-Disease Risk Profiler

An end-to-end **Data Science + Machine Learning application** for estimating multiple health risks from clinical, demographic, and lifestyle information.

The goal is to build a practical ML system that combines **data analysis, predictive modeling, explainability, backend APIs, and an interactive frontend** into one unified project.

---

## 🎯 Project Goal

Multi-Disease Risk Profiler aims to analyze a person's health-related information and generate a unified risk profile for multiple conditions.

The planned prediction areas are:

- 🩸 **Diabetes**
- ❤️ **Hypertension**
- ⚖️ **Obesity**
- 💓 **Cardiovascular / Heart Disease**

The application will allow a user to enter relevant patient information through a web interface and receive ML-based risk predictions and interpretable insights.

---

## 🧠 Planned System

The project will follow an end-to-end ML application architecture:

```text
                    ┌─────────────────────┐
                    │   Frontend (React)  │
                    │                     │
                    │ Patient Information │
                    │ Risk Dashboard      │
                    │ Visualizations      │
                    └──────────┬──────────┘
                               │ REST API
                               ▼
                    ┌─────────────────────┐
                    │ Backend (Spring    │
                    │ Boot)               │
                    │                     │
                    │ API + Business Logic│
                    │ Validation          │
                    │ Prediction History  │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Python ML Service   │
                    │                     │
                    │ Preprocessing       │
                    │ Feature Engineering │
                    │ Model Inference     │
                    │ Explainability      │
                    └──────────┬──────────┘
                               │
                               ▼
                    ┌─────────────────────┐
                    │ Trained ML Models   │
                    │                     │
                    │ Diabetes            │
                    │ Hypertension        │
                    │ Obesity             │
                    │ Cardiovascular      │
                    └─────────────────────┘
```

The backend is intended to separate the web application from the Python ML layer and provide a clean API-based architecture.

---

## 📊 Dataset

The current dataset is:

```text
Data Warehouse Multiclass.csv
```

It contains **280,985 records and 39 columns** covering demographic, clinical, physiological, lifestyle, and derived health information.

Examples of available variables include:

- Age and gender
- BMI
- Smoking
- Diabetes
- Blood pressure
- Systolic and diastolic BP
- HbA1c
- Glucose
- Cholesterol
- HDL / LDL
- Triglycerides
- Sleep hours
- Physical activity
- Family history
- Stress level
- Alcohol and salt intake
- Heart rate
- CRP
- Homocysteine
- Education and employment information

The dataset also contains derived/label-related fields such as:

- `disease_flags`
- `sublabel`
- `label`
- `source_dataset`
- `composite_key`

These fields require careful investigation before being used for ML training because some may directly encode the prediction targets or introduce data leakage.

---

## ⚠️ Dataset Validation

The dataset is **promising for the project, but it should NOT be used directly for final model training yet**.

Initial inspection identified several important issues that must be handled during the data-preparation stage:

### 1. Possible target leakage

Columns such as:

```text
disease_flags
sublabel
label
source_dataset
```

may contain information derived from the disease outcomes.

They should therefore be investigated before being used as input features.

### 2. Highly duplicated / repeated records

The dataset contains many repeated rows, and `composite_key` has only a small number of unique values relative to the total number of records.

This suggests that duplication or synthetic/expanded records may exist.

Duplicate analysis and patient-level/group-level splitting will therefore be important.

### 3. Target definitions need to be established

The `diabetes` column is a clear categorical target (`Yes` / `No`).

However, `hypertension` and `heart_disease` currently contain many numeric values rather than simple binary labels. Their exact meaning and construction must be investigated before modeling.

### 4. Obesity target

`bmi_level` already contains categories such as:

```text
Underweight
Normal
Overweight
Obese
```

However, using `bmi_level` as the target while also using raw `bmi` as an input feature would create direct target leakage because BMI is used to define the category.

A clean target definition and feature policy must therefore be established.

---

## 🔬 Machine Learning Plan

The modeling workflow will be developed in stages.

### Phase 1 — Data Understanding

- Dataset profiling
- Missing-value analysis
- Duplicate analysis
- Cardinality analysis
- Distribution analysis
- Outlier analysis
- Target analysis
- Leakage detection

### Phase 2 — Exploratory Data Analysis

Investigate relationships between:

- Demographics and disease outcomes
- BMI and health risks
- Glucose / HbA1c and diabetes
- Blood pressure and hypertension
- Lipid indicators and cardiovascular risk
- Lifestyle variables and disease outcomes
- Family history and disease outcomes

### Phase 3 — Feature Engineering

Potential work includes:

- Encoding categorical variables
- Scaling numerical features where required
- Feature selection
- Clinically meaningful derived features
- Removing leakage-prone variables
- Handling class imbalance

### Phase 4 — Baseline Models

Candidate baseline models:

- Logistic Regression
- Decision Tree
- Random Forest

### Phase 5 — Advanced Models

Candidate advanced models:

- XGBoost
- LightGBM / other boosting methods if appropriate
- Neural-network models only if justified by the data

Model selection will be based on appropriate metrics rather than accuracy alone.

---

## 📈 Evaluation

Because disease prediction can involve class imbalance, the project will evaluate models using multiple metrics:

- Precision
- Recall
- F1-score
- ROC-AUC
- PR-AUC
- Confusion Matrix

For multiclass problems, macro/weighted metrics will also be considered where appropriate.

The final model for each prediction task will be selected based on performance, stability, interpretability, and practical usefulness.

---

## 🔍 Explainable AI

The project will aim to provide interpretable predictions rather than returning only:

```text
Risk = High
```

Possible explainability components include:

- Feature importance
- SHAP explanations
- Individual prediction explanations
- Risk-factor summaries

Example:

```text
Diabetes Risk: High

Important contributing factors:
• Elevated glucose
• Elevated HbA1c
• BMI category
• Family history
```

These explanations are intended to make the ML output easier to understand and demonstrate the practical value of the system.

---

## 🌐 Application

The final application is planned as a full-stack ML system.

### Frontend

Planned responsibilities:

- Patient information form
- Input validation
- Prediction request
- Risk cards
- Charts and visualizations
- Explainability section
- Overall health-risk profile

### Backend

Planned responsibilities:

- REST APIs
- Request validation
- Business logic
- Communication with the ML service
- Prediction history (if implemented)
- Database integration (if required)

### ML Service

Planned responsibilities:

- Load trained models
- Apply preprocessing
- Generate predictions
- Generate explainability information
- Return structured prediction results

---

## 🛠️ Technology Stack

### Data Science / ML

- Python
- Pandas
- NumPy
- Scikit-learn
- XGBoost
- Matplotlib
- Seaborn
- SHAP
- Jupyter Notebook

### Backend

- Java
- Spring Boot
- REST API

### ML API / Service

- Python
- FastAPI

### Frontend

- React
- HTML
- CSS
- JavaScript

### Optional Database

- PostgreSQL / MySQL

---

## 📁 Planned Project Structure

```text
multidisease-risk-profiler/
│
├── data/
│   ├── raw/
│   │   └── Data Warehouse Multiclass.csv
│   └── processed/
│
├── notebooks/
│   ├── 01_data_understanding.ipynb
│   ├── 02_eda.ipynb
│   ├── 03_preprocessing.ipynb
│   ├── 04_baseline_models.ipynb
│   ├── 05_advanced_models.ipynb
│   └── 06_model_evaluation.ipynb
│
├── ml-service/
│   ├── app/
│   ├── models/
│   ├── preprocessing/
│   └── requirements.txt
│
├── backend/
│   └── spring-boot/
│
├── frontend/
│   └── react-app/
│
├── reports/
│   ├── figures/
│   └── results/
│
├── README.md
└── .gitignore
```

---

## 🚀 Development Roadmap

### Stage 1 — Data Science

- [ ] Validate dataset
- [ ] Define the four prediction targets
- [ ] Detect and remove leakage
- [ ] Analyze duplicates
- [ ] Perform EDA
- [ ] Build preprocessing pipeline
- [ ] Train baseline models
- [ ] Train advanced models
- [ ] Evaluate and compare models
- [ ] Add explainability

### Stage 2 — ML Deployment

- [ ] Save production-ready models
- [ ] Save preprocessing pipelines
- [ ] Create Python prediction service
- [ ] Create prediction API
- [ ] Test inference independently

### Stage 3 — Backend

- [ ] Create Spring Boot project
- [ ] Design REST API
- [ ] Validate patient input
- [ ] Connect Spring Boot with ML service
- [ ] Add prediction-history functionality if needed

### Stage 4 — Frontend

- [ ] Design patient input form
- [ ] Build risk dashboard
- [ ] Display predictions
- [ ] Add charts
- [ ] Add explainability
- [ ] Connect frontend to backend

### Stage 5 — Finalization

- [ ] Integration testing
- [ ] API testing
- [ ] Model inference testing
- [ ] UI/UX improvement
- [ ] Documentation
- [ ] Final presentation/demo

---

## 📌 Important Project Principle

The objective is not simply to achieve the highest possible model accuracy.

The project aims to demonstrate a complete and responsible ML workflow:

```text
Data
 ↓
Understanding
 ↓
Cleaning
 ↓
EDA
 ↓
Feature Engineering
 ↓
Modeling
 ↓
Evaluation
 ↓
Explainability
 ↓
API
 ↓
Backend
 ↓
Frontend
 ↓
End-to-End ML Application
```

---

## ⚕️ Disclaimer

This application is an academic/research project and should not be used as a substitute for professional medical diagnosis, treatment, or clinical decision-making.

---

## 📄 License

This project is licensed under the MIT License.
