# EB-2 PERM Approval Probability Model (FY2024)

This repository contains an end-to-end machine learning solution that predicts the **probability that an EB-2 PERM labor certification case will be approved** by the U.S. Department of Labor.  
The model is trained using **FY2024 DOL PERM Disclosure Data** and deployed as an interactive **Streamlit web application**.

---

## 🚀 Business Problem

U.S. immigration law firms must quickly assess the likelihood that an EB-2 PERM case will be approved.  
Approval outcomes depend on many factors:

- Occupational classification (SOC)
- Employer industry (NAICS)
- Prevailing wage vs. offered wage
- Education requirements
- Worksite state
- Foreign worker ownership interest

This evaluation is typically manual, subjective, and time-consuming.

### **Objective**  
Build a data-driven model that provides an **approval probability (%)** to support **risk assessment, case triage, and client guidance**.

---

## 🧠 Analytical Framing

This problem is framed as a **binary classification task**:

- **1 = Certified (Approved)**
- **0 = Denied / Withdrawn / Other**

The model outputs a calibrated **approval probability**, not just a yes/no label.  
This probability supports nuanced decision-making for attorneys and clients.

---

## 🛠️ Technical Approach

### **1. Data Source**
FY2024 PERM Disclosure Data from the U.S. Department of Labor (OFLC).  
Approximately **60,000 records** after cleaning.

### **2. Feature Engineering**
- Annualized prevailing wage  
- Annualized offered wage  
- Wage ratio (offered ÷ prevailing)  
- Education level  
- SOC code  
- NAICS code  
- Ownership interest  
- Worksite state  
- Fiscal year flag  

---

### **3. Models Implemented**

A range of baseline and advanced models were tested:

- **Logistic Regression** (best-performing model)
- Decision Tree  
- Gaussian Naive Bayes  
- Random Forest  
- Neural Network (MLP)  

### **Evaluation Metrics**
- **ROC-AUC**  
- **F1 Score**  
- **Precision**  
- **Recall**  
- **Accuracy**  

All experiments were tracked using **Databricks + MLflow**.

---

## 🎯 Best Model

**Logistic Regression** achieved the strongest and most stable performance:

- **ROC-AUC:** 0.810  
- **Accuracy:** 0.715  
- **F1 Score:** 0.670  
- **Recall:** 0.753  
- **Precision:** 0.603  

Chosen for its:
- Strong predictive power  
- Excellent calibration  
- Interpretability (critical for legal audiences)  
- Reliable performance in high-dimensional one-hot encoded data  

The final trained pipeline is saved as:
```text


## 🔗 Deployed Application

Live model available here:

👉 **

## 🔗 Deployed Application

Live model available here:

👉 **https://perm-eb2-approval-ml-pmerzmujwm8jfxgyands3v.streamlit.app/**

model_perm_best.pkl/**

model_perm_best.pkl
