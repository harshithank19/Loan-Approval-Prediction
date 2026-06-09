# 💳 Loan Approval Prediction – Mini Project 

A machine learning model that predicts loan approval outcomes based on applicant financial and personal data. Features an interactive 3-step Streamlit web dashboard to assess loan eligibility in real time.

![Python](https://img.shields.io/badge/Python-3.10+-blue?style=flat&logo=python)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat&logo=streamlit&logoColor=white)
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-F7931E?style=flat&logo=scikit-learn&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat&logo=pandas&logoColor=white)

---

## 📌 Overview

Traditional loan approval processes are manual, time-consuming, and prone to bias. This Mini project uses machine learning to automate the prediction of loan approvals based on key financial indicators like income, CIBIL score, assets, and loan amount — reducing human error and promoting fair decision-making.

<img width="1905" height="912" alt="Homepage" src="https://github.com/user-attachments/assets/90a0b314-a028-42a4-8d67-755e087c2d5e" />
<img width="1919" height="912" alt="Financial Profile" src="https://github.com/user-attachments/assets/a4b5d1d5-8187-431b-8579-9f5f51199065" />
<img width="1912" height="917" alt="Financial Profile (3)" src="https://github.com/user-attachments/assets/c5604ae1-ebbe-4930-af33-f99b280d90b5" />

---

## 🚀 Features

- 3-step interactive web dashboard (Introduction → Personal Details → Financial Details)
- Real-time loan approval prediction using a trained ML model
- CIBIL score-based eligibility assessment
- Personalized improvement suggestions on rejection
- Aadhaar and mobile number validation
- Loan type selection and document upload flow post-approval

---

## 🛠️ Tech Stack

| Layer | Technology |
|-------|-----------|
| ML Model | Scikit-learn (Classification) |
| Data Processing | Pandas, NumPy |
| Web App | Streamlit |
| Visualization | Matplotlib |
| Model Storage | Pickle (.pkl) |
| Dataset | loan_approval_dataset.csv |

---

## 📁 Repository Structure

```
loan-approval-prediction/
├── app.py                        ← Streamlit web application
├── Loan_Approval_Pred_Model.ipynb ← Jupyter notebook (model training)
├── loan_approval_dataset.csv      ← Dataset
├── model.pkl                      ← Trained ML model
├── scaler.pkl                     ← Feature scaler
├── Report/                        ← Project report
├── Screenshots/                   ← App screenshots
└── README.md
```

---

## ⚙️ How to Run

```bash
# Clone the repository
git clone https://github.com/harshithank19/loan-approval-prediction.git
cd loan-approval-prediction

# Install dependencies
pip install streamlit pandas scikit-learn matplotlib

# Run the app
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## 📊 Model Details

- **Algorithm:** Classification (trained on historical loan data)
- **Input Features:** Number of dependents, education, employment status, annual income, loan amount, loan duration, CIBIL score, assets
- **Output:** Approved ✅ / Rejected ❌
- **Preprocessing:** Standard scaling applied via `scaler.pkl`

---

## 🔍 Key Insights

- CIBIL score is the strongest predictor of loan approval
- Higher income-to-loan ratio improves approval chances
- Self-employed applicants face stricter evaluation
- Applicants with more dependents have lower approval rates

---

## 👤 Developed by 

**Harshitha N K**  - Team Lead    
**Shreya N** 

Computer Science, GAT 

[GitHub](https://github.com/harshithank19)

This project is developed for academic and research purposes, 2024.
