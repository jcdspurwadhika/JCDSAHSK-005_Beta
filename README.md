# Bank Marketing – Term Deposit Subscription Prediction

## 1. Project Overview

Project ini bertujuan membangun model Machine Learning untuk memprediksi kemungkinan nasabah melakukan **subscription term deposit** berdasarkan karakteristik nasabah, riwayat campaign, dan kondisi ekonomi.

Dataset memiliki kondisi **class imbalance**, sehingga evaluasi model menggunakan **F2-Score sebagai primary metric** karena memberikan penekanan lebih besar terhadap Recall.

Project ini juga dilengkapi dengan analisis faktor yang berkontribusi terhadap prediksi, simulasi dampak finansial, aplikasi prediksi menggunakan Streamlit, dan dashboard analisis menggunakan Power BI.

---

## 2. Project Objectives

- Membangun model klasifikasi untuk memprediksi subscription term deposit.
- Menangani permasalahan class imbalance.
- Mengoptimalkan model dengan **F2-Score sebagai primary metric**.
- Mengidentifikasi faktor yang berkontribusi terhadap prediksi menggunakan SHAP.
- Memberikan insight untuk mendukung strategi marketing.
- Menyediakan aplikasi prediksi menggunakan Streamlit dan dashboard analisis menggunakan Power BI.

---

## 3. Final Model

Model final yang digunakan adalah:

**XGBoost – Balanced**

- Primary Metric: **F2-Score**
- Threshold: **0.50**
- Final Test F2-Score: **0.5766**
- Recall: **0.6573**
- Precision: **0.3866**
- ROC-AUC: **0.8135**
- PR-AUC: **0.4846**

Model dipilih berdasarkan hasil **5-Fold Cross-Validation** dan hyperparameter tuning menggunakan F2-Score sebagai primary metric.

---

## 4. Repository Structure

```text
Bank-Marketing-Project/
│
├── README.md
├── Final_Project - F2.ipynb
├── bank-additional.csv
├── termdeposit.py
├── bank_marketing_final_model.pkl
└── PowerBI/
    └── Dashboard Final Project Purwadhika - For Submit.pbix
