# 🏦 Customer Segmentation & Churn Pattern Analytics in European Banking
### Empirical Portfolio Intelligence, Supervisory Risk Diagnostics & Retention Architecture

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://share.streamlit.io)
[![Python 3.10+](https://img.shields.io/badge/Python-3.10%2B-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

An end-to-end institutional financial analytics and machine learning solution for diagnosing, analyzing, and mitigating customer churn across European retail banking portfolios (France, Germany, Spain).

---

## 📌 Project Overview

Customer churn represents one of the largest hidden costs in retail banking, triggering core deposit runoff, elevated customer acquisition costs, and degradation of prudential liquidity ratios under Basel III. 

This repository provides an institutional-grade analytics suite and interactive web dashboard built according to European Central Bank (ECB) retail banking analytical standards.

### Key Insights Uncovered:
- **Overall Portfolio Churn**: **20.37%** (2,037 exited out of 10,000 accounts).
- **The German Retail Paradox**: German branches suffer a **32.44% churn rate** (nearly double France at 16.15% and Spain at 16.67%), representing **€168.6M (90.8%) of all churned capital**.
- **The Pre-Retirement Vulnerability Zone**: Customers aged 46–60 experience an acute churn spike of **51.12%**, accounting for over 41% of all defections.
- **The Multi-Product Retention Paradox**: Holding two products constitutes the optimal retention anchor (**7.58% churn**), whereas holding 3 or 4 products results in catastrophic defection (**82.71%** and **100.0%**).
- **High-Value Capital at Risk**: Total churned deposit balances equal **€185,588,095**, with **€170.7M (92.0%)** concentrated among high-value accounts ($\ge$ €100k).

---

## 🏛️ Application Architecture & Core Modules

The interactive Streamlit application (`app.py`) provides five comprehensive modules:

1. **📊 Executive Overview & KPI Dashboard**:
   - 5 Dynamic KPI metric cards (Overall Churn Rate, Active Customer Base, High-Value Churn Rate, Total Capital at Risk in Millions of Euros, Inactivity Churn Multiplier).
   - Portfolio retention donut chart, regional churn comparison with ECB benchmark line, and product distribution.
2. **🌍 Geographic Churn Patterns & The German Paradox**:
   - Cross-country comparative analysis (France, Germany, Spain).
   - In-depth deconstruction of why 100% of German customers hold active balances (averaging €119,730) and experience elevated churn.
   - Total capital lost by country in Euros.
3. **👥 Demographic & Tenure Cohort Patterns**:
   - Age vs Churn analysis visualizing the 46–60 pre-retirement peak.
   - Gender disparities: Female churn (25.1%) vs Male churn (16.5%).
   - Tenure cohort stability analysis across 10-year customer lifecycles.
4. **💎 High-Value Customer & Capital Risk Explorer**:
   - Balance vs Estimated Salary scatter plot with €100k high-value cutoff.
   - Multi-Product hazard breakdown.
   - Searchable, sortable drill-down table of at-risk high-value accounts with **CSV export capability**.
5. **🔮 Predictive Risk Scoring & Retention Simulator**:
   - Supervised Balanced Random Forest Classifier (**ROC-AUC: 0.875, Accuracy: 81.1%**).
   - Gini feature importance chart (Age: 38.6%, NumOfProducts: 26.6%, Germany: 6.9%, Inactivity: 6.7%, Balance: 6.6%).
   - Real-time interactive customer profile simulator providing instant churn risk probabilities, risk classifications (Low, Moderate, High, Critical), and tailored retention action playbooks.

---

## 📁 Repository Structure

```
├── data/
│   └── bank_churn_european.csv         # Verified 10,000-record European banking dataset
├── src/
│   ├── __init__.py
│   ├── data_loader.py                  # Ingestion, validation, transformation & segmentation
│   └── analytics_engine.py             # KPI computations, statistical modeling & ML classifier
├── app.py                              # Main interactive Streamlit Web Application
├── RESEARCH_PAPER.md                   # Full academic research paper
├── research_paper.html                 # Publication-grade, printable HTML research paper
├── EXECUTIVE_SUMMARY.md                # Policy brief for ECB and banking executives
├── requirements.txt                    # Project Python dependencies
├── setup_git.bat                       # Windows batch script for automated Git setup
├── setup_git.ps1                       # PowerShell script for automated Git setup
└── README.md                           # Project documentation
```

---

## 🚀 Quick Start Guide

### 1. Clone or Download Repository
```bash
git clone https://github.com/Ankita8174/european-banking-churn-analytics.git
cd european-banking-churn-analytics
```

### 2. Install Dependencies
```bash
pip install -r requirements.txt
```

### 3. Launch Streamlit Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501`.

---

## 🌐 Live Cloud Deployment Guide (Streamlit Community Cloud)

To deploy this application online for free:
1. Push this repository to your GitHub account: `https://github.com/Ankita8174/european-banking-churn-analytics`.
2. Visit [Streamlit Community Cloud](https://share.streamlit.io/) and sign in with GitHub.
3. Click **"New app"**.
4. Select:
   - **Repository**: `Ankita8174/european-banking-churn-analytics`
   - **Branch**: `main`
   - **Main file path**: `app.py`
5. Click **"Deploy!"** — Your app will be live with a shareable URL like:
   `https://ankita8174-european-banking-churn-analytics.streamlit.app`

---

## 📄 Key Deliverables & Documentation Links

- **Academic Research Paper**: [`RESEARCH_PAPER.md`](RESEARCH_PAPER.md) | Formatted Publication: [`research_paper.html`](research_paper.html)
- **Executive Policy Brief**: [`EXECUTIVE_SUMMARY.md`](EXECUTIVE_SUMMARY.md)
- **Interactive Web App**: [`app.py`](app.py)

---

## 📜 License & Citation

This project is licensed under the MIT License. Developed in conjunction with Unified Mentor and European Banking Analytics Research Frameworks.
