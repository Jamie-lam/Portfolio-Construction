# 📘 Smart Beta Portfolio Construction with PCA Mimicking Portfolios

This repository demonstrates how to combine **Smart‑Beta portfolio optimisation** with **Principal Component Analysis (PCA)** to analyse factor‑mimicking portfolios built from a universe of equities. All optimisation logic is packaged in the companion library **`factors.ipynb`**.

---

## 📦 Library — `smartbeta.py`
The file below implements five weighting schemes (**EW, RP, DR, GMV, MSR**) using the covariance matrix and exposes helper methods to obtain weights and portfolio return series.

---

## 🔧 Workflow Overview

### 1️⃣ PCA Factor Extraction

### 2️⃣ PCA Regression

### 3️⃣ Mimicking Portfolio Projection

### 4️⃣ Smart‑Beta Weight Optimisation


---

## 📊 Weighting Schemes & Objectives
| Scheme | Objective (minimise unless noted) |
|--------|------------------------------------|
| **EW** | \( w_i = 1/N \) |
| **RP** | Var of risk contributions \( RC_i \) |
| **DR** | \( -\, \text{DR} = -\frac{\sum w_i\sigma_i}{\sqrt{w^T\Sigma w}} \) |
| **GMV**| Portfolio variance \( w^T\Sigma w \) |
| **MSR**| \( -\text{Sharpe} = -\frac{w^T\mu-r_f}{\sqrt{w^T\Sigma w}} \) |

Ledoit‑Wolf shrinkage is applied to \(\Sigma\) for robustness.

---

## 📈 Risk Metrics
- **CAGR**, **Annualised Return & Volatility**, **Sharpe Ratio**.


---

## 👤 Author
**Jamie Lam** – MSc Financial Engineering, EDHEC

---

## 📄 License
Academic & interview demonstration only.

