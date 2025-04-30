# 📘 Smart Beta Portfolio Construction with PCA Mimicking Portfolios

This repository demonstrates how to combine **Smart‑Beta portfolio optimisation** with **Principal Component Analysis (PCA)** to analyse factor‑mimicking portfolios built from a universe of equities. All optimisation logic is packaged in the companion library **`smartbeta_refined.py`** (included in the repo).

---

## 📦 Library — `smartbeta_refined.py`
The file below implements five weighting schemes (**EW, RP, DR, GMV, MSR**) using Ledoit‑Wolf shrinkage for the covariance matrix and exposes helper methods to obtain weights and portfolio return series.

```python
import pandas as pd
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.covariance import LedoitWolf
import statsmodels.api as sm
from scipy.optimize import minimize
from pathlib import Path

class SmartBeta:
    """Smart‑beta optimiser supporting EW, RP, DR, GMV, MSR."""

    def __init__(self, prices: pd.DataFrame, scheme: str,
                 rf: float = 0.0, lb: float = 0.01, ub: float = 0.10,
                 tol: float = 1e-10):
        self.prices = prices.dropna(how="any")
        self.scheme = scheme.upper()
        self.rf, self.lb, self.ub, self.tol = rf, lb, ub, tol
        self.returns = self.prices.pct_change().dropna()
        self.cov = LedoitWolf().fit(self.returns).covariance_
        self.mu = self.returns.mean().values
        self.n_assets = self.returns.shape[1]

    # --- objective functions --------------------------------------------
    def _risk_parity(self, w):
        vol = np.sqrt(w @ self.cov @ w)
        mrc = (self.cov @ w) / vol
        rc = w * mrc
        return np.var(rc)

    def _div_ratio(self, w):
        asset_vol = np.sqrt(np.diag(self.cov))
        dr = (w @ asset_vol) / np.sqrt(w @ self.cov @ w)
        return -dr

    def _min_var(self, w):
        return w @ self.cov @ w

    def _neg_sharpe(self, w):
        ret = self.mu @ w
        vol = np.sqrt(w @ self.cov @ w)
        return -(ret - self.rf) / vol

    # --- optimisation ----------------------------------------------------
    def optimise(self):
        if self.scheme == "EW":
            return np.repeat(1 / self.n_assets, self.n_assets)
        obj = {"RP": self._risk_parity, "DR": self._div_ratio,
               "GMV": self._min_var, "MSR": self._neg_sharpe}[self.scheme]
        x0 = np.repeat(1 / self.n_assets, self.n_assets)
        bounds = [(self.lb, self.ub)] * self.n_assets
        cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1},)
        res = minimize(obj, x0, method="SLSQP", bounds=bounds,
                       constraints=cons, tol=self.tol)
        if not res.success:
            raise RuntimeError(res.message)
        return res.x

    # --- public helpers --------------------------------------------------
    def weights(self):
        return pd.Series(self.optimise(), index=self.returns.columns, name=self.scheme)

    def portfolio_returns(self):
        w = self.optimise()
        return (self.returns @ w).rename(self.scheme)

# --- convenience ---------------------------------------------------------

def compute_scheme_returns(prices, schemes=("EW", "RP", "DR", "GMV", "MSR"), rf=0):
    return pd.concat([SmartBeta(prices, s, rf).portfolio_returns() for s in schemes], axis=1)

if __name__ == "__main__":
    fp = Path("./HW2.xlsx")
    equity_df = pd.read_excel(fp, sheet_name="equity", index_col=0, parse_dates=[0])
    factor_df = pd.read_excel(fp, sheet_name="factor", index_col=0, parse_dates=[0])

    # PCA factors (Part A)
    returns_f = factor_df.pct_change().dropna()
    pcs = PCA(n_components=5).fit_transform(StandardScaler().fit_transform(returns_f))
    pcs_df = pd.DataFrame(pcs, index=returns_f.index, columns=[f"PC{i+1}" for i in range(5)])

    # Betas (Part B)
    excess = equity_df.pct_change().dropna()
    model = sm.OLS(pcs_df, excess).fit()
    model.params.to_csv("hw2b_beta.csv")

    # Scheme returns (Part Q3)
    compute_scheme_returns(equity_df, rf=0).to_csv("scheme_returns.csv")
```

---

## 🔧 Workflow Overview

### 1️⃣ PCA Factor Extraction
```python
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(factor_returns)
pca = PCA(n_components=5)
pcs = pca.fit_transform(standardized_data)
```

### 2️⃣ PCA Regression
```python
import statsmodels.api as sm
model = sm.OLS(pcs_df, excess_returns).fit()
betas = model.params
```

### 3️⃣ Mimicking Portfolio Projection
```python
X = excess_returns @ betas
model_mimic = sm.OLS(pcs_df, X).fit()
```

### 4️⃣ Smart‑Beta Weight Optimisation
See **`smartbeta_refined.py`** above for objective functions and SLSQP optimisation.

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
- **CAGR**, **Annualised Return & Volatility**, **Sharpe Ratio** (\( r_f=0 \)) computed from `scheme_returns.csv`.

---

## 🚀 Quick Start
```bash
pip install -r requirements.txt      # includes jupyter, pandas, numpy, scikit-learn, statsmodels, openpyxl
python smartbeta_refined.py          # generates CSV outputs
jupyter notebook factors.ipynb       # optional interactive exploration
```

---

## 👤 Author
**Jamie Lam** – MSc Financial Engineering, EDHEC

---

## 📄 License
Academic & interview demonstration only.

