# Portfolio-Construction

Smart Beta Portfolio Construction with PCA Mimicking Portfolios

This project explores Smart Beta weighting schemes combined with Principal Component Analysis (PCA) to understand factor exposures of equity portfolios. It includes portfolio optimization, PCA-based factor construction, and performance analysis.

📁 Files

factors.ipynb: Full notebook including PCA projection, regression analysis, portfolio optimization, and return decomposition.

HW2.xlsx: Contains raw equity and factor data.

scheme_returns.csv: Output of optimized portfolio returns under various weighting schemes.

hw2b_beta.csv, hw2b_beta2.csv: PCA factor loadings from regressions.

⚙️ Project Structure

Part A: PCA Factor Extraction

Perform PCA on standardized factor return data to construct five principal components (PCs).

from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

standardized_data = StandardScaler().fit_transform(factor_returns)
pca = PCA(n_components=5)
principal_components = pca.fit_transform(standardized_data)

Part B: PCA Regression

Regress standardized excess equity returns onto PCA scores to estimate factor betas.

import statsmodels.api as sm
model = sm.OLS(pcs_df, excess_returns).fit()
beta_values = model.params

Part C: Mimicking Portfolio Construction

Use the betas to project back onto equity space and estimate mimicking portfolio returns.

X = np.dot(excess_returns, beta_values)
model_m = sm.OLS(pcs_df, X).fit()

📊 Weighting Schemes (Smart Beta Models)

Scheme

Description

Objective Function

EW

Equal Weight



RP

Risk Parity

Minimize variance of risk contributions: 

DR

Diversification Ratio



GMV

Global Minimum Variance



MSR

Maximum Sharpe Ratio



Covariance matrix  is estimated using Ledoit-Wolf shrinkage for numerical stability.

📊 Risk Metrics Reported

CAGR: Compound Annual Growth Rate

Annualized Return & Volatility

Sharpe Ratio: Assuming risk-free rate 

All metrics are computed from the scheme-level return series.

📌 Usage

Install dependencies:

pip install pandas numpy matplotlib scikit-learn statsmodels openpyxl jupyter

Open factors.ipynb in Jupyter or VSCode.

Run the notebook to generate return tables and plots.

📈 Sample Output

PCA component loadings

Scheme vs. PCA exposure matrix

Cumulative return plots

Smart beta risk metrics

👤 Author

Jamie LamMSc Financial Engineering Candidate @ EDHEC

📄 License

This project is for academic use and interview demonstration purposes.
