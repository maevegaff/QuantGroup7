# QuantGroup7

MSc Group Assignment for **Quantitative Methods, Coding and AI in Finance**.


This repository contains Python scripts for portfolio analytics and optimisation using a provided dataset (Excel), including return/covariance estimation, visual diagnostics, Monte Carlo portfolio search, and additional statistical tests (risk attribution + stationarity/cointegration).

## Repository contents

project2.py= Portfolio construction + Monte Carlo simulation to search across 5-asset portfolios under constraints, ranking by Sharpe ratio, and exporting plots/results.
project2q.py =Additional quantitative analysis including:
  - risk attribution (marginal contribution to risk / contribution to risk)
  - Dickey–Fuller stationarity test
  - Engle–Granger cointegration tests and heatmap


Both scripts currently load data from a **local Windows path**:

- `C:\Users\maeve\Downloads\Dataset_2026.xlsx`

To run on another machine, you’ll need to either:
1. Update `file_path` inside each script, or
2. Refactor to accept a relative path / CLI argument (recommended).

This Data Set is included in this repository and if this script is run with alternative data the format must be the same 

Asset classes are inferred by column position (slices), e.g. Equities, Bonds, Commodities, FX. If the dataset column ordering changes, you must update the asset-class slicing logic.


Assumptions 
0% riskfree rate
252 trading days per year 
Asset classses are hard coded 


---
Author(s): QuantGroup7
