# QuantGroup7

MSc Group Assignment for **Quantitative Methods, Coding and AI in Finance**.

Group Members: Aakash Arul, Maeve Gaffney, Anna Kurkina, Mykyta Nedzelskiy, Aryaa Sushil Punyarthi

## Overview

This repository contains Python code for portfolio analysis and optimisation using the included Excel dataset (`Dataset_2026.xlsx`). It covers:

- Data cleaning and return/covariance estimation
- Monte Carlo portfolio construction under constraints (e.g., diversification across asset classes)
- Backtesting and portfolio comparison (equal-weight vs. dynamic weighting)
- Statistical diagnostics such as ADF stationarity tests and Engle–Granger cointegration
- A simple Streamlit UI to run the analysis interactively

## Repository contents

- `Dataset_2026.xlsx` — input dataset (Excel).
- `1ASuccess.py` — asset selection & portfolio optimisation workflow (loads the dataset, cleans it, defines asset classes by column slices, and runs portfolio analytics/Monte Carlo search).
- `1B1C1DSuccess.py` — portfolio construction / rebalancing and comparison logic using a list of “top assets” from a frequency CSV (equal-weight vs. momentum/volatility “dual dynamic” weights; rebalanced monthly).
- `q2Success.py` — additional quantitative diagnostics on selected assets (ADF stationarity test outputs `adf_results.csv`; Engle–Granger cointegration tests across asset pairs).
- `UITrial.py` — Streamlit app that lets you upload an Excel file and run the analysis in a browser, producing figures/CSVs for download.

## How to run

### 1) Create an environment

Install Python dependencies (examples):

- `pandas`, `numpy`, `matplotlib`, `seaborn`
- `statsmodels`
- `streamlit` (for the UI)

### 2) Data file location

Some scripts currently reference **local Windows paths** (e.g. `C:\Users\maeve\Downloads\Dataset_2026.xlsx` and a local `asset_frequency_top_5_pct.csv`). For portability, you can:

1. Change the path variables (`file_path`, `price_file`, `freq_file`) to point to your local copies, or
2. Refactor the scripts to use the repository file `Dataset_2026.xlsx` (recommended) and to accept paths via CLI args.

### 3) Run scripts

From the repository root:

- Run the analysis scripts with Python, e.g. `python 1ASuccess.py`
- For the UI: `streamlit run UITrial.py`

## Notes / assumptions

- 0% risk-free rate (as used in Sharpe ratio calculations).
- 252 trading days per year.
- Asset classes are inferred by **column position** (hard-coded slices). If the dataset column ordering changes, update the slicing logic accordingly.

---
Author(s): QuantGroup7