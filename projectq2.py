import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

#Phase 1 - Data Preparation and Cleaning, and Preliminary Return Analysis and covariance calculations

# Load the Excel file
file_path = r"C:\Users\maeve\Downloads\Dataset_2026.xlsx"
print(f"File exists: {Path(file_path).exists()}")

# Read the data; print included for debugging
prices_df = pd.read_excel(file_path)
print("\nOriginal Data:")
print(prices_df.head())


# Clean the Data
####Date is set as the index so the column labels will be used to define asset classes
####Any Nan values are dropped and interpolated 

# Set Date as index if not already
if 'Date' in prices_df.columns:
    prices_df['Date'] = pd.to_datetime(prices_df['Date'])
    prices_df = prices_df.set_index('Date')

# Remove rows with missing values in price columns
prices_df = prices_df.dropna()
print(f"\nData shape after dropping NaN rows: {prices_df.shape}")

# Interpolate any remaining missing values (just in case)
prices_df = prices_df.interpolate(method='linear', limit_direction='forward', axis=0)

print("\nCleaned Data:")
print(prices_df.head())


# Define Asset Classes 
#### The four asset classes are defined as a list here to be use through out when referencing the assets included in the data file

asset_classes = {
    'Equities': prices_df.columns[1:6],      # Columns 1-5
    'Bonds': prices_df.columns[6:11],        # Columns 6-10
    'Commodities': prices_df.columns[11:16], # Columns 11-15
    'FX': prices_df.columns[16:21]           # Columns 16-20
}

print("\nAsset Classes Detected:")
for asset_class, assets in asset_classes.items():
    print(f"{asset_class}: {list(assets)}")


#Preliminary asset analysis

#### Calculate Daily Returns
returns_df = prices_df.pct_change().dropna()
print(f"\nDaily Returns DataFrame shape: {returns_df.shape}")
print(returns_df.head())

#### Calculate Annual Returns 
average_daily_returns = returns_df.mean()
expected_annual_returns = average_daily_returns * 252  # 252 trading days per year
print("\nExpected Annual Returns:")
print(expected_annual_returns)

#### Calculate Annual Covariance Matrix
annual_cov_matrix = returns_df.cov() * 252
print("\nAnnual Covariance Matrix:")
print(annual_cov_matrix)

#### A heat map of the Covariance Matrix is created as a visual and saved
correlation_matrix = returns_df.corr()
plt.figure(figsize=(12, 10))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, 
            square=True, cbar_kws={"shrink": 0.8})
plt.title('Correlation Matrix of Asset Returns')
plt.tight_layout()
plt.savefig('correlation_matrix.png', dpi=300)
plt.close()
print("\nCorrelation matrix saved as 'correlation_matrix.png'")


###Risk Attribution Analysis####
###Need to determine which weights of profolios we are running this on
###Currently run on equal weight but can be duplicated for other selections 
###Need to calculate the weight from sharpe max and pull forom equal weight function 


# Example weights for a portfolio (equal weight for simplicity)
num_assets = len(returns_df.columns)
weights = np.array([1/num_assets] * num_assets)

## Calculate portfolio volatility
portfolio_variance = weights.T @ annual_cov_matrix @ weights
portfolio_volatility = np.sqrt(portfolio_variance)
print(f"\nPortfolio Volatility: {portfolio_volatility:.4f}")


# Function to calclate MCTR for each asset class
def calculate_mctr(weights, cov_matrix):
    portfolio_variance = weights.T @ cov_matrix @ weights
    mctr = (cov_matrix @ weights) / np.sqrt(portfolio_variance)
    return mctr

#Calculate the Margibnal Contribution to Risk (MCTR)
mctr = calculate_mctr(weights, annual_cov_matrix)
print("\nMarginal Contribution to Risk (MCTR) for each asset:")
print(mctr)

#Compute the contribution to risk for each asset class
contribution_to_risk = mctr * weights
print("\nContribution to Risk for each asset:")
print(contribution_to_risk)

#Calculate the percentage contribution to risk for each asset class
percentage_contribution_to_risk = contribution_to_risk / np.sum(contribution_to_risk) * 100
print("\nPercentage Contribution to Risk for each asset:")
print(percentage_contribution_to_risk)


#Dickey-Fuller Test for Stationarity
#Using the price data for the first asset as an example
asset_to_test = returns_df.columns[0]  # Test the first asset
result = adfuller(returns_df[asset_to_test])
print(f"\nDickey-Fuller Test for {asset_to_test}:")
print(f"ADF Statistic: {result[0]}")
print(f"p-value: {result[1]}")

# Engle Granger Cointegration Test

# Function to perform Engle-Granger Cointegration Test
def engle_granger_test(asset1, asset2, data):
    score, p_value, _ = coint(data[asset1], data[asset2])
    return score, p_value

# Perform the test for all pairs of assets
cointegration_results = []
for asset1, asset2 in combinations(returns_df.columns, 2):
    score, p_value = engle_granger_test(asset1, asset2, returns_df)
    cointegration_results.append((asset1, asset2, score, p_value))

# Display results
print("\nEngle-Granger Cointegration Test Results:")
for asset1, asset2, score, p_value in cointegration_results:
    print(f"{asset1} and {asset2}: Score = {score:.4f}, p-value = {p_value:.4f}")


#Conintegration Matrix for all assets and display 
cointegration_matrix = pd.DataFrame(index=returns_df.columns, columns=returns_df.columns)
for asset1, asset2 in combinations(returns_df.columns, 2):
    score, p_value = engle_granger_test(asset1, asset2, returns_df)
    cointegration_matrix.loc[asset1, asset2] = p_value
    cointegration_matrix.loc[asset2, asset1] = p_value
plt.figure(figsize=(12, 10))
sns.heatmap(cointegration_matrix.astype(float), annot=True, cmap='coolwarm', vmin=0, vmax=1, 
            square=True, cbar_kws={"shrink": 0.8})
plt.title('Cointegration Matrix (p-values)')
plt.tight_layout()
plt.savefig('cointegration_matrix.png', dpi=300)
plt.close()
print("\nCointegration matrix saved as 'cointegration_matrix.png'")

