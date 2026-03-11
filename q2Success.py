####potentially add cvs files to both scripts for out put readability and to save the results of the calculations for future use. This would allow us to easily access and analyze the data without having to rerun the entire script each time. We can use pandas to export the DataFrames to CSV files after performing the necessary calculations.


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from itertools import combinations
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.stattools import coint

#This scripts aims to explore the given data, and optimise asset selection 
price_file= r"C:\Users\maeve\Downloads\Dataset_2026.xlsx"
freq_file = r"C:\Users\maeve\.vscode\asset_frequency_top_5_pct.csv"

#Time Frames and Parameters
top_n = 5
momentum_window= 63
volatility_window= 63
rebalance_freq= "ME"
trading_days= 252

# Load data
prices = pd.read_excel(price_file, sheet_name="Data")

# Set date index
prices["Date"] = pd.to_datetime(prices["Date"])
prices = prices.set_index("Date").sort_index()

# Cleaning the data in the same manner as the previous script for consistentcy 
# Remove rows where all prices missing 
prices = prices.dropna(how="all")
# Interpolate remaining gaps
prices = prices.interpolate(method="linear")

print("Price data shape:", prices.shape) #debugging 


# Define top 5 assets within the csv file
freq_df = pd.read_csv(freq_file)
top_assets = freq_df.iloc[:5, 0].tolist()

print("Selected assets:", top_assets)

# Check assets exist (debugging step)
missing = [a for a in top_assets if a not in prices.columns]
if missing:
    raise ValueError(f"Missing assets in dataset: {missing}")

# Subset prices- select prices associated for the asset 
selected_prices = prices[top_assets].ffill().dropna()

print("Selected price data shape:", selected_prices.shape)

# Compute the daily log returns
returns_df = np.log(selected_prices / selected_prices.shift(1)).dropna()
print("Return matrix:", returns_df.shape)


###Risk Attribution Analysis####
###Need to determine which weights of profolios we are running this on
###Currently run on equal weight but can be duplicated for other selections 
###Need to calculate the weight from sharpe max and pull forom equal weight function 



# ADF Stationarity Test on price series
print("\nADF Stationarity Test (Prices)")

adf_results = []

for asset in selected_prices.columns:
    result = adfuller(selected_prices[asset])
    p_value = result[1]
    adf_results.append((asset, p_value))
    print(f"{asset}: p-value = {p_value:.4f}")

adf_df = pd.DataFrame(adf_results, columns=["Asset", "ADF p-value"])
adf_df.to_csv("adf_results.csv", index=False)


# Engle Granger Cointegration Test

# Function to perform Engle-Granger Cointegration Test
def engle_granger_test(asset1, asset2, data):
    score, p_value, _ = coint(data[asset1], data[asset2], trend = 'c') #constant trends
    return score, p_value

# Perform the test for all pairs of assets
cointegration_results = []
for asset1, asset2 in combinations(selected_prices.columns, 2):
    score, p_value = engle_granger_test(asset1, asset2, selected_prices)
    cointegration_results.append((asset1, asset2, score, p_value))

# Convert to DataFrame
cointegration_df = pd.DataFrame(
    cointegration_results,
    columns=["Asset 1", "Asset 2", "Score", "p-value"]
)

cointegration_df.to_csv("cointegration_results.csv", index=False)

print("\nEngle-Granger Cointegration Results:")
print(cointegration_df)

#Conintegration Matrix for all assets and display 
cointegration_matrix = pd.DataFrame(index=returns_df.columns, columns=returns_df.columns)
for asset1, asset2 in combinations(returns_df.columns, 2):
    score, p_value = engle_granger_test(asset1, asset2, selected_prices)
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

# A summary table of the asset pairs, if the pairs are integrated, and a decision at a significance level of 0.01
summary_results = []
significance_level = 0.01

for asset1, asset2, score, p_value in cointegration_results:
    is_cointegrated = p_value < significance_level
    decision = "Cointegrated" if is_cointegrated else "Not Cointegrated"
    summary_results.append((asset1, asset2, score, p_value, decision))

# Create a DataFrame for the summary
summary_df = pd.DataFrame(summary_results, columns=['Asset 1', 'Asset 2', 'Score', 'p-value', 'Decision'])

# Save the summary table to a CSV file
summary_df.to_csv('cointegration_summary.csv', index=False)

# Display the summary table
print("\nSummary of Cointegration Results:")
print(summary_df)
