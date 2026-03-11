import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path #used for checking file paths, and for saving the output charts in the same directory as the script
#import warnings
#warnings.filterwarnings("ignore")

# FILE PATHS
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

# compute the dayily returns 
returns = selected_prices.pct_change().dropna()

print("Return matrix:", returns.shape)

print("start of Part 1B")

# Portfolio 1 -> Equal weighted with 5 selected assests 
equal_weight = 1 / top_n
eq_weights = pd.Series(equal_weight, index=top_assets)
eq_daily = (returns * eq_weights).sum(axis=1)

print("Equal weight per asset:", equal_weight)
# Portfolio 2 -> dual dynamic weighted 

rebalance_dates = returns.resample(rebalance_freq).last().index
weights_history = []

# Function provides an option to rebalance based on the window specification for rebalancing.
# If it is time for rebalancing, a dual dynamic approach is taken dividing the momentum by the volatility.
for date in rebalance_dates:
    hist = returns.loc[:date]
    if len(hist) < momentum_window:
        weights = pd.Series(equal_weight, index=top_assets)
    else:
        momentum = (1 + hist.tail(momentum_window)).prod() - 1
        vol = hist.tail(volatility_window).std() * np.sqrt(trading_days)
        vol = vol.replace(0, np.nan).fillna(vol.mean())
        score = momentum / vol
        score = score.clip(lower=0)
        if score.sum() == 0:
            weights = pd.Series(equal_weight, index=top_assets)
        else:
            weights = score / score.sum()
            for _ in range(10):  # iterate until stable
                weights = weights.clip(upper=0.5)
                weights = weights / weights.sum()
                # Ensure no asset has a weight higher than 50%
                if (weights <= 0.5 + 1e-9).all():
                    break
            # Re-normalize weights to ensure they sum to 1
            weights = weights / weights.sum()
    weights_history.append({"date": date, **weights.to_dict()})

weights_df = pd.DataFrame(weights_history).set_index("date")

# Provide a sample dynamic weight for debugging 
print("Sample dynamic weights:")
print(weights_df.head())

# Weights over time 

dyn_daily = pd.Series(index=returns.index, dtype=float)

for i, date in enumerate(rebalance_dates):
    start = date
    if i + 1 < len(rebalance_dates):
        end = rebalance_dates[i+1]
    else:
        end = returns.index[-1]
    mask = (returns.index >= start) & (returns.index <= end)
    w = weights_df.loc[date]
    dyn_daily.loc[mask] = (returns.loc[mask] * w).sum(axis=1)

dyn_daily = dyn_daily.dropna()

# Line up the equal and dynamic portfolios 

common_index = eq_daily.index.intersection(dyn_daily.index)

eq_daily = eq_daily.loc[common_index]
dyn_daily = dyn_daily.loc[common_index]

# Equal and Dynamic Portfolio Annual Returns and Annual Volatility 

ann_return_eq = eq_daily.mean() * trading_days
ann_return_dyn = dyn_daily.mean() * trading_days

ann_vol_eq = eq_daily.std() * np.sqrt(trading_days)
ann_vol_dyn = dyn_daily.std() * np.sqrt(trading_days)

# Rolling Volatility Calculation
rolling_vol_eq = eq_daily.rolling(volatility_window).std() * np.sqrt(trading_days)
rolling_vol_dyn = dyn_daily.rolling(volatility_window).std() * np.sqrt(trading_days)

performance = pd.DataFrame({
    "Portfolio": ["Equal Weight", "Dynamic"],
    "Annual Return": [ann_return_eq, ann_return_dyn],
    "Annual Volatility": [ann_vol_eq, ann_vol_dyn]
})

print("\nPerformance Summary")
print(performance)

# Display Rolling Volatility Summary
print("\nRolling Volatility (Last 5 Values):")
print("Equal Weight Rolling Volatility:")
print(rolling_vol_eq.tail())
print("Dynamic Rolling Volatility:")
print(rolling_vol_dyn.tail())

# Calculating and ploting cumulative returns 

eq_cum = (1 + eq_daily).cumprod()
dyn_cum = (1 + dyn_daily).cumprod()
plt.figure(figsize=(10,6))
plt.plot(eq_cum, label="Equal Weight")
plt.plot(dyn_cum, label="Dynamic")
plt.title("Cumulative Portfolio Returns")
plt.legend()
plt.savefig("cumulative_portfolio_returns.png")
plt.close()

# Plotting Rolling Volatility 

plt.figure(figsize=(10,6))
plt.plot(rolling_vol_eq, label="Equal Weight")
plt.plot(rolling_vol_dyn, label="Dynamic")
plt.title(f"Rolling Volatility ({volatility_window} Day Window)")
plt.legend()
plt.savefig("rolling_volatility_plot.png")
plt.close()

# Checking the last period of dynamic weighting 

print("\nLatest Dynamic Weights")

latest = weights_df.iloc[-1]

for asset, weight in latest.sort_values(ascending=False).items():
    print(f"{asset}: {weight:.2%}")


print("Start of 1C")

# Compute the Parametric VaR of portfolio for equal and dynamic portfolios 

# Confidence levels
confidence_levels = [0.05, 0.01]  # 5% (95% CI) and 1% (99% CI)

# Initialize a list to store VaR results
var_results = []

# Calculate VaR for each confidence level
for confidence_level in confidence_levels:
    z_score = -np.percentile(np.random.normal(0, 1, 100000), confidence_level * 100)
    
    # Equal Weighted Portfolio
    eq_portfolio_std = eq_daily.std(ddof=0)
    parametric_var_eq = z_score * eq_portfolio_std
    
    # Dynamic Weighted Portfolio
    dyn_portfolio_std = dyn_daily.std()
    parametric_var_dyn = z_score * dyn_portfolio_std
    
    # Append results to the list
    var_results.append({
        "Confidence Level": f"{100 - confidence_level*100:.0f}%",
        "Equal Weighted VaR": parametric_var_eq,
        "Dynamic Weighted VaR": parametric_var_dyn
    })

# Convert results to a DataFrame
var_table = pd.DataFrame(var_results)

# Print the VaR table
print("\nValue at Risk (VaR) Table:")
print(var_table)


print ("Start of 1D")

#calculate MVar for each asset in the equal portfolio 
eq_var = eq_daily.var()
print("\nMarginal VaR for Equal Weighted Portfolio:")
 
mvar_results = []
for asset in top_assets:
    cov = returns[asset].cov(eq_daily)
    mvar = cov / eq_daily.std()
    print(f"{asset}: {mvar:.6f}")
    mvar_results.append({"Asset": asset, "Marginal VaR": mvar})

#save the mVar results to a csv file
mvar_df = pd.DataFrame(mvar_results)
mvar_df.to_csv("marginal_var_results.csv", index=False)

#calculate MVar for each asset in the dynamic portfolio
print("\nMarginal VaR for Dynamic Weighted Portfolio:")
dyn_var = dyn_daily.var()
mvar_dyn_results = []
for asset in top_assets:
    cov = returns[asset].cov(dyn_daily)
    mvar = cov / dyn_daily.std()
    print(f"{asset}: {mvar:.6f}")
    mvar_dyn_results.append({"Asset": asset, "Marginal VaR": mvar})

#save the mVar results to a csv file
mvar_dyn_df = pd.DataFrame(mvar_dyn_results)
mvar_dyn_df.to_csv("marginal_var_dynamic_results.csv", index=False)

#Calculate the CVar for the equal weighted portfolio for each asset
print("\nConditional VaR (CVaR) for Equal Weighted Portfolio:")
cvar_results = []
for asset in top_assets:
    cov = returns[asset].cov(eq_daily)
    cvar = cov / eq_daily.mean()  # Using mean return for CVaR calculation
    print(f"{asset}: {cvar:.6f}")
    cvar_results.append({"Asset": asset, "Conditional VaR": cvar})

#save the cVar results to a csv file
cvar_df = pd.DataFrame(cvar_results)
cvar_df.to_csv("conditional_var_results.csv", index=False)

#Calculate the CVar for the dynamic weighted portfolio for each asset
print("\nConditional VaR (CVaR) for Dynamic Weighted Portfolio:")
cvar_dyn_results = []
for asset in top_assets:
    cov = returns[asset].cov(dyn_daily)
    cvar = cov / dyn_daily.mean()  # Using mean return for CVaR calculation
    print(f"{asset}: {cvar:.6f}")
    cvar_dyn_results.append({"Asset": asset, "Conditional VaR": cvar})

#save the cVar results to a csv file
cvar_dyn_df = pd.DataFrame(cvar_dyn_results)
cvar_dyn_df.to_csv("conditional_var_dynamic_results.csv", index=False)


# Conduct a Var decomposition for the equal weighted portfolio 
print("\nVaR Decomposition for Equal Weighted Portfolio:")
var_decomposition_results = []
for asset in top_assets:
    cov = returns[asset].cov(eq_daily)
    mvar = cov / eq_daily.std()
    contribution = mvar * eq_weights[asset]
    print(f"{asset}: Contribution to VaR = {contribution:.6f}")
    var_decomposition_results.append({"Asset": asset, "Contribution to VaR": contribution})

#save the var decomposition results to a csv file
var_decomposition_df = pd.DataFrame(var_decomposition_results)
var_decomposition_df.to_csv("var_decomposition_equal_results.csv", index=False)


# Create a Decomposition table for the equal weighted portfolio
decomposition_results = []

for asset in top_assets:
    # Current weight
    weight = eq_weights[asset]
    
    # Individual VaR
    individual_var = returns[asset].std() * -np.percentile(np.random.normal(0, 1, 100000), 5)
    
    # Marginal VaR
    cov = returns[asset].cov(eq_daily)
    mvar = cov / eq_daily.std()
    
    # Conditional VaR
    cvar = cov / eq_daily.mean()
    
    # CVaR percentage
    cvar_pct = cvar / eq_daily.mean() * 100
    
    decomposition_results.append({
        "Asset": asset,
        "Weight": weight,
        "Individual VaR": individual_var,
        "Marginal VaR": mvar,
        "Conditional VaR": cvar,
        "CVaR%": cvar_pct
    })

# Convert results to a DataFrame
decomposition_df = pd.DataFrame(decomposition_results)

# Save the decomposition table to a CSV file
decomposition_df.to_csv("equal_weight_decomposition_table.csv", index=False)

# Print the decomposition table
print("\nEqual Weighted Portfolio Decomposition Table:")
print(decomposition_df)


# Conduct a Var decomposition for the dynamic weighted portfolio
print("\nVaR Decomposition for Dynamic Weighted Portfolio:")
var_decomposition_dyn_results = []
for asset in top_assets:
    cov = returns[asset].cov(dyn_daily)
    mvar = cov / dyn_daily.std()
    contribution = mvar * weights_df.iloc[-1][asset]  # Using the latest dynamic weight
    print(f"{asset}: Contribution to VaR = {contribution:.6f}")
    var_decomposition_dyn_results.append({"Asset": asset, "Contribution to VaR": contribution})

#save the var decomposition results to a csv file
var_decomposition_dyn_df = pd.DataFrame(var_decomposition_dyn_results)
var_decomposition_dyn_df.to_csv("var_decomposition_dynamic_results.csv", index=False)

# Create a Decomposition table for the dynamic weighted portfolio
decomposition_dyn_results = []

for asset in top_assets:
    # Current weight
    weight = weights_df.iloc[-1][asset]  # Using the latest dynamic weight
    
    # Individual VaR
    individual_var = returns[asset].std() * -np.percentile(np.random.normal(0, 1, 100000), 5)
    
    # Marginal VaR
    cov = returns[asset].cov(dyn_daily)
    mvar = cov / dyn_daily.std()
    
    # Conditional VaR
    cvar = cov / dyn_daily.mean()
    
    # CVaR percentage
    cvar_pct = cvar / dyn_daily.mean() * 100
    
    decomposition_dyn_results.append({
        "Asset": asset,
        "Weight": weight,
        "Individual VaR": individual_var,
        "Marginal VaR": mvar,
        "Conditional VaR": cvar,
        "CVaR%": cvar_pct
    })

# Convert results to a DataFrame
decomposition_dyn_df = pd.DataFrame(decomposition_dyn_results)

# Save the decomposition table to a CSV file
decomposition_dyn_df.to_csv("dynamic_weight_decomposition_table.csv", index=False)

# Print the decomposition table
print("\nDynamic Weighted Portfolio Decomposition Table:")
print(decomposition_dyn_df)



