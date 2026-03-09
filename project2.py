###########AI Disclaimer and Code origins###########
# This code was generated with the assistance of the VS code copilot extension 
# This helped to speed up the coding process and provided suggestions for syntax and debugging tags to alert potential problems in the code
# Claude AI code assist was used to clean up the constraint functions and clarify the code in the Monte Carlo simulation section.
# GUI is modeled after a perviously developed project and can be found in the users GitHub repository 
# The code was written and debugged by the user, with the AI tools providing suggestions and assistance along the way. Python and VS code were used due to user firmilarity 

#gone through to line 151
#make the code messier by adding in some of his techniques 

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns
from itertools import combinations

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


# PHASE 2 - MONTE CARLO SIMULATION FOR PORTFOLIO OPTIMIZATION

####Function to define the asser class in preparion to assign portfolio constraints
####use of if statement to determine if the asset class can be found 
def get_asset_class(asset_name, asset_classes_dict):
    for asset_class, assets in asset_classes_dict.items():
        if asset_name in assets:
            return asset_class
    return "Unknown"

####Function to count the number of asset classes which will be used to meet the 3 asset class paramenter constraint
def check_asset_class_constraint(selected_assets, asset_classes_dict, max_per_class=3):

    class_count = {}
    for asset in selected_assets:
        asset_class = get_asset_class(asset, asset_classes_dict)
        class_count[asset_class] = class_count.get(asset_class, 0) + 1
    
    # Check if any class exceeds the maximum
    for count in class_count.values():
        if count > max_per_class:
            return False
    return True

#### Function to check the weights of each asset in the portfolio to check violations of the 50% weight constraint
def check_weight_constraint(weights, max_weight=0.5):

    return all(w <= max_weight for w in weights)


#### Create a list of all available assets
all_assets = list(prices_df.columns)
print(f"\nTotal assets available: {len(all_assets)}")#debugging
print(f"Assets: {all_assets}")


# Generate All Valid 5-Asset Portfolio Combinations
print("GENERATING VALID PORTFOLIO COMBINATIONS")

####creating a variable to store portfolios that meet the weight and asset number constraints 
####define the number of simulations to be run, 500 has been chosen as it is the largest sample size within a resonable run time
valid_portfolios = []
num_simulations = 100  # Number of random weight simulations per combination

# Determine the number of 5-asset combinations
all_combinations = list(combinations(all_assets, 5))
print(f"Total possible 5-asset combinations: {len(all_combinations)}")

# Using the previously defined variable filter combinations that meet the asset class constraint
valid_combinations = []
for combo in all_combinations:
    if check_asset_class_constraint(list(combo), asset_classes):
        valid_combinations.append(combo)

print(f"Valid combinations (max 3 per class): {len(valid_combinations)}")

# Run Monte Carlo Simulations on Valid Combinations
####rubberducking note there may be delays in this area due to the number of combinations 

for combo_idx, selected_assets in enumerate(valid_combinations):
    if combo_idx % 100 == 0:
        print(f"Processing combination {combo_idx} of {len(valid_combinations)}...")
    
    # Run multiple simulations with random weights for this combination
    best_sharpe = -np.inf
    best_weights = None
    best_return = 0
    best_risk = 0
    avg_correlation = 0
    
    for sim in range(num_simulations):
        # Generate random weights
        weights = np.random.random(5)
        weights /= np.sum(weights)  # Normalize so they sum to 1
        
        # Check weight constraint (no weight > 50%)
        if not check_weight_constraint(weights, max_weight=0.5):
            continue
        
        # Calculate portfolio metrics
        portfolio_return = np.dot(expected_annual_returns[list(selected_assets)].values, weights)
        portfolio_variance = np.dot(
            weights.T, 
            np.dot(annual_cov_matrix.loc[list(selected_assets), list(selected_assets)].values, weights)
        )
        portfolio_std_dev = np.sqrt(portfolio_variance)
        
        # Calculate Sharpe Ratio (assuming risk-free rate = 0%)
        sharpe_ratio = portfolio_return / portfolio_std_dev if portfolio_std_dev > 0 else 0
        
        # Keep track of the best weights for this combination
        if sharpe_ratio > best_sharpe:
            best_sharpe = sharpe_ratio
            best_weights = weights
            best_return = portfolio_return
            best_risk = portfolio_std_dev
    
    # Calculate average correlation for this portfolio
    subset_returns = returns_df[list(selected_assets)]
    corr_matrix = subset_returns.corr()
    avg_correlation = corr_matrix.where(~np.eye(5, dtype=bool)).stack().mean()
    
    # Store the results for this combination
    valid_portfolios.append({
        'Assets': selected_assets,
        'Weights': best_weights,
        'Annual_Return': best_return,
        'Annual_Risk': best_risk,
        'Sharpe_Ratio': best_sharpe,
        'Avg_Correlation': avg_correlation,
        'Max_Weight': np.max(best_weights) if best_weights is not None else 0
    })

print(f"\nTotal valid portfolios created: {len(valid_portfolios)}")


# Convert Results to DataFrame and Analyze


results_df = pd.DataFrame(valid_portfolios)


print("PORTFOLIO ANALYSIS RESULTS")


# Display summary statistics
print("\nPortfolio Performance Summary:")
print(results_df[['Annual_Return', 'Annual_Risk', 'Sharpe_Ratio', 'Max_Weight']].describe())

# Get top 10 portfolios by Sharpe Ratio
top_10 = results_df.nlargest(10, 'Sharpe_Ratio')
print("\n" + "="*70)
print("TOP 10 PORTFOLIOS BY SHARPE RATIO")
print("="*70)
for idx, row in top_10.iterrows():
    print(f"\n--- Portfolio {idx + 1} ---")
    print(f"Assets: {', '.join(row['Assets'])}")
    print(f"Weights: {[f'{w:.2%}' for w in row['Weights']]}")
    print(f"Annual Return: {row['Annual_Return']:.2%}")
    print(f"Annual Risk (Volatility): {row['Annual_Risk']:.2%}")
    print(f"Sharpe Ratio: {row['Sharpe_Ratio']:.4f}")
    print(f"Avg Correlation: {row['Avg_Correlation']:.4f}")
    print(f"Max Weight: {row['Max_Weight']:.2%}")


# Visualizations

# Plot 1: Scatter plot of Risk vs Return (Efficient Frontier)
plt.figure(figsize=(12, 8))
scatter = plt.scatter(results_df['Annual_Risk'], results_df['Annual_Return'], 
                     c=results_df['Sharpe_Ratio'], cmap='viridis', 
                     alpha=0.6, s=50, edgecolors='black', linewidth=0.5)
plt.colorbar(scatter, label='Sharpe Ratio')
plt.xlabel('Annual Risk (Volatility)', fontsize=12)
plt.ylabel('Annual Return', fontsize=12)
plt.title('Portfolio Efficient Frontier', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('efficient_frontier.png', dpi=300)
plt.close()
print("\nEfficient frontier plot saved as 'efficient_frontier.png'")

# Plot 2: Distribution of Sharpe Ratios
plt.figure(figsize=(10, 6))
plt.hist(results_df['Sharpe_Ratio'], bins=50, edgecolor='black', alpha=0.7)
plt.xlabel('Sharpe Ratio', fontsize=12)
plt.ylabel('Frequency', fontsize=12)
plt.title('Distribution of Sharpe Ratios Across All Portfolios', fontsize=14, fontweight='bold')
plt.grid(True, alpha=0.3, axis='y')
plt.tight_layout()
plt.savefig('sharpe_distribution.png', dpi=300)
plt.close()
print("Sharpe ratio distribution saved as 'sharpe_distribution.png'")

# Plot 3: Top 10 portfolios comparison
fig, axes = plt.subplots(2, 2, figsize=(14, 10))

# Return comparison
axes[0, 0].barh(range(len(top_10)), top_10['Annual_Return'], color='green', alpha=0.7)
axes[0, 0].set_yticks(range(len(top_10)))
axes[0, 0].set_yticklabels([f"P{i+1}" for i in range(len(top_10))])
axes[0, 0].set_xlabel('Annual Return')
axes[0, 0].set_title('Top 10 Portfolios - Return')
axes[0, 0].grid(True, alpha=0.3, axis='x')

# Risk comparison
axes[0, 1].barh(range(len(top_10)), top_10['Annual_Risk'], color='red', alpha=0.7)
axes[0, 1].set_yticks(range(len(top_10)))
axes[0, 1].set_yticklabels([f"P{i+1}" for i in range(len(top_10))])
axes[0, 1].set_xlabel('Annual Risk')
axes[0, 1].set_title('Top 10 Portfolios - Risk')
axes[0, 1].grid(True, alpha=0.3, axis='x')

# Sharpe Ratio comparison
axes[1, 0].barh(range(len(top_10)), top_10['Sharpe_Ratio'], color='blue', alpha=0.7)
axes[1, 0].set_yticks(range(len(top_10)))
axes[1, 0].set_yticklabels([f"P{i+1}" for i in range(len(top_10))])
axes[1, 0].set_xlabel('Sharpe Ratio')
axes[1, 0].set_title('Top 10 Portfolios - Sharpe Ratio')
axes[1, 0].grid(True, alpha=0.3, axis='x')

# Max Weight comparison
axes[1, 1].barh(range(len(top_10)), top_10['Max_Weight'], color='orange', alpha=0.7)
axes[1, 1].set_yticks(range(len(top_10)))
axes[1, 1].set_yticklabels([f"P{i+1}" for i in range(len(top_10))])
axes[1, 1].set_xlabel('Max Weight')
axes[1, 1].set_title('Top 10 Portfolios - Maximum Weight')
axes[1, 1].grid(True, alpha=0.3, axis='x')

plt.tight_layout()
plt.savefig('top_10_portfolios.png', dpi=300)
plt.close()
print("Top 10 portfolios comparison saved as 'top_10_portfolios.png'")


# Asset Class Analysis


# Analyze which assets appear most frequently in top portfolios
top_5_pct = results_df.nlargest(int(len(results_df) * 0.05), 'Sharpe_Ratio')

asset_frequency = {}
for assets in top_5_pct['Assets']:
    for asset in assets:
        asset_frequency[asset] = asset_frequency.get(asset, 0) + 1


print("ASSET FREQUENCY IN TOP 5% PORTFOLIOS")

for asset, count in sorted(asset_frequency.items(), key=lambda x: x[1], reverse=True):
    asset_class = get_asset_class(asset, asset_classes)
    print(f"{asset} ({asset_class}): {count} times")


# Save Results to CSV


# Create a simplified results DataFrame for export (without numpy arrays)
export_df = results_df.copy()
export_df['Weights_Str'] = export_df['Weights'].apply(lambda w: ', '.join([f'{x:.4f}' for x in w]))
export_df['Assets_Str'] = export_df['Assets'].apply(lambda x: ', '.join(x))
export_df = export_df[['Assets_Str', 'Weights_Str', 'Annual_Return', 'Annual_Risk', 
                       'Sharpe_Ratio', 'Avg_Correlation', 'Max_Weight']]
export_df.columns = ['Assets', 'Weights', 'Annual_Return', 'Annual_Risk', 
                     'Sharpe_Ratio', 'Avg_Correlation', 'Max_Weight']
export_df = export_df.sort_values('Sharpe_Ratio', ascending=False)
export_df.to_csv('portfolio_results.csv', index=False)
print("\nResults saved to 'portfolio_results.csv'")


print("ANALYSIS COMPLETE")
