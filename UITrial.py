#Note that this User Interface is a summary for the full output file avaibliy including the Var decomposition, please run the scripts individually 

import streamlit as st #used to create the uset interface for the project, so that a user my add inputs on a local host
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns #used to create the heatmap matrix
from itertools import combinations #used to run the monte carlo simulation
from statsmodels.tsa.stattools import adfuller, coint #used for stationary and cointegration tests
import io #used to create in memory files for download
import zipfile #used to create a zip file to download the results on the user interface

#titles the diplay of the Interface
st.set_page_config(page_title="Portfolio Analyser", layout="centered")
st.title("Portfolio Analyser")

# Outline declaring the inputs that will be displayed on the sidebar 
with st.sidebar:
    st.header("Settings")
    uploaded_file     = st.file_uploader("Upload Price Data (.xlsx)", type="xlsx")
    top_n             = st.slider("Top N Assets", 3, 10, 5)
    num_simulations   = st.slider("Monte Carlo Simulations", 10, 500, 100, step=10)
    momentum_window   = st.slider("Momentum Window (days)", 20, 252, 63)
    volatility_window = st.slider("Volatility Window (days)", 20, 252, 63)
    significance      = st.number_input("Cointegration Significance", value=0.01, step=0.001, format="%.3f")

TRADING_DAYS  = 252 #declare number of trading days 
outputs = {}  # stores filename witch will be used for downloads 

#creates functions to save figures and csv files (using io library)
def save_fig(fig):
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=150, bbox_inches="tight")
    buf.seek(0)
    return buf.read()

def save_csv(df):
    return df.to_csv(index=False).encode()

#Message for user to upload the desire excel file for analysis 
if uploaded_file is None:
    st.info("Upload your Excel file in the sidebar to begin.")
    st.stop()

# Load & clean prices, sets the date as index, drops NAN and interpolates, similar to scripts 
#Prints success message for debugging
prices = pd.read_excel(uploaded_file, sheet_name="Data")
prices["Date"] = pd.to_datetime(prices["Date"])
prices = prices.set_index("Date").sort_index().dropna(how="all").interpolate()
st.success(f"Loaded {prices.shape[0]} rows × {prices.shape[1]} assets")

if st.button(" Run Full Analysis"):

   
    # 1A — Monte Carlo Asset Selection
    
    st.header("1A — Asset Selection")
    # Define asset classes based on column positions from the dataframe
    asset_classes = {
        "Equities":    prices.columns[1:6],
        "Bonds":       prices.columns[6:11],
        "Commodities": prices.columns[11:16],
        "FX":          prices.columns[16:21]
    }
#Ensure that asset classes are present and follow there are no more then 3 assets per class when creating a portfolio
    def get_class(asset):
        for cls, assets in asset_classes.items():
            if asset in assets: return cls
        return "Unknown"

    def valid_combo(combo):
        counts = {}
        for a in combo:
            c = get_class(a)
            counts[c] = counts.get(c, 0) + 1
        return all(v <= 3 for v in counts.values())

    #calculating returns using simple returns
    returns       = prices.pct_change().dropna()
    ann_returns   = returns.mean() * TRADING_DAYS
    ann_cov       = returns.cov() * TRADING_DAYS

    # Correlation heatmap
    fig, ax = plt.subplots(figsize=(12, 10))
    sns.heatmap(returns.corr(), annot=True, cmap="coolwarm", vmin=-1, vmax=1, ax=ax)
    ax.set_title("Correlation Matrix")
    plt.tight_layout()
    outputs["correlation_matrix.png"] = save_fig(fig)
    st.pyplot(fig); plt.close(fig)

    # Monte Carlo

    #Search for valid combinations 
    valid_combos = [c for c in combinations(prices.columns, top_n) if valid_combo(c)]
    st.write(f"Running {num_simulations} simulations across {len(valid_combos)} valid combinations...")
    bar = st.progress(0)
#save to results, run through combination (similar to unwrapped scripts)
    results = []
    for i, combo in enumerate(valid_combos):
        bar.progress((i + 1) / len(valid_combos))
        combo = list(combo)
        best = {"sharpe": -np.inf, "weights": None, "return": 0, "risk": 0}
        for _ in range(num_simulations):
            w = np.random.dirichlet(np.ones(top_n))
            if any(w > 0.5): continue
            r = np.dot(w, ann_returns[combo])
            v = np.sqrt(w @ ann_cov.loc[combo, combo].values @ w)
            s = r / v if v > 0 else -np.inf
            if s > best["sharpe"]:
                best = {"sharpe": s, "weights": w, "return": r, "risk": v}
        if best["weights"] is not None:
            results.append({"Assets": combo, "Weights": best["weights"],
                            "Annual_Return": best["return"], "Annual_Risk": best["risk"],
                            "Sharpe_Ratio": best["sharpe"]})

    results_df = pd.DataFrame(results)
    top10 = results_df.nlargest(10, "Sharpe_Ratio")

    # Asset frequency in top 5%
    top5pct = results_df.nlargest(int(len(results_df) * 0.05), "Sharpe_Ratio")
    freq = {}
    for assets in top5pct["Assets"]:
        for a in assets: freq[a] = freq.get(a, 0) + 1
    freq_df = pd.DataFrame(
        [{"Asset": a, "Frequency": c, "Asset_Class": get_class(a)}
         for a, c in sorted(freq.items(), key=lambda x: x[1], reverse=True)]
    )
    outputs["asset_frequency_top_5_pct.csv"] = save_csv(freq_df)
    st.subheader("Top Assets by Frequency")
    st.dataframe(freq_df)

    # Efficient frontier
    fig, ax = plt.subplots(figsize=(10, 6))
    sc = ax.scatter(results_df["Annual_Risk"], results_df["Annual_Return"],
                    c=results_df["Sharpe_Ratio"], cmap="viridis", alpha=0.6)
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    ax.set(xlabel="Risk", ylabel="Return", title="Efficient Frontier")
    plt.tight_layout()
    outputs["efficient_frontier.png"] = save_fig(fig)
    st.pyplot(fig); plt.close(fig)

    # Top 10 CSV
    top10_csv = top10.copy()
    top10_csv["Assets"]  = top10_csv["Assets"].apply(lambda x: ", ".join(x))
    top10_csv["Weights"] = top10_csv["Weights"].apply(lambda w: ", ".join(f"{x:.4f}" for x in w))
    outputs["top_10_portfolios.csv"] = save_csv(top10_csv)
    st.subheader("Top 10 Portfolios")
    st.dataframe(top10_csv[["Assets", "Annual_Return", "Annual_Risk", "Sharpe_Ratio"]])


    # 1B/C/D — Portfolio Construction & Risk
   
    st.header("1B/C/D — Portfolio Construction & Risk")

    top_assets      = freq_df.iloc[:top_n, 0].tolist()
    sel_prices      = prices[top_assets].ffill().dropna()
    sel_returns     = sel_prices.pct_change().dropna()
    eq_w            = 1 / top_n
    eq_daily        = (sel_returns * eq_w).sum(axis=1)

    # Dynamic portfolio
    rebal_dates     = sel_returns.resample("ME").last().index
    weights_history = []
    for date in rebal_dates:
        hist = sel_returns.loc[:date]
        if len(hist) < momentum_window:
            w = pd.Series(eq_w, index=top_assets)
        else:
            mom   = (1 + hist.tail(momentum_window)).prod() - 1
            vol   = hist.tail(volatility_window).std() * np.sqrt(TRADING_DAYS)
            vol   = vol.replace(0, np.nan).fillna(vol.mean())
            score = (mom / vol).clip(lower=0)
            w     = pd.Series(eq_w, index=top_assets) if score.sum() == 0 else score / score.sum()
            for _ in range(10):
                w = (w.clip(upper=0.5)); w = w / w.sum()
                if (w <= 0.5 + 1e-9).all(): break
        weights_history.append({"date": date, **w.to_dict()})

    weights_df = pd.DataFrame(weights_history).set_index("date")

    dyn_daily = pd.Series(index=sel_returns.index, dtype=float)
    for i, date in enumerate(rebal_dates):
        end  = rebal_dates[i + 1] if i + 1 < len(rebal_dates) else sel_returns.index[-1]
        mask = (sel_returns.index >= date) & (sel_returns.index <= end)
        dyn_daily.loc[mask] = (sel_returns.loc[mask] * weights_df.loc[date]).sum(axis=1)
    dyn_daily = dyn_daily.dropna()

    common      = eq_daily.index.intersection(dyn_daily.index)
    eq_daily    = eq_daily.loc[common]
    dyn_daily   = dyn_daily.loc[common]

    perf = pd.DataFrame({
        "Portfolio":         ["Equal Weight", "Dynamic"],
        "Annual Return":     [eq_daily.mean() * TRADING_DAYS,  dyn_daily.mean() * TRADING_DAYS],
        "Annual Volatility": [eq_daily.std()  * np.sqrt(TRADING_DAYS), dyn_daily.std() * np.sqrt(TRADING_DAYS)]
    })
    outputs["performance_summary.csv"] = save_csv(perf)
    st.subheader("Performance Summary")
    st.dataframe(perf)

    # Cumulative returns plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot((1 + eq_daily).cumprod(),  label="Equal Weight")
    ax.plot((1 + dyn_daily).cumprod(), label="Dynamic")
    ax.set(title="Cumulative Returns"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    outputs["cumulative_returns.png"] = save_fig(fig)
    st.pyplot(fig); plt.close(fig)

    # Rolling volatility plot
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(eq_daily.rolling(volatility_window).std()  * np.sqrt(TRADING_DAYS), label="Equal Weight")
    ax.plot(dyn_daily.rolling(volatility_window).std() * np.sqrt(TRADING_DAYS), label="Dynamic")
    ax.set(title=f"Rolling Volatility ({volatility_window}d)"); ax.legend(); ax.grid(alpha=0.3)
    plt.tight_layout()
    outputs["rolling_volatility.png"] = save_fig(fig)
    st.pyplot(fig); plt.close(fig)

    # VaR table
    var_rows = []
    for cl in [0.05, 0.01]:
        z = -np.percentile(np.random.normal(0, 1, 100000), cl * 100)
        var_rows.append({"Confidence": f"{100 - cl*100:.0f}%",
                         "Equal VaR":   z * eq_daily.std(),
                         "Dynamic VaR": z * dyn_daily.std()})
    var_df = pd.DataFrame(var_rows)
    outputs["var_table.csv"] = save_csv(var_df)
    st.subheader("Value at Risk")
    st.dataframe(var_df)

   
    # Q2 — Cointegration Analysis
    
    st.header("Q2 — Cointegration Analysis")

    # ADF on price levels
    adf_rows = []
    for asset in top_assets:
        p = adfuller(sel_prices[asset])[1]
        adf_rows.append({"Asset": asset, "ADF p-value": round(p, 4),
                         "Non-stationary": "Yes" if p > 0.05 else "No"})
    adf_df = pd.DataFrame(adf_rows)
    outputs["adf_results.csv"] = save_csv(adf_df)
    st.subheader("ADF Test (price levels)")
    st.dataframe(adf_df)

    # Cointegration
    bonferroni = significance / len(list(combinations(top_assets, 2)))
    coint_rows = []
    coint_matrix = pd.DataFrame(index=top_assets, columns=top_assets, dtype=float)
    for a1, a2 in combinations(top_assets, 2):
        score, p, _ = coint(sel_prices[a1], sel_prices[a2], trend="ct")
        decision = "Cointegrated" if p < bonferroni else "Not Cointegrated"
        coint_rows.append({"Asset 1": a1, "Asset 2": a2, "p-value": round(p, 4), "Decision": decision})
        coint_matrix.loc[a1, a2] = p
        coint_matrix.loc[a2, a1] = p

    coint_df = pd.DataFrame(coint_rows)
    outputs["cointegration_summary.csv"] = save_csv(coint_df)
    st.subheader("Cointegration Results")
    st.dataframe(coint_df)

    # Heatmap
    fig, ax = plt.subplots(figsize=(8, 6))
    sns.heatmap(coint_matrix.astype(float), annot=True, cmap="coolwarm", vmin=0, vmax=1, ax=ax)
    ax.set_title("Cointegration Matrix (p-values)")
    plt.tight_layout()
    outputs["cointegration_matrix.png"] = save_fig(fig)
    st.pyplot(fig); plt.close(fig)

    
    # Download all outputs
    #Create a button that allows you to to download the files displayed
    #employs the zipfile library  

    st.header("Download Outputs")
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as zf:
        for name, data in outputs.items():
            zf.writestr(name, data)
    buf.seek(0)
    st.download_button("⬇️ Download All as ZIP", data=buf,
                       file_name="portfolio_analysis.zip", mime="application/zip")