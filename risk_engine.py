import os
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t, kstest

def run_risk_engine():
    print("=== Quantitative Financial Risk Engine ===")
    print("Step 1: Fetching historical market data via yfinance...")
    
    # We will fetch 10 years of S&P 500 index data
    ticker = "^GSPC" # S&P 500 Index
    start_date = "2016-01-01"
    end_date = "2026-01-01"
    
    try:
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty or len(data) < 100:
            raise ValueError("Fetched data is empty or too short.")
        print(f"Successfully fetched {len(data)} trading days of S&P 500 data.")
    except Exception as e:
        print(f"Error fetching data for {ticker}: {e}")
        print("Attempting backup ticker: SPY...")
        ticker = "SPY"
        data = yf.download(ticker, start=start_date, end=end_date)
        if data.empty:
            print("Failed to download real data. Creating synthetic market returns for simulation...")
            # Create synthetic returns with fat tails (Student's t with df=3.5)
            np.random.seed(42)
            synthetic_dates = pd.date_range(start=start_date, end=end_date, freq='B')
            synthetic_returns = t.rvs(df=3.5, loc=0.0003, scale=0.01, size=len(synthetic_dates))
            data = pd.DataFrame(index=synthetic_dates)
            # Reconstruct prices from returns
            prices = 100 * np.exp(np.cumsum(synthetic_returns))
            data['Adj Close'] = prices
            print("Synthetic data generated.")

    # Step 2: Compute logarithmic returns
    # Daily log returns: R_t = ln(P_t / P_{t-1})
    close_col = None
    if isinstance(data.columns, pd.MultiIndex):
        lvl0 = data.columns.get_level_values(0)
        if 'Adj Close' in lvl0:
            close_col = 'Adj Close'
        elif 'Close' in lvl0:
            close_col = 'Close'
        
        if close_col is not None:
            adj_close = data[close_col].values.flatten()
        else:
            raise ValueError(f"Could not find closing price column in MultiIndex columns: {list(data.columns)}")
    else:
        if 'Adj Close' in data.columns:
            close_col = 'Adj Close'
        elif 'Close' in data.columns:
            close_col = 'Close'
            
        if close_col is not None:
            adj_close = data[close_col].values.flatten()
        else:
            raise ValueError(f"Could not find closing price column in columns: {list(data.columns)}")

    returns = np.log(adj_close[1:] / adj_close[:-1])
    
    # Filter out NaNs if any
    returns = returns[~np.isnan(returns)]
    
    n_samples = len(returns)
    print(f"Calculated {n_samples} daily log returns.")
    
    # Step 3: Compute Empirical VaR
    # VaR_c = -quantile(1 - c) of return distribution
    # For 95% confidence, it is the negative of the 5th percentile
    # For 99% confidence, it is the negative of the 1st percentile
    emp_var_95 = -np.percentile(returns, 5)
    emp_var_99 = -np.percentile(returns, 1)
    
    # Step 4: Fit Parametric Distributions via MLE
    # Normal distribution fit
    mu_norm, std_norm = norm.fit(returns)
    print(f"\nNormal Distribution Fit (MLE):")
    print(f"  Mean (mu): {mu_norm:.6f}")
    print(f"  Vol (sigma): {std_norm:.6f}")
    
    # Student's t-distribution fit
    df_t, loc_t, scale_t = t.fit(returns)
    print(f"\nStudent's t-Distribution Fit (MLE):")
    print(f"  Degrees of Freedom (nu): {df_t:.2f}")
    print(f"  Location (mu): {loc_t:.6f}")
    print(f"  Scale (s): {scale_t:.6f}")
    
    # Step 5: Compute Parametric VaR
    # Normal VaR
    norm_var_95 = -norm.ppf(0.05, loc=mu_norm, scale=std_norm)
    norm_var_99 = -norm.ppf(0.01, loc=mu_norm, scale=std_norm)
    
    # Student's t VaR
    t_var_95 = -t.ppf(0.05, df_t, loc=loc_t, scale=scale_t)
    t_var_99 = -t.ppf(0.01, df_t, loc=loc_t, scale=scale_t)
    
    # Step 6: Kolmogorov-Smirnov (KS) Goodness-of-Fit Test
    # KS distance D_n = sup |F_n(x) - F(x)|
    ks_stat_norm, p_val_norm = kstest(returns, 'norm', args=(mu_norm, std_norm))
    ks_stat_t, p_val_t = kstest(returns, 't', args=(df_t, loc_t, scale_t))
    
    print("\n" + "="*50)
    print(f"VALUE AT RISK (VaR) RESULTS COMPARISON (Daily loss as %)")
    print("="*50)
    print(f"Confidence Level  |  Empirical  |  Normal Fit  |  Student's t Fit")
    print("-"*50)
    print(f"95.0% VaR         |   {emp_var_95*100:6.3f}%   |   {norm_var_95*100:6.3f}%   |   {t_var_95*100:6.3f}%")
    print(f"99.0% VaR         |   {emp_var_99*100:6.3f}%   |   {norm_var_99*100:6.3f}%   |   {t_var_99*100:6.3f}%")
    print("="*50)
    
    print("\n" + "="*50)
    print("KOLMOGOROV-SMIRNOV (KS) GOODNESS-OF-FIT TEST")
    print("="*50)
    print(f"Normal Distribution Fit:")
    print(f"  KS Statistic (D_n): {ks_stat_norm:.5f}")
    print(f"  p-value           : {p_val_norm:.5e}")
    print(f"Student's t-Distribution Fit:")
    print(f"  KS Statistic (D_n): {ks_stat_t:.5f}")
    print(f"  p-value           : {p_val_t:.5e}")
    print("="*50)
    
    # Step 7: Generate Comparative CDF Plot with zoomed-in risk tail inset
    print("\nGenerating publication-quality CDF plot...")
    
    sorted_returns = np.sort(returns)
    ecdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
    
    # Generate points for plotting parametric curves
    x_vals = np.linspace(sorted_returns[0] - 0.01, sorted_returns[-1] + 0.01, 2000)
    cdf_norm = norm.cdf(x_vals, loc=mu_norm, scale=std_norm)
    cdf_t = t.cdf(x_vals, df_t, loc=loc_t, scale=scale_t)
    
    # Figure setup
    fig, ax = plt.subplots(figsize=(10, 6.5))
    
    # Main plot (Entire distribution CDF)
    ax.step(sorted_returns, ecdf, label='Empirical CDF ($F_n$)', color='#2c3e50', alpha=0.8, where='post', lw=1.5)
    ax.plot(x_vals, cdf_norm, label=f'Normal CDF ($F_{{Normal}}$, KS p-val={p_val_norm:.2e})', color='#e74c3c', lw=2, linestyle='--')
    ax.plot(x_vals, cdf_t, label=f"Student's t CDF ($F_t$, df={df_t:.1f}, p-val={p_val_t:.2e})", color='#2ecc71', lw=2)
    
    ax.set_title('Cumulative Distribution Function (CDF) Risk Model Comparison', fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Daily Return', fontsize=12)
    ax.set_ylabel('$F(x) = Pr(X \\leq x)$', fontsize=12)
    ax.grid(True, linestyle=':', alpha=0.6)
    ax.legend(loc='lower right', fontsize=10)
    
    # Customizing axes style
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    
    # Left Tail Inset Plot (Risk Zoom)
    # The inset is positioned where the space is empty (top-left)
    ax_inset = fig.add_axes([0.15, 0.45, 0.38, 0.38])
    
    # Find tail range to zoom in (e.g., returns between -6% and -1% and probabilities up to 10%)
    # Let's set the x limit of inset based on empirical 99% and 95% values
    inset_x_min = -max(emp_var_99 * 1.5, 0.04)
    inset_x_max = -0.005
    
    # Plot curves in the inset
    ax_inset.step(sorted_returns, ecdf, color='#2c3e50', where='post', lw=1.5)
    ax_inset.plot(x_vals, cdf_norm, color='#e74c3c', lw=2, linestyle='--')
    ax_inset.plot(x_vals, cdf_t, color='#2ecc71', lw=2)
    
    # Highlight VaR thresholds in the inset
    # 95% thresholds
    ax_inset.axvline(x=-emp_var_95, color='#2c3e50', linestyle=':', alpha=0.7)
    ax_inset.axvline(x=-norm_var_95, color='#e74c3c', linestyle=':', alpha=0.7)
    ax_inset.axvline(x=-t_var_95, color='#2ecc71', linestyle=':', alpha=0.7)
    
    # 99% thresholds
    ax_inset.axvline(x=-emp_var_99, color='#2c3e50', linestyle='-.', alpha=0.7)
    ax_inset.axvline(x=-norm_var_99, color='#e74c3c', linestyle='-.', alpha=0.7)
    ax_inset.axvline(x=-t_var_99, color='#2ecc71', linestyle='-.', alpha=0.7)
    
    # Labels and limits for inset
    ax_inset.set_xlim(inset_x_min, inset_x_max)
    ax_inset.set_ylim(0.0, 0.08) # focus on probabilities from 0% to 8% (tail risk)
    ax_inset.set_title('Left Tail Risk Zoom (Losses)', fontsize=10, fontweight='bold', pad=5)
    ax_inset.set_xlabel('Daily Return', fontsize=9)
    ax_inset.set_ylabel('$F_n(x)$', fontsize=9)
    ax_inset.grid(True, linestyle=':', alpha=0.5)
    
    # Annotate VaR levels in the inset
    # Add a marker or note
    ax_inset.annotate('95% VaR', xy=(-emp_var_95, 0.05), xytext=(-emp_var_95 + 0.005, 0.06),
                      arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=0.8), fontsize=8, color='#2c3e50')
    ax_inset.annotate('99% VaR', xy=(-emp_var_99, 0.01), xytext=(-emp_var_99 - 0.015, 0.02),
                      arrowprops=dict(arrowstyle="->", color='#2c3e50', lw=0.8), fontsize=8, color='#2c3e50')
    
    # Save image
    output_img = "var_cdf_comparison.png"
    plt.savefig(output_img, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"Comparative CDF plot successfully saved as '{output_img}'.")
    
    # Save a small text file with the raw results to make reporting easier
    with open("results_summary.txt", "w") as rf:
        rf.write("VALUE AT RISK (VaR) ANALYSIS SUMMARY RESULTS\n")
        rf.write("==================================================\n")
        rf.write(f"Asset Index Analyzed: {ticker}\n")
        rf.write(f"Sample Period       : {start_date} to {end_date}\n")
        rf.write(f"Total Trading Days  : {n_samples}\n")
        rf.write(f"Fitted Normal parameters: mu={mu_norm:.6f}, sigma={std_norm:.6f}\n")
        rf.write(f"Fitted Student's t parameters: nu={df_t:.4f}, mu={loc_t:.6f}, scale={scale_t:.6f}\n")
        rf.write("--------------------------------------------------\n")
        rf.write(f"95% Empirical VaR: {emp_var_95*100:.4f}%\n")
        rf.write(f"95% Normal VaR   : {norm_var_95*100:.4f}%\n")
        rf.write(f"95% Student's t VaR: {t_var_95*100:.4f}%\n")
        rf.write("--------------------------------------------------\n")
        rf.write(f"99% Empirical VaR: {emp_var_99*100:.4f}%\n")
        rf.write(f"99% Normal VaR   : {norm_var_99*100:.4f}%\n")
        rf.write(f"99% Student's t VaR: {t_var_99*100:.4f}%\n")
        rf.write("--------------------------------------------------\n")
        rf.write("KOLMOGOROV-SMIRNOV GOODNESS-OF-FIT RESULTS:\n")
        rf.write(f"Normal Distribution: KS Distance={ks_stat_norm:.5f}, p-value={p_val_norm:.2e}\n")
        rf.write(f"Student's t-Distribution: KS Distance={ks_stat_t:.5f}, p-value={p_val_t:.2e}\n")
        rf.write("==================================================\n")
    print("Results summary successfully written to 'results_summary.txt'.")
    print("=== Quantitative Risk Engine Runs Completed ===")

if __name__ == "__main__":
    run_risk_engine()
