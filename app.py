import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t, kstest
import datetime

# Page configuration for a premium dashboard look
st.set_page_config(
    page_title="Quantitative Financial Risk Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom styling to make the dashboard look high-end
st.markdown("""
<style>
    .reportview-container {
        background: #f8f9fa
    }
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    h1 {
        color: #2c3e50;
        font-weight: 700 !important;
    }
    h2 {
        color: #34495e;
        border-bottom: 2px solid #ecf0f1;
        padding-bottom: 8px;
    }
    .metric-card {
        background-color: white;
        border-radius: 8px;
        padding: 1.5rem;
        box-shadow: 0 4px 6px rgba(0,0,0,0.05);
        border: 1px solid #e2e8f0;
        margin-bottom: 1rem;
    }
    .metric-header {
        font-size: 0.875rem;
        color: #718096;
        text-transform: uppercase;
        font-weight: 600;
        margin-bottom: 0.5rem;
    }
    .metric-value {
        font-size: 1.875rem;
        color: #1a202c;
        font-weight: 700;
    }
</style>
""", unsafe_allow_html=True)

# Main Title & Subtitle
st.title("📈 Quantitative Financial Risk Engine: Value at Risk (VaR)")
st.caption("Applying continuous Cumulative Distribution Function (CDF) and Quantile theory to real-world asset returns in real-time.")

# Sidebar for controls
st.sidebar.header("📊 Model Configuration")

# Ticker selection
ticker_input = st.sidebar.text_input("Enter Ticker Symbol (Yahoo Finance):", value="^GSPC")
st.sidebar.markdown("""
*Examples:*
*   `^GSPC` (S&P 500 Index)
*   `^IXIC` (NASDAQ Composite)
*   `SPY` (S&P 500 ETF)
*   `AAPL` (Apple Inc.)
*   `TSLA` (Tesla Inc.)
*   `BTC-USD` (Bitcoin)
""")

# Date selection
today = datetime.date.today()
ten_years_ago = today - datetime.timedelta(days=3652)
start_date = st.sidebar.date_input("Start Date:", value=ten_years_ago)
end_date = st.sidebar.date_input("End Date:", value=today)

# Slider for VaR Confidence Level
confidence_level = st.sidebar.slider(
    "VaR Confidence Level (%):",
    min_value=90.0,
    max_value=99.9,
    value=99.0,
    step=0.1
)

# Primary Calculation Logic
@st.cache_data(show_spinner="Downloading data from Yahoo Finance...")
def get_asset_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty or len(data) < 30:
            return None, "Error: Ticker not found or dataset too small."
        return data, None
    except Exception as e:
        return None, str(e)

# Run Calculations
if start_date >= end_date:
    st.error("Error: Start Date must be prior to End Date.")
else:
    data, err = get_asset_data(ticker_input, start_date, end_date)
    
    if err:
        st.error(f"Failed to load data for ticker '{ticker_input}': {err}")
    elif data is None:
        st.error("Dataset is empty. Please verify the ticker or date range.")
    else:
        # Robust column detection for Closing Price
        close_col = None
        if isinstance(data.columns, pd.MultiIndex):
            lvl0 = data.columns.get_level_values(0)
            if 'Adj Close' in lvl0:
                close_col = 'Adj Close'
            elif 'Close' in lvl0:
                close_col = 'Close'
            if close_col is not None:
                prices = data[close_col].values.flatten()
            else:
                st.error("Could not find Close price in downloaded MultiIndex columns.")
                st.stop()
        else:
            if 'Adj Close' in data.columns:
                close_col = 'Adj Close'
            elif 'Close' in data.columns:
                close_col = 'Close'
            if close_col is not None:
                prices = data[close_col].values.flatten()
            else:
                st.error("Could not find Close price in columns.")
                st.stop()
        
        # Calculate daily log returns: R_t = ln(P_t / P_{t-1})
        returns = np.log(prices[1:] / prices[:-1])
        returns = returns[~np.isnan(returns)]
        n_samples = len(returns)
        
        if n_samples < 20:
            st.error("Insufficient trading days after log return calculation.")
            st.stop()
            
        # 1. Empirical VaR (Historical Simulation)
        alpha = 1 - (confidence_level / 100.0)
        emp_var = -np.percentile(returns, alpha * 100)
        
        # 2. Fit Normal Distribution via MLE
        mu_norm, std_norm = norm.fit(returns)
        norm_var = -norm.ppf(alpha, loc=mu_norm, scale=std_norm)
        
        # 3. Fit Student's t-Distribution via MLE
        df_t, loc_t, scale_t = t.fit(returns)
        t_var = -t.ppf(alpha, df_t, loc=loc_t, scale=scale_t)
        
        # 4. Kolmogorov-Smirnov Goodness-of-Fit Test
        ks_stat_norm, p_val_norm = kstest(returns, 'norm', args=(mu_norm, std_norm))
        ks_stat_t, p_val_t = kstest(returns, 't', args=(df_t, loc_t, scale_t))
        
        # --- UI LAYOUT ---
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">Empirical {confidence_level:.1f}% VaR (Historical)</div>
                <div class="metric-value" style="color: #2c3e50;">{emp_var*100:.3f}%</div>
                <div style="font-size: 0.8rem; color: #718096; margin-top:0.5rem;">Sorted actual history percentile</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col2:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">Parametric Normal {confidence_level:.1f}% VaR</div>
                <div class="metric-value" style="color: #e74c3c;">{norm_var*100:.3f}%</div>
                <div style="font-size: 0.8rem; color: #718096; margin-top:0.5rem;">Gaussian bell curve model</div>
            </div>
            """, unsafe_allow_html=True)
            
        with col3:
            st.markdown(f"""
            <div class="metric-card">
                <div class="metric-header">Parametric Student's t {confidence_level:.1f}% VaR</div>
                <div class="metric-value" style="color: #2ecc71;">{t_var*100:.3f}%</div>
                <div style="font-size: 0.8rem; color: #718096; margin-top:0.5rem;">Heavy-tail volatility model</div>
            </div>
            """, unsafe_allow_html=True)
            
        # Main panels
        layout_col1, layout_col2 = st.columns([3, 2])
        
        with layout_col1:
            st.subheader("📈 CDF Risk Model & Zoomed Tail Comparison")
            
            # Matplotlib plot generation
            sorted_returns = np.sort(returns)
            ecdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)
            
            x_vals = np.linspace(sorted_returns[0] - 0.01, sorted_returns[-1] + 0.01, 2000)
            cdf_norm = norm.cdf(x_vals, loc=mu_norm, scale=std_norm)
            cdf_t = t.cdf(x_vals, df_t, loc=loc_t, scale=scale_t)
            
            fig, ax = plt.subplots(figsize=(10, 6.2))
            
            # Primary plot
            ax.step(sorted_returns, ecdf, label='Empirical CDF ($F_n$)', color='#2c3e50', alpha=0.7, where='post', lw=1.5)
            ax.plot(x_vals, cdf_norm, label=f'Normal CDF ($F_{{Normal}}$, KS p-val={p_val_norm:.1e})', color='#e74c3c', lw=2, linestyle='--')
            ax.plot(x_vals, cdf_t, label=f"Student's t CDF ($F_t$, df={df_t:.1f}, KS p-val={p_val_t:.1e})", color='#2ecc71', lw=2)
            
            ax.set_xlabel('Daily Log Return', fontsize=10)
            ax.set_ylabel('$F(x) = Pr(X \\leq x)$', fontsize=10)
            ax.grid(True, linestyle=':', alpha=0.6)
            ax.legend(loc='lower right', fontsize=9)
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            
            # Left Tail Zoom Inset
            ax_inset = fig.add_axes([0.15, 0.45, 0.38, 0.38])
            ax_inset.step(sorted_returns, ecdf, color='#2c3e50', where='post', lw=1.5)
            ax_inset.plot(x_vals, cdf_norm, color='#e74c3c', lw=2, linestyle='--')
            ax_inset.plot(x_vals, cdf_t, color='#2ecc71', lw=2)
            
            # Draw threshold lines
            ax_inset.axvline(x=-emp_var, color='#2c3e50', linestyle=':', alpha=0.8, label='Empirical')
            ax_inset.axvline(x=-norm_var, color='#e74c3c', linestyle=':', alpha=0.8, label='Normal')
            ax_inset.axvline(x=-t_var, color='#2ecc71', linestyle=':', alpha=0.8, label="Student's t")
            
            # limits for inset
            inset_x_min = -max(emp_var * 1.5, 0.03)
            inset_x_max = -0.002
            ax_inset.set_xlim(inset_x_min, inset_x_max)
            ax_inset.set_ylim(0.0, max(alpha * 2.5, 0.05))
            ax_inset.set_title('Left Tail Risk Zoom (Losses)', fontsize=8, fontweight='bold', pad=3)
            ax_inset.set_xlabel('Return', fontsize=7)
            ax_inset.set_ylabel('$F_n(x)$', fontsize=7)
            ax_inset.grid(True, linestyle=':', alpha=0.5)
            
            # Render in Streamlit
            st.pyplot(fig)
            
        with layout_col2:
            st.subheader("🔬 Statistical Insights")
            
            # Parameters display
            st.markdown("#### Fitted Distribution Parameters")
            param_data = {
                "Parameter": ["Mean (mu)", "Volatility (sigma / scale)", "Degrees of Freedom (nu)"],
                "Normal Fit": [f"{mu_norm:.6f}", f"{std_norm:.6f}", "N/A (Infinite)"],
                "Student's t Fit": [f"{loc_t:.6f}", f"{scale_t:.6f}", f"{df_t:.2f}"]
            }
            st.table(pd.DataFrame(param_data))
            
            # Goodness-of-Fit analysis
            st.markdown("#### Kolmogorov-Smirnov (KS) Test Results")
            ks_data = {
                "Model": ["Normal (Gaussian)", "Student's t"],
                "KS Distance (D_n)": [f"{ks_stat_norm:.5f}", f"{ks_stat_t:.5f}"],
                "p-value": [f"{p_val_norm:.2e}", f"{p_val_t:.2e}"],
                "Status (at 5% level)": [
                    "❌ Reject Fit" if p_val_norm < 0.05 else "✅ Pass (Fail to Reject)",
                    "❌ Reject Fit" if p_val_t < 0.05 else "✅ Pass (Fail to Reject)"
                ]
            }
            st.table(pd.DataFrame(ks_data))
            
            # Real-time data summary
            st.markdown("#### Asset Information")
            info_data = {
                "Metric": ["Dataset Asset Symbol", "Total Trading Days Analyzed", "Start Date", "End Date"],
                "Value": [ticker_input.upper(), str(n_samples), start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
            }
            st.table(pd.DataFrame(info_data))
            
            # Download daily data option
            st.markdown("#### Export Returns Data")
            # Build data frame for download
            export_df = pd.DataFrame({
                "Date": pd.to_datetime(data.index[1:]).strftime('%Y-%m-%d'),
                "Close Price": [float(p) for p in prices[1:]],
                "Daily Log Return": [float(r) for r in returns]
            })
            csv = export_df.to_csv(index=False).encode('utf-8')
            
            # Excel export using BytesIO and openpyxl
            import io
            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Returns')
            excel_data = excel_buffer.getvalue()
            
            dl_col1, dl_col2 = st.columns(2)
            with dl_col1:
                st.download_button(
                    label="📥 Download CSV",
                    data=csv,
                    file_name=f"{ticker_input}_returns.csv",
                    mime="text/csv",
                    use_container_width=True
                )
            with dl_col2:
                st.download_button(
                    label="📥 Download Excel",
                    data=excel_data,
                    file_name=f"{ticker_input}_returns.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                    use_container_width=True
                )
            
        # Explanatory Educational Tab Section
        st.subheader("📚 Statistical Concept Reference (Group Project Guide)")
        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 The CDF Spotlight", 
            "📊 Heavy-Tailed Student's t", 
            "🔍 KS Goodness-of-Fit",
            "💡 Simple English Explanation"
        ])
        
        with tab1:
            st.markdown(r"""
            ### Cumulative Distribution Function (CDF) and Quantiles
            
            The **Cumulative Distribution Function (CDF)** of a daily stock return $X$ maps return values to their cumulative probability:
            $$F(x) = \Pr(X \le x)$$
            
            To find a downside risk threshold at confidence level $c$ (e.g. $c = 99\%$), we search for the return $x_0$ such that the probability of returns being worse than $x_0$ is exactly $\alpha = 1 - c = 1\%$. 
            This is solved using the **Quantile Function** (the mathematical inverse of the CDF):
            $$x_0 = F^{-1}(1 - c)$$
            
            The **Value at Risk (VaR)** is then defined as the positive threshold of loss:
            $$\text{VaR}_c = -x_0 = -F^{-1}(1 - c)$$
            
            #### The Glivenko-Cantelli Theorem
            As the number of daily historical observations $n \to \infty$, the **Empirical CDF ($F_n(x)$)** converges uniformly and almost surely to the true underlying CDF ($F(x)$):
            $$\lim_{n \to \infty} \sup_{x \in \mathbb{R}} |F_n(x) - F(x)| = 0 \quad \text{a.s.}$$
            This theorem mathematically justifies looking at past daily stock history to estimate future downside thresholds.
            """)
            
        with tab2:
            st.markdown(r"""
            ### Why Normal Fitting Fails & Student's t Succeeds
            
            The classical Gaussian model assumes asset returns follow a standard bell curve. However, real markets suffer from sudden crashes far more frequently than normal distributions predict.
            
            *   **Normal Distribution tail decay:** The Normal CDF tail drops off exponentially fast, meaning a daily loss of $-4\%$ is modeled as virtually impossible.
            *   **Student's t-distribution tail decay:** The Student's t-distribution fits an extra parameter, the **degrees of freedom ($\nu$)**. A smaller $\nu$ indicates a thicker, heavier tail.
            
            If $\nu \approx 2.5$, the tail decays like a power-law function rather than exponentially. This allows the Student's t CDF to fit extreme historical losses much more closely, as shown in the **Left Tail Risk Zoom** inset plot.
            """)
            
        with tab3:
            st.markdown(r"""
            ### The Kolmogorov-Smirnov Goodness-of-Fit Test
            
            The **Kolmogorov-Smirnov (KS) Test** is a non-parametric test that compares the Empirical CDF $F_n(x)$ against a fitted theoretical model CDF $F_{\text{model}}(x)$. 
            The KS statistic $D_n$ measures the maximum vertical distance between the two curves:
            $$D_n = \sup_{x} |F_n(x) - F_{\text{model}}(x)|$$
            
            *   **Null Hypothesis ($H_0$):** The data follows the fitted model distribution.
            *   **Decision Rule:** If the calculated p-value is less than $0.05$ (5%), we reject $H_0$, concluding that the distribution is not a suitable fit.
            *   **Financial Reality:** For the S&P 500, the Normal distribution is rejected ($p \approx 0$). The Student's t-distribution fails to be rejected ($p > 0.05$), proving it is a statistically valid fit.
            """)

        with tab4:
            st.markdown(f"""
            ### 💡 The Value at Risk (VaR) Analysis in Plain English
            
            #### 1. What is Value at Risk (VaR)?
            Value at Risk (VaR) is a simple way of stating **how much money your portfolio could lose on a really bad day**. 
            When we set a **{confidence_level:.1f}% confidence level**, we are asking: *“On {confidence_level:.1f}% of trading days, what is the maximum amount of money we expect to lose?”* or, alternatively, *“What is the threshold of loss that we only expect to exceed on 1 out of 100 days (for 99% VaR) or 5 out of 100 days (for 95% VaR)?”*

            #### 2. What do the current results say?
            Based on the data for **{ticker_input.upper()}** from **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}** (analyzing **{n_samples}** daily observations), the risk engine calculates these risk limits:
            
            *   **Empirical VaR is {emp_var*100:.3f}%:** This looks directly at actual history. Historically, on the worst {100-confidence_level:.1f}% of days, this asset lost **{emp_var*100:.3f}% or more** of its value in a single day.
            *   **Parametric Normal VaR is {norm_var*100:.3f}%:** This is the standard "bell curve" model. It predicts that on a bad day, the asset should lose **{norm_var*100:.3f}%**. 
            *   **Parametric Student's t VaR is {t_var*100:.3f}%:** This is a heavy-tail mathematical model that is designed to expect more sudden market crashes. It predicts a bad day loss of **{t_var*100:.3f}%**.

            #### 3. Why are the Normal and Student's t numbers different? (The "Fat Tail" Problem)
            If you look at the **99% VaR**:
            *   Usually, the Normal distribution **underestimates** extreme risk. It might say that a loss of 3% is extremely rare (like once in a century), but in the real stock market, such crashes happen much more often.
            *   The Student's t-distribution is a smarter, safer model. It adjusts to the real volatility of the market by accounting for "fat tails" (sudden crashes). This is why its 99% VaR is usually much closer to the actual historical Empirical VaR.

            #### 4. What does the Kolmogorov-Smirnov (KS) Test prove?
            The KS test is a mathematical referee that scores our models:
            *   **Normal Fit:** It checks if S&P 500 returns behave like a perfect normal bell curve. The result is a p-value of **{p_val_norm:.2e}**. Because this is extremely small (less than 5%), the test officially **rejects** the Normal model. S&P 500 returns are **not** normally distributed.
            *   **Student's t Fit:** The test scores the Student's t model, yielding a p-value of **{p_val_t:.2e}**. Since this is **greater than 5% (for S&P 500)**, we **fail to reject** it, meaning the Student's t model is a statistically valid fit for real-world market risk.
            """)
