import streamlit as st
import yfinance as yf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import norm, t, kstest
import datetime
import io

# Page configuration for a premium dashboard look
st.set_page_config(
    page_title="Quantitative Financial Risk Engine",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────────────
# Premium Dark Theme CSS — Glassmorphism + Modern Typography
# ─────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;500;600;700;800&family=Plus+Jakarta+Sans:wght@300;400;500;600;700&display=swap');

    /* ── Global Overrides ── */
    html, body, [data-testid="stAppViewContainer"] {
        background-color: #080c14 !important;
        color: #f3f4f6;
    }
    [data-testid="stAppViewContainer"] {
        background: linear-gradient(160deg, #080c14 0%, #0d1423 40%, #080c14 100%) !important;
    }
    [data-testid="stHeader"] {
        background: rgba(8, 12, 20, 0.8) !important;
        backdrop-filter: blur(12px);
    }

    /* ── Sidebar ── */
    [data-testid="stSidebar"] {
        background: rgba(10, 15, 28, 0.95) !important;
        border-right: 1px solid rgba(255,255,255,0.06);
    }
    [data-testid="stSidebar"] * {
        color: #c8d0dc !important;
    }
    [data-testid="stSidebar"] h1, [data-testid="stSidebar"] h2,
    [data-testid="stSidebar"] h3, [data-testid="stSidebar"] .stMarkdown h1,
    [data-testid="stSidebar"] .stMarkdown h2, [data-testid="stSidebar"] .stMarkdown h3 {
        color: #f3f4f6 !important;
    }
    [data-testid="stSidebar"] input,
    [data-testid="stSidebar"] [data-baseweb="input"] input,
    [data-testid="stSidebar"] [data-testid="stDateInput"] input {
        background: rgba(255,255,255,0.06) !important;
        border: 1px solid rgba(255,255,255,0.1) !important;
        border-radius: 8px !important;
        color: #f3f4f6 !important;
    }
    /* Date picker & select box containers */
    [data-testid="stSidebar"] [data-baseweb="input"],
    [data-testid="stSidebar"] [data-baseweb="base-input"] {
        background: rgba(255,255,255,0.06) !important;
        border-color: rgba(255,255,255,0.1) !important;
    }
    /* Date picker popover / calendar panel */
    [data-baseweb="calendar"], [data-baseweb="popover"],
    [data-baseweb="datepicker"] {
        background: #111827 !important;
        color: #f3f4f6 !important;
    }
    [data-baseweb="calendar"] * {
        color: #d1d5db !important;
    }
    [data-baseweb="calendar"] [aria-selected="true"] {
        background: #00d2ff !important;
        color: #080c14 !important;
    }
    /* Slider & checkbox */
    [data-testid="stSidebar"] .stSlider > div > div > div {
        color: #f3f4f6 !important;
    }

    /* ── Main container ── */
    .main .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1280px;
    }

    /* ── Typography ── */
    h1 {
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        letter-spacing: -1px !important;
        background: linear-gradient(135deg, #ffffff 30%, #a5b4fc 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        padding-bottom: 0.25rem;
    }
    h2, h3 {
        font-family: 'Outfit', sans-serif !important;
        color: #e2e8f0 !important;
        font-weight: 700 !important;
    }
    p, li, span, div {
        font-family: 'Plus Jakarta Sans', sans-serif;
    }
    .stCaption, [data-testid="stCaptionContainer"] {
        color: #9ca3af !important;
        font-size: 1rem !important;
    }

    /* ── Glass Metric Cards ── */
    .metric-card {
        background: rgba(13, 20, 35, 0.65);
        backdrop-filter: blur(16px);
        -webkit-backdrop-filter: blur(16px);
        border: 1px solid rgba(255,255,255,0.07);
        border-radius: 16px;
        padding: 1.75rem 1.5rem;
        transition: all 0.35s cubic-bezier(0.4, 0, 0.2, 1);
        position: relative;
        overflow: hidden;
    }
    .metric-card::before {
        content: '';
        position: absolute;
        top: 0; left: 0;
        width: 100%; height: 3px;
        border-radius: 16px 16px 0 0;
    }
    .metric-card.blue::before  { background: linear-gradient(90deg, #00d2ff 0%, #0084ff 100%); }
    .metric-card.red::before   { background: linear-gradient(90deg, #ff5252 0%, #ff1744 100%); }
    .metric-card.green::before { background: linear-gradient(90deg, #00e676 0%, #00c853 100%); }
    .metric-card:hover {
        border-color: rgba(255,255,255,0.14);
        transform: translateY(-4px);
        box-shadow: 0 12px 40px rgba(0,0,0,0.35);
    }
    .metric-header {
        font-family: 'Plus Jakarta Sans', sans-serif;
        font-size: 0.78rem;
        color: #9ca3af;
        text-transform: uppercase;
        font-weight: 600;
        letter-spacing: 0.8px;
        margin-bottom: 0.6rem;
    }
    .metric-value {
        font-family: 'Outfit', sans-serif;
        font-size: 2rem;
        font-weight: 800;
        letter-spacing: -0.5px;
    }
    .metric-sub {
        font-size: 0.78rem;
        color: #6b7280;
        margin-top: 0.5rem;
        font-weight: 500;
    }

    /* ── Glass Panel (chart/table wrappers) ── */
    .glass-panel {
        background: rgba(13, 20, 35, 0.55);
        backdrop-filter: blur(14px);
        -webkit-backdrop-filter: blur(14px);
        border: 1px solid rgba(255,255,255,0.06);
        border-radius: 16px;
        padding: 2rem;
        margin-bottom: 1.5rem;
    }
    .glass-panel:hover {
        border-color: rgba(0, 210, 255, 0.15);
    }

    /* ── Section Title ── */
    .section-label {
        font-family: 'Outfit', sans-serif;
        font-size: 1.35rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 1.25rem;
        display: flex;
        align-items: center;
        gap: 0.6rem;
    }

    /* ── Streamlit table dark theme ── */
    [data-testid="stTable"] table {
        background: rgba(13, 20, 35, 0.5) !important;
        border-radius: 10px;
        overflow: hidden;
    }
    [data-testid="stTable"] th {
        background: rgba(0, 210, 255, 0.08) !important;
        color: #00d2ff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        font-size: 0.82rem !important;
        text-transform: uppercase;
        letter-spacing: 0.5px;
        border-bottom: 1px solid rgba(255,255,255,0.08) !important;
    }
    [data-testid="stTable"] td {
        color: #d1d5db !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.88rem !important;
        border-bottom: 1px solid rgba(255,255,255,0.04) !important;
    }

    /* ── Tabs ── */
    [data-testid="stTabs"] button {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 600 !important;
        color: #9ca3af !important;
        border-bottom: 2px solid transparent !important;
        transition: all 0.3s ease;
    }
    [data-testid="stTabs"] button[aria-selected="true"] {
        color: #00d2ff !important;
        border-bottom-color: #00d2ff !important;
    }
    [data-testid="stTabs"] button:hover {
        color: #e2e8f0 !important;
    }

    /* ── Download buttons ── */
    [data-testid="stDownloadButton"] button {
        background: rgba(0, 210, 255, 0.1) !important;
        border: 1px solid rgba(0, 210, 255, 0.25) !important;
        color: #00d2ff !important;
        font-weight: 600 !important;
        border-radius: 10px !important;
        transition: all 0.3s ease !important;
    }
    [data-testid="stDownloadButton"] button:hover {
        background: rgba(0, 210, 255, 0.2) !important;
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0, 210, 255, 0.15);
    }

    /* ── Divider ── */
    hr {
        border-color: rgba(255,255,255,0.06) !important;
        margin: 2rem 0 !important;
    }

    /* ── Expander ── */
    [data-testid="stExpander"] {
        background: rgba(13, 20, 35, 0.4) !important;
        border: 1px solid rgba(255,255,255,0.06) !important;
        border-radius: 12px !important;
    }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────────────
# Header
# ─────────────────────────────────────────────────────────────
st.title("📈 Quantitative Financial Risk Engine")
st.caption("Applying continuous Cumulative Distribution Function (CDF) and Quantile theory to real-world asset returns in real-time.")

# ─────────────────────────────────────────────────────────────
# Sidebar — Model Configuration
# ─────────────────────────────────────────────────────────────
st.sidebar.markdown("""
<div style="text-align:center; padding: 0.5rem 0 1rem;">
    <span style="font-family: 'Outfit', sans-serif; font-size: 1.3rem; font-weight: 700;
    background: linear-gradient(135deg, #fff, #00d2ff); -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;">⚙️ Model Configuration</span>
</div>
""", unsafe_allow_html=True)

# Ticker selection
ticker_input = st.sidebar.text_input("Enter Ticker Symbol (Yahoo Finance):", value="^GSPC")
st.sidebar.markdown("""
*Quick-pick tickers:*
*   `^GSPC` — S&P 500 Index
*   `^IXIC` — NASDAQ Composite
*   `^NSEI` — NIFTY 50 Index 🇮🇳
*   `SPY` — S&P 500 ETF
*   `AAPL` — Apple Inc.
*   `TSLA` — Tesla Inc.
*   `BTC-USD` — Bitcoin
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

# Checkbox for running parametric bootstrap
run_bootstrap = st.sidebar.checkbox(
    "Run Bootstrap KS Correction",
    value=False,
    help="Parametric bootstrap refits Student's t over 500 resampled datasets to correct the KS test p-value for parameter estimation bias. Takes ~1-2 minutes."
)

st.sidebar.divider()
st.sidebar.caption("Built for DSC003 Probability & Statistics · [GitHub](https://github.com/gk5760476-hash/GAURAVSTATISTICS)")

# ─────────────────────────────────────────────────────────────
# Data Fetching (cached)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner="Downloading data from Yahoo Finance...")
def get_asset_data(ticker, start, end):
    try:
        data = yf.download(ticker, start=start, end=end)
        if data.empty or len(data) < 30:
            return None, "Error: Ticker not found or dataset too small."
        return data, None
    except Exception as e:
        return None, str(e)

# ─────────────────────────────────────────────────────────────
# Bootstrap Computation (BUG FIX #2: st.progress moved OUTSIDE
# the cached function to avoid StreamlitAPIException)
# ─────────────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _compute_bootstrap_ks(returns_tuple, df_t, loc_t, scale_t, B=500):
    """Pure computation — no Streamlit widgets inside cache."""
    returns = np.array(returns_tuple)
    n = len(returns)
    ks_stat, _ = kstest(returns, t(df=df_t, loc=loc_t, scale=scale_t).cdf)
    bootstrap_stats = []
    for i in range(B):
        boot_sample = t.rvs(df_t, loc=loc_t, scale=scale_t, size=n)
        b_df, b_loc, b_scale = t.fit(boot_sample)
        b_ks, _ = kstest(boot_sample, t(df=b_df, loc=b_loc, scale=b_scale).cdf)
        bootstrap_stats.append(b_ks)
    bootstrap_stats = np.array(bootstrap_stats)
    corrected_p_val = np.sum(bootstrap_stats >= ks_stat) / B
    return corrected_p_val


def get_bootstrap_p_value(returns, df_t, loc_t, scale_t, B=500):
    """Wrapper that shows a progress spinner outside the cache boundary."""
    with st.spinner(f"Running parametric bootstrap ({B} iterations) — this may take 1-2 minutes..."):
        result = _compute_bootstrap_ks(tuple(returns), df_t, loc_t, scale_t, B)
    return result


# ─────────────────────────────────────────────────────────────
# MAIN COMPUTATION
# ─────────────────────────────────────────────────────────────
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
        raw_log_returns = np.log(prices[1:] / prices[:-1])
        valid_mask = ~np.isnan(raw_log_returns) & ~np.isinf(raw_log_returns)

        returns = raw_log_returns[valid_mask]
        aligned_dates = data.index[1:][valid_mask]
        aligned_prices = prices[1:][valid_mask]

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
        ks_stat_norm, p_val_norm = kstest(returns, norm(loc=mu_norm, scale=std_norm).cdf)
        ks_stat_t, p_val_t = kstest(returns, t(df=df_t, loc=loc_t, scale=scale_t).cdf)

        # Calculate bootstrap corrected p-value if checked
        p_val_t_corr = None
        if run_bootstrap:
            p_val_t_corr = get_bootstrap_p_value(returns, df_t, loc_t, scale_t, B=500)

        # ═══════════════════════════════════════════════════
        # UI LAYOUT — Premium Dark Dashboard
        # ═══════════════════════════════════════════════════

        # ── Row 1: VaR Metric Cards ──
        st.markdown('<div class="section-label">📊 Value at Risk Summary</div>', unsafe_allow_html=True)

        col1, col2, col3 = st.columns(3, gap="medium")

        with col1:
            st.markdown(f"""
            <div class="metric-card blue">
                <div class="metric-header">Empirical {confidence_level:.1f}% VaR</div>
                <div class="metric-value" style="color: #00d2ff;">{emp_var*100:.3f}%</div>
                <div class="metric-sub">Historical simulation · sorted percentile</div>
            </div>
            """, unsafe_allow_html=True)

        with col2:
            st.markdown(f"""
            <div class="metric-card red">
                <div class="metric-header">Normal {confidence_level:.1f}% VaR</div>
                <div class="metric-value" style="color: #ff5252;">{norm_var*100:.3f}%</div>
                <div class="metric-sub">Gaussian bell curve model</div>
            </div>
            """, unsafe_allow_html=True)

        with col3:
            st.markdown(f"""
            <div class="metric-card green">
                <div class="metric-header">Student's t {confidence_level:.1f}% VaR</div>
                <div class="metric-value" style="color: #00e676;">{t_var*100:.3f}%</div>
                <div class="metric-sub">Heavy-tail volatility model</div>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<div style='height: 1.5rem;'></div>", unsafe_allow_html=True)

        # ── Row 2: CDF Chart + Statistical Fit ──
        chart_col, stats_col = st.columns([3, 2], gap="large")

        with chart_col:
            st.markdown('<div class="section-label">📈 CDF Risk Model & Tail Comparison</div>', unsafe_allow_html=True)

            # Matplotlib plot with dark theme
            sorted_returns = np.sort(returns)
            ecdf = np.arange(1, len(sorted_returns) + 1) / len(sorted_returns)

            x_vals = np.linspace(sorted_returns[0] - 0.01, sorted_returns[-1] + 0.01, 2000)
            cdf_norm = norm.cdf(x_vals, loc=mu_norm, scale=std_norm)
            cdf_t = t.cdf(x_vals, df_t, loc=loc_t, scale=scale_t)

            # Dark figure style
            with plt.style.context('dark_background'):
                fig, ax = plt.subplots(figsize=(10, 6.2))
                fig.patch.set_facecolor('#0d1423')
                ax.set_facecolor('#0d1423')

                # Primary plot
                ax.step(sorted_returns, ecdf, label='Empirical CDF ($F_n$)',
                        color='#00d2ff', alpha=0.7, where='post', lw=1.5)
                ax.plot(x_vals, cdf_norm,
                        label=f'Normal CDF (KS p={p_val_norm:.1e})',
                        color='#ff5252', lw=2, linestyle='--')

                # Student's t label
                if p_val_t_corr is not None:
                    t_label = f"Student's t CDF (df={df_t:.1f}, KS p={p_val_t:.1e}, corrected={p_val_t_corr:.3f})"
                else:
                    t_label = f"Student's t CDF (df={df_t:.1f}, KS p={p_val_t:.1e})"
                ax.plot(x_vals, cdf_t, label=t_label, color='#00e676', lw=2)

                ax.set_xlabel('Daily Log Return', fontsize=10, color='#9ca3af')
                ax.set_ylabel('$F(x) = Pr(X \\leq x)$', fontsize=10, color='#9ca3af')
                ax.tick_params(colors='#6b7280')
                ax.grid(True, linestyle=':', alpha=0.25, color='#374151')
                ax.legend(loc='lower right', fontsize=8.5, facecolor='#0d1423',
                          edgecolor='#1f2937', labelcolor='#d1d5db')
                ax.spines['top'].set_visible(False)
                ax.spines['right'].set_visible(False)
                ax.spines['bottom'].set_color('#1f2937')
                ax.spines['left'].set_color('#1f2937')

                # Left Tail Zoom Inset
                ax_inset = fig.add_axes([0.15, 0.45, 0.38, 0.38])
                ax_inset.set_facecolor('#111827')
                ax_inset.step(sorted_returns, ecdf, color='#00d2ff', where='post', lw=1.5)
                ax_inset.plot(x_vals, cdf_norm, color='#ff5252', lw=2, linestyle='--')
                ax_inset.plot(x_vals, cdf_t, color='#00e676', lw=2)

                ax_inset.axvline(x=-emp_var, color='#00d2ff', linestyle=':', alpha=0.8)
                ax_inset.axvline(x=-norm_var, color='#ff5252', linestyle=':', alpha=0.8)
                ax_inset.axvline(x=-t_var, color='#00e676', linestyle=':', alpha=0.8)

                inset_x_min = -max(emp_var * 1.5, 0.03)
                inset_x_max = -0.002
                ax_inset.set_xlim(inset_x_min, inset_x_max)
                ax_inset.set_ylim(0.0, max(alpha * 2.5, 0.05))
                ax_inset.set_title('Left Tail Risk Zoom', fontsize=8,
                                   fontweight='bold', pad=3, color='#d1d5db')
                ax_inset.set_xlabel('Return', fontsize=7, color='#9ca3af')
                ax_inset.set_ylabel('$F_n(x)$', fontsize=7, color='#9ca3af')
                ax_inset.tick_params(colors='#6b7280', labelsize=7)
                ax_inset.grid(True, linestyle=':', alpha=0.2, color='#374151')
                for spine in ax_inset.spines.values():
                    spine.set_color('#1f2937')

                fig.tight_layout()
                st.pyplot(fig)

        with stats_col:
            # ── Fitted Parameters ──
            st.markdown('<div class="section-label">🔬 Model Estimation</div>', unsafe_allow_html=True)

            st.markdown("**Fitted Distribution Parameters**")
            param_data = {
                "Parameter": ["Mean (μ)", "Volatility (σ / scale)", "Degrees of Freedom (ν)"],
                "Normal Fit": [f"{mu_norm:.6f}", f"{std_norm:.6f}", "∞ (Gaussian)"],
                "Student's t Fit": [f"{loc_t:.6f}", f"{scale_t:.6f}", f"{df_t:.2f}"]
            }
            st.table(pd.DataFrame(param_data))

            # ── KS Test Results ──
            st.markdown("**Kolmogorov-Smirnov (KS) Test**")

            p_val_t_display = f"{p_val_t:.2e} (naive)"
            if p_val_t_corr is not None:
                p_val_t_display += f" | {p_val_t_corr:.3f} (corrected)"
                t_status = "❌ Reject" if p_val_t_corr < 0.05 else "✅ Pass"
            else:
                p_val_t_display += " | Run bootstrap ↑"
                t_status = "⏳ Bootstrap needed"

            ks_data = {
                "Model": ["Normal (Gaussian)", "Student's t"],
                "KS Distance (Dₙ)": [f"{ks_stat_norm:.5f}", f"{ks_stat_t:.5f}"],
                "p-value": [f"{p_val_norm:.2e}", p_val_t_display],
                "Status (5%)": [
                    "❌ Reject" if p_val_norm < 0.05 else "✅ Pass",
                    t_status
                ]
            }
            st.table(pd.DataFrame(ks_data))
            if p_val_t_corr is None:
                st.caption("💡 *Enable 'Run Bootstrap KS Correction' in the sidebar for corrected p-values.*")

        st.divider()

        # ── Row 3: Asset Details & Data Export (Side-by-Side to prevent vertical gap) ──
        info_col, export_col = st.columns([3, 2], gap="large")

        with info_col:
            st.markdown('<div class="section-label">📋 Dataset Information</div>', unsafe_allow_html=True)
            info_data = {
                "Metric": ["Ticker Symbol", "Trading Days Analyzed", "Start Date", "End Date"],
                "Value": [ticker_input.upper(), str(n_samples),
                          start_date.strftime("%Y-%m-%d"), end_date.strftime("%Y-%m-%d")]
            }
            st.table(pd.DataFrame(info_data))

        with export_col:
            st.markdown('<div class="section-label">💾 Export Returns Data</div>', unsafe_allow_html=True)
            st.caption("Download the aligned daily close prices and log returns for custom models or spreadsheet analysis.")
            
            export_df = pd.DataFrame({
                "Date": pd.to_datetime(aligned_dates).strftime('%Y-%m-%d'),
                "Close Price": [float(p) for p in aligned_prices],
                "Daily Log Return": [float(r) for r in returns]
            })
            csv = export_df.to_csv(index=False).encode('utf-8')

            excel_buffer = io.BytesIO()
            with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
                export_df.to_excel(writer, index=False, sheet_name='Returns')
            excel_data = excel_buffer.getvalue()

            st.markdown("<div style='height: 0.75rem;'></div>", unsafe_allow_html=True)
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

        st.divider()

        # ═══════════════════════════════════════════════════
        # Educational Reference Tabs
        # ═══════════════════════════════════════════════════
        st.markdown('<div class="section-label">📚 Statistical Concept Reference</div>', unsafe_allow_html=True)

        tab1, tab2, tab3, tab4 = st.tabs([
            "🎯 The CDF Spotlight",
            "📊 Heavy-Tailed Student's t",
            "🔍 KS Goodness-of-Fit",
            "💡 Explanation"
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

            The **Kolmogorov-Smirnov (KS) Test** compares the Empirical CDF $F_n(x)$ against a fitted theoretical model CDF $F_{\text{model}}(x)$.
            The KS statistic $D_n$ measures the maximum vertical distance between the two curves:
            $$D_n = \sup_{x} |F_n(x) - F_{\text{model}}(x)|$$

            *   **Null Hypothesis ($H_0$):** The data follows the fitted model distribution.
            *   **Decision Rule:** If the p-value is less than $0.05$ (5%), we reject $H_0$, concluding that the distribution is not a suitable fit.
            *   **The Parameter Estimation Bias (Lilliefors Effect):** When the parameters of the distribution ($\mu, \sigma$ for Normal, or $\nu, \mu, s$ for Student's t) are estimated from the *same* sample data on which the test is performed, the classical KS test p-value formula does not strictly apply and is inflated. Because the fitted curve is "tuned" to the data, $D_n$ is naturally smaller, leading to an optimistically inflated naive p-value.
            *   **Parametric Bootstrap Correction:** By running a parametric bootstrap (generating $B = 500$ datasets from the fitted distribution, refitting parameters, and calculating the null distribution of $D_n$), we find the true corrected p-value, which is typically more conservative than the naive value.
            """)

        # BUG FIX #1: Guard p_val_t_corr formatting — it is None when bootstrap is unchecked
        with tab4:
            # Build the corrected p-value text conditionally
            if p_val_t_corr is not None:
                corr_text = f"**{p_val_t_corr:.3f}** (computed live via bootstrap)"
                ks_t_verdict = (
                    f"Since {p_val_t_corr:.3f} {'< 0.05, we technically reject it under strict statistical rules' if p_val_t_corr < 0.05 else '>= 0.05, we fail to reject it'}."
                )
            else:
                corr_text = "*(not yet computed — enable Bootstrap in the sidebar)*"
                ks_t_verdict = "Enable the bootstrap checkbox in the sidebar to compute the corrected p-value and determine the final verdict."

            st.markdown(f"""
            ### 💡 The Value at Risk (VaR) Analysis — Explanation

            #### 1. What is Value at Risk (VaR)?
            Value at Risk (VaR) is a simple way of stating **how much money your portfolio could lose on a really bad day**.
            When we set a **{confidence_level:.1f}% confidence level**, we are asking: *"On {confidence_level:.1f}% of trading days, what is the maximum amount of money we expect to lose?"* or, alternatively, *"What is the threshold of loss that we only expect to exceed on {100-confidence_level:.1f} out of 100 days?"*

            #### 2. What do the current results say?
            Based on the data for **{ticker_input.upper()}** from **{start_date.strftime('%Y-%m-%d')}** to **{end_date.strftime('%Y-%m-%d')}** (analyzing **{n_samples}** daily observations), the risk engine calculates these risk limits:

            *   **Empirical VaR is {emp_var*100:.3f}%:** This looks directly at actual history. Historically, on the worst {100-confidence_level:.1f}% of days, this asset lost **{emp_var*100:.3f}% or more** of its value in a single day.
            *   **Parametric Normal VaR is {norm_var*100:.3f}%:** This is the standard "bell curve" model. It predicts that on a bad day, the asset should lose **{norm_var*100:.3f}%**.
            *   **Parametric Student's t VaR is {t_var*100:.3f}%:** This is a heavy-tail mathematical model that is designed to expect more sudden market crashes. It predicts a bad day loss of **{t_var*100:.3f}%**.

            #### 3. Why are the Normal and Student's t numbers different? (The "Fat Tail" Problem)
            If you look at the **{confidence_level:.1f}% VaR**:
            *   Usually, the Normal distribution **underestimates** extreme risk. It might say that a loss of 3% is extremely rare (like once in a century), but in the real stock market, such crashes happen much more often.
            *   The Student's t-distribution is a smarter, safer model. It adjusts to the real volatility of the market by accounting for "fat tails" (sudden crashes). This is why its VaR is usually much closer to the actual historical Empirical VaR.

            #### 4. What does the Kolmogorov-Smirnov (KS) Test prove?
            The KS test is a mathematical referee that scores our models:
            *   **Normal Fit:** It checks if the returns behave like a perfect normal bell curve. The result is a p-value of **{p_val_norm:.2e}**. {"Because this is extremely small (less than 5%), the test officially **rejects** the Normal model." if p_val_norm < 0.05 else "The test does not reject the Normal model at the 5% level."}
            *   **Student's t Fit:** The naive p-value is **{p_val_t:.2e}**. The corrected p-value is {corr_text}. {ks_t_verdict} However, the Student's t model remains **materially superior**, as it reduces the KS distance Dₙ by **{ks_stat_norm/ks_stat_t:.1f}×** compared to the Normal distribution ({ks_stat_t:.5f} vs {ks_stat_norm:.5f}).
            """)
