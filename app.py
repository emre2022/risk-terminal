import streamlit as st
import appdirs as ad
# --- YFINANCE CACHE FIX ---
ad.user_cache_dir = lambda *args: "/tmp"
# --------------------------

import yfinance as yf
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import datetime
from scipy.stats import norm
from data_cleaner import clean_market_data

# ---------------------------------------------------------
# PAGE SETTINGS
# ---------------------------------------------------------
st.set_page_config(page_title="Institutional Risk Terminal", layout="wide", page_icon="🏦")

# ---------------------------------------------------------
# MASTER INSTRUMENT LIST
# ---------------------------------------------------------
INSTRUMENTS = {
    "Indices": {
        "NQ=F": {"name": "Nasdaq 100 E-mini", "size": 20, "def_iv": 18.12, "cat": "Equity"},
        "ES=F": {"name": "S&P 500 E-mini", "size": 50, "def_iv": 14.50, "cat": "Equity"},
        "YM=F": {"name": "Dow Jones E-mini", "size": 5, "def_iv": 13.00, "cat": "Equity"},
        "^GDAXI": {"name": "DAX 40 Index", "size": 25, "def_iv": 16.00, "cat": "Equity"},
    },
    "Forex": {
        "EURUSD=X": {"name": "EUR/USD Spot", "size": 100000, "def_iv": 7.50, "cat": "FX"},
        "6E=F": {"name": "Euro FX Futures", "size": 125000, "def_iv": 7.50, "cat": "FX"},
        "GBPUSD=X": {"name": "GBP/USD Spot", "size": 100000, "def_iv": 8.20, "cat": "FX"},
        "USDJPY=X": {"name": "USD/JPY Spot", "size": 100000, "def_iv": 10.50, "cat": "FX"},
        "USDTRY=X": {"name": "USD/TRY Spot", "size": 100000, "def_iv": 25.00, "cat": "FX"},
    },
    "Commodities & Energy": {
        "GC=F": {"name": "Gold Futures (100oz)", "size": 100, "def_iv": 15.00, "cat": "Commodity"},
        "SI=F": {"name": "Silver Futures (5000oz)", "size": 5000, "def_iv": 28.00, "cat": "Commodity"},
        "CL=F": {"name": "Crude Oil WTI", "size": 1000, "def_iv": 35.00, "cat": "Commodity"},
        "NG=F": {"name": "Natural Gas", "size": 10000, "def_iv": 50.00, "cat": "Commodity"},
        "PL=F": {"name": "Platinum", "size": 50, "def_iv": 22.00, "cat": "Commodity"},
        "PA=F": {"name": "Palladium", "size": 100, "def_iv": 30.00, "cat": "Commodity"},
    },
    "Crypto": {
        "BTC-USD": {"name": "Bitcoin", "size": 1, "def_iv": 55.00, "cat": "Crypto"},
        "ETH-USD": {"name": "Ethereum", "size": 1, "def_iv": 60.00, "cat": "Crypto"},
    }
}

# --- NAME MAPPING ---
NAME_MAPPING = {
    "EUR/USD": "EURUSD=X",
    "GBP/USD": "GBPUSD=X",
    "USD/JPY": "USDJPY=X",
    "USD/TRY": "USDTRY=X",
    "DAX": "^GDAXI",    
    "US30": "YM=F",     # Dow Jones
    "NAS100": "NQ=F",   # Nasdaq
    "E DJI": "YM=F",    # Dow Jones alt
    "Em NQ": "NQ=F",    # Nasdaq alt
    "XAU/USD": "GC=F",  # Gold
    "Lt Crude": "CL=F", # Crude Oil
    "Brent Crude": "CL=F",
    "Natural Gas": "NG=F",
    "Platinum": "PL=F",
    "BTC/USD": "BTC-USD",
    "ETH/USD": "ETH-USD"
}

FLAT_INSTRUMENTS = {}
for cat, items in INSTRUMENTS.items():
    for ticker, info in items.items():
        FLAT_INSTRUMENTS[f"{info['name']} ({ticker})"] = {"ticker": ticker, "size": info['size']}

# ---------------------------------------------------------
# CALLBACK FUNCTION
# ---------------------------------------------------------
def update_portfolio_from_cleaner():
    if 'cleaned_data' in st.session_state:
        df_result = st.session_state['cleaned_data']
        match_count = 0
        for index, row in df_result.iterrows():
            # Updated to match English column names from data_cleaner.py
            inst_name = row['Instrument']
            val = row['Value']
            
            if pd.isna(val): continue
            
            ticker = NAME_MAPPING.get(inst_name)
            if ticker:
                lot_size = abs(float(val))
                side = "Long" if float(val) >= 0 else "Short"
                st.session_state[f"lot_{ticker}"] = lot_size
                st.session_state[f"side_{ticker}"] = side
                match_count += 1
        if match_count > 0:
            st.toast(f"✅ {match_count} instruments updated successfully!", icon="🚀")
        else:
            st.toast("⚠️ No matching instruments found.", icon="⚠️")

# ---------------------------------------------------------
# CORE FUNCTIONS
# ---------------------------------------------------------
@st.cache_data
def get_data(tickers, years=7):
    end_date = datetime.date.today()
    start_date = end_date - datetime.timedelta(days=years*365)
    all_tickers = tickers + ['^GSPC']
    try:
        data = yf.download(all_tickers, start=start_date, end=end_date, progress=False, auto_adjust=True)['Close']
        if isinstance(data, pd.Series):
            data = data.to_frame()
            data.columns = all_tickers
        data = data.ffill().dropna()
        return data
    except Exception as e:
        return pd.DataFrame()

@st.cache_data
def get_ohlc_data(ticker, years=2):
    try:
        df = yf.download(ticker, period="730d", interval="1h", progress=False, auto_adjust=True)
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        
        if df.empty:
            return pd.DataFrame()
        return df
    except Exception as e:
        return pd.DataFrame()

def run_backtest_strategy(df_hourly, sma_period, tp_usd, sl_usd, contract_size):
    # Data Copy & Timezone Fix
    df_h = df_hourly.copy()
    if df_h.index.tz is not None:
        df_h.index = df_h.index.tz_localize(None)
        
    # Resample to Daily
    df_daily = df_h.resample('D').agg({'Open': 'first', 'High': 'max', 'Low': 'min', 'Close': 'last'}).dropna()
    df_daily['Daily_SMA'] = df_daily['Close'].rolling(window=sma_period).mean()
    df_daily['Prev_Daily_Close'] = df_daily['Close'].shift(1)
    df_daily['Prev_Daily_SMA'] = df_daily['Daily_SMA'].shift(1)
    
    # Merge Hourly with Daily Stats
    df_h['Date_Only'] = df_h.index.normalize()
    df_daily['Date_Only'] = df_daily.index.normalize()
    
    daily_stats = df_daily[['Date_Only', 'Daily_SMA', 'Prev_Daily_Close', 'Prev_Daily_SMA']]
    merged_data = pd.merge(df_h.reset_index(), daily_stats, on='Date_Only', how='left')
    merged_data.set_index('Datetime', inplace=True)
    merged_data[['Daily_SMA', 'Prev_Daily_Close', 'Prev_Daily_SMA']] = merged_data[['Daily_SMA', 'Prev_Daily_Close', 'Prev_Daily_SMA']].ffill()
    merged_data = merged_data.dropna()
    
    # Simulation Variables
    active_trades = []
    closed_trades_pnl = 0
    total_wins = 0
    total_losses = 0
    
    tp_points = tp_usd / contract_size
    sl_points = sl_usd / contract_size
    
    # Numpy Arrays for Speed
    open_arr = merged_data['Open'].values
    high_arr = merged_data['High'].values
    low_arr = merged_data['Low'].values
    close_arr = merged_data['Close'].values
    daily_sma_arr = merged_data['Daily_SMA'].values
    prev_d_close_arr = merged_data['Prev_Daily_Close'].values
    prev_d_sma_arr = merged_data['Prev_Daily_SMA'].values
    
    for i in range(len(merged_data)):
        curr_open = float(open_arr[i])
        curr_high = float(high_arr[i])
        curr_low = float(low_arr[i])
        curr_close = float(close_arr[i])
        curr_daily_sma = float(daily_sma_arr[i])
        prev_close = float(prev_d_close_arr[i])
        prev_sma = float(prev_d_sma_arr[i])
        
        # 1. NEW POSITION ENTRY
        if prev_close > prev_sma and curr_low <= curr_daily_sma:
            entry_price = curr_daily_sma if curr_open > curr_daily_sma else curr_open
            active_trades.append({'type': 'long', 'entry_price': entry_price, 'sl': entry_price - sl_points, 'tp': entry_price + tp_points, 'entry_index': i})
            
        elif prev_close < prev_sma and curr_high >= curr_daily_sma:
            entry_price = curr_daily_sma if curr_open < curr_daily_sma else curr_open
            active_trades.append({'type': 'short', 'entry_price': entry_price, 'sl': entry_price + sl_points, 'tp': entry_price - tp_points, 'entry_index': i})

        # 2. MANAGE ACTIVE TRADES
        remaining_trades = []
        for trade in active_trades:
            trade_closed = False
            
            if trade['type'] == 'long':
                # Same bar check
                if trade['entry_index'] == i:
                    if curr_low <= trade['sl']:
                        closed_trades_pnl -= sl_usd
                        total_losses += 1
                        trade_closed = True
                    elif not trade_closed and (curr_close > curr_open and curr_high >= trade['tp']):
                        closed_trades_pnl += tp_usd
                        total_wins += 1
                        trade_closed = True
                # Subsequent bars
                else:
                    if curr_low <= trade['sl']:
                        closed_trades_pnl -= sl_usd
                        total_losses += 1
                        trade_closed = True
                    elif curr_high >= trade['tp']:
                        closed_trades_pnl += tp_usd
                        total_wins += 1
                        trade_closed = True
            
            elif trade['type'] == 'short':
                if trade['entry_index'] == i:
                    if curr_high >= trade['sl']:
                        closed_trades_pnl -= sl_usd
                        total_losses += 1
                        trade_closed = True
                    elif not trade_closed and (curr_close < curr_open and curr_low <= trade['tp']):
                        closed_trades_pnl += tp_usd
                        total_wins += 1
                        trade_closed = True
                else:
                    if curr_high >= trade['sl']:
                        closed_trades_pnl -= sl_usd
                        total_losses += 1
                        trade_closed = True
                    elif curr_low <= trade['tp']:
                        closed_trades_pnl += tp_usd
                        total_wins += 1
                        trade_closed = True
            
            if not trade_closed:
                remaining_trades.append(trade)
        active_trades = remaining_trades

    total_trades_count = total_wins + total_losses
    win_rate = (total_wins / total_trades_count * 100) if total_trades_count > 0 else 0
    return closed_trades_pnl, win_rate, total_trades_count

def calculate_performance_metrics(portfolio_returns, benchmark_returns):
    rf_rate = 0.04
    daily_rf = rf_rate / 252
    excess_returns = portfolio_returns - daily_rf
    sharpe_ratio = (excess_returns.mean() / portfolio_returns.std()) * np.sqrt(252) if portfolio_returns.std() > 0 else 0
    downside_returns = excess_returns[excess_returns < 0]
    downside_std = downside_returns.std()
    sortino_ratio = (excess_returns.mean() / downside_std) * np.sqrt(252) if downside_std > 0 else 0
    try:
        covariance = np.cov(portfolio_returns, benchmark_returns)[0][1]
        variance = np.var(benchmark_returns)
        beta = covariance / variance
    except:
        beta = 0
    return sharpe_ratio, sortino_ratio, beta

def calculate_advanced_risk(positions_df, correlation_matrix, confidence_level=0.99):
    exposures = positions_df['Signed_Exposure'].values
    annual_ivs = positions_df['IV_Percent'].values / 100
    daily_vols = annual_ivs / np.sqrt(252)
    vol_matrix = np.outer(daily_vols, daily_vols)
    covariance_matrix = correlation_matrix.values * vol_matrix
    portfolio_variance = np.dot(exposures, np.dot(covariance_matrix, exposures))
    portfolio_std_dev = np.sqrt(abs(portfolio_variance))
    z_score = norm.ppf(confidence_level)
    
    standalone_vars = np.abs(exposures) * daily_vols * z_score
    undiversified_var = np.sum(standalone_vars)
    diversified_var = portfolio_std_dev * z_score
    diversification_benefit = undiversified_var - diversified_var
    
    cov_times_exposure = np.dot(covariance_matrix, exposures)
    if portfolio_std_dev > 0:
        marginal_vars = (cov_times_exposure / portfolio_std_dev) * z_score
    else:
        marginal_vars = np.zeros_like(exposures)
    component_vars = marginal_vars * exposures
    
    positions_df['Marginal_VaR'] = marginal_vars
    positions_df['Component_VaR'] = component_vars
    positions_df['Risk_Contribution_%'] = (component_vars / diversified_var * 100) if diversified_var > 0 else 0
    
    return diversified_var, portfolio_std_dev, positions_df, undiversified_var, diversification_benefit

# ---------------------------------------------------------
# UI - SIDEBAR
# ---------------------------------------------------------
st.sidebar.title("Trading Desk")

# --- 1. POSITION INPUT ---
st.sidebar.markdown("### 1. Portfolio Inputs")
portfolio_input = []
selected_tickers = []

for category, items in INSTRUMENTS.items():
    with st.sidebar.expander(f"📁 {category}", expanded=False):
        for ticker, info in items.items():
            st.markdown(f"**{info['name']}**") 
            c1, c2, c3 = st.columns([1.5, 1, 1.2])
            lots = c1.number_input("Lots", min_value=0.0, step=0.1, key=f"lot_{ticker}", label_visibility="collapsed")
            side = c2.selectbox("Side", ["Long", "Short"], key=f"side_{ticker}", label_visibility="collapsed")
            iv = c3.number_input("IV%", value=info['def_iv'], step=0.1, key=f"iv_{ticker}", help="Annual IV")
            
            if lots > 0:
                direction = 1 if side == 'Long' else -1
                portfolio_input.append({
                    "Ticker": ticker,
                    "Name": info["name"],
                    "Category": info.get("cat", "Other"),
                    "Lots": lots,
                    "Side": side,
                    "Direction": direction,
                    "Size": info["size"],
                    "IV_Percent": iv
                })
                selected_tickers.append(ticker)

# --- 2. SCENARIO ANALYSIS ---
st.sidebar.markdown("---")
st.sidebar.markdown("### ⚡ Stress Testing")
shock_equity = st.sidebar.slider("Equity Shock (%)", -50, 50, 0, step=1)
shock_crypto = st.sidebar.slider("Crypto Shock (%)", -50, 50, 0, step=1)
shock_fx = st.sidebar.slider("FX Shock (%)", -20, 20, 0, step=1)
shock_commodity = st.sidebar.slider("Commodity Shock (%)", -30, 30, 0, step=1)

scenarios = {"Equity": shock_equity/100, "Crypto": shock_crypto/100, "FX": shock_fx/100, "Commodity": shock_commodity/100}

# ---------------------------------------------------------
# MAIN SCREEN
# ---------------------------------------------------------
st.title("🛡️ Institutional Risk Terminal (v4.3)")
st.markdown("**Modules:** VaR Engine | Scenario Analysis | Performance | **Backtest Optimizer**")

tab1, tab2, tab3, tab4, tab5 = st.tabs(["📊 Risk Breakdown", "📉 Simulation & Heatmap", "📑 Detailed Report", "🚀 SMA Backtest Optimizer", "📋 Data Cleaner"])

if selected_tickers:
    with st.spinner('Running FRM Risk Models...'):
        pass

run_risk = st.sidebar.button("CALCULATE RISK", type="primary")

if run_risk:
    if not selected_tickers:
        st.error("⚠️ Please enter positions from the sidebar.")
    else:
        # 1. Data Prep
        raw_data = get_data(selected_tickers, years=2)
        if not raw_data.empty:
            if '^GSPC' in raw_data.columns:
                benchmark_data = raw_data['^GSPC']
            else:
                benchmark_data = raw_data.iloc[:, 0]
            
            asset_data = raw_data[selected_tickers]
            latest_prices = asset_data.iloc[-1]
            returns_df = asset_data.pct_change().dropna()
            benchmark_returns = benchmark_data.pct_change().dropna() 
            common_index = returns_df.index.intersection(benchmark_returns.index)
            returns_df = returns_df.loc[common_index]
            benchmark_returns = benchmark_returns.loc[common_index]

            # 2. Positions
            pos_df = pd.DataFrame(portfolio_input)
            current_prices_list = []
            simulated_pnl_impact = []
            
            for index, row in pos_df.iterrows():
                base_price = latest_prices[row['Ticker']]
                shock_pct = scenarios.get(row['Category'], 0)
                shocked_price = base_price * (1 + shock_pct)
                current_val = base_price * row['Lots'] * row['Size'] * row['Direction']
                shocked_val = shocked_price * row['Lots'] * row['Size'] * row['Direction']
                current_prices_list.append(base_price)
                simulated_pnl_impact.append(shocked_val - current_val)
            
            pos_df['Price'] = current_prices_list
            pos_df['Scenario_PnL'] = simulated_pnl_impact
            pos_df['Gross_Exposure'] = pos_df['Price'] * pos_df['Lots'] * pos_df['Size']
            pos_df['Signed_Exposure'] = pos_df['Gross_Exposure'] * pos_df['Direction']
            
            total_gross_exposure = pos_df['Gross_Exposure'].sum()
            total_net_exposure = pos_df['Signed_Exposure'].sum()
            total_scenario_pnl = pos_df['Scenario_PnL'].sum()

            # 3. Metrics
            if total_gross_exposure > 0:
                weights = pos_df.set_index('Ticker')['Signed_Exposure'] / total_net_exposure if total_net_exposure != 0 else 0
                aligned_weights = [weights.get(t, 0) for t in returns_df.columns]
                portfolio_daily_returns = returns_df.dot(aligned_weights)
                sharpe, sortino, beta = calculate_performance_metrics(portfolio_daily_returns, benchmark_returns)
            else:
                sharpe, sortino, beta = 0, 0, 0

            # 4. Risk
            corr_matrix = returns_df[pos_df['Ticker']].corr()
            iv_var_99, iv_daily_std_dollar, detailed_risk_df, undiv_var, div_benefit = calculate_advanced_risk(pos_df, corr_matrix)
            iv_var_95 = iv_daily_std_dollar * norm.ppf(0.95)
            portfolio_iv_vol_pct = ((iv_daily_std_dollar * np.sqrt(252)) / total_gross_exposure) * 100 if total_gross_exposure > 0 else 0
            hist_pnl = returns_df[pos_df['Ticker']].dot(pos_df['Signed_Exposure'].values)
            hist_var_99 = hist_pnl.quantile(0.01)

            # --- VISUALIZATION ---
            with tab1:
                k1, k2, k3, k4 = st.columns(4)
                k1.metric("Gross Exposure", f"${total_gross_exposure:,.0f}")
                k2.metric("IV VaR (99%)", f"${iv_var_99:,.0f}")
                k3.metric("Div. Benefit", f"${div_benefit:,.0f}", delta=f"%{(div_benefit/undiv_var)*100:.1f}")
                k4.metric("Exp. Volatility", f"%{portfolio_iv_vol_pct:.2f}")
                
                c1, c2 = st.columns(2)
                with c1:
                    fig_pie = px.pie(detailed_risk_df, values='Gross_Exposure', names='Name', title='Asset Allocation', hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                with c2:
                    colors = ['crimson' if x > 0 else 'mediumseagreen' for x in detailed_risk_df['Component_VaR']]
                    fig_bar = go.Figure(go.Bar(x=detailed_risk_df['Component_VaR'], y=detailed_risk_df['Name'], orientation='h', marker_color=colors))
                    fig_bar.update_layout(title="Risk Contribution ($)")
                    st.plotly_chart(fig_bar, use_container_width=True)

            with tab2:
                c1, c2 = st.columns(2)
                with c1:
                    fig = go.Figure()
                    fig.add_trace(go.Histogram(x=hist_pnl, name='Historical', opacity=0.6, marker_color='#3498DB'))
                    fig.add_vline(x=-iv_var_99, line_dash="dash", line_color="red", annotation_text="VaR 99%")
                    st.plotly_chart(fig, use_container_width=True)
                with c2:
                    st.metric("Scenario Impact", f"${total_scenario_pnl:,.0f}", delta=f"{(total_scenario_pnl/total_gross_exposure)*100:.2f}%")
                    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation")
                    st.plotly_chart(fig_corr, use_container_width=True)
            
            with tab3:
                st.dataframe(detailed_risk_df.style.format({"Signed_Exposure": "${:,.0f}", "Marginal_VaR": "{:.4f}", "Component_VaR": "${:,.0f}", "Risk_Contribution_%": "%{:.2f}"}).background_gradient(subset=['Risk_Contribution_%'], cmap='Reds'), width="stretch")

# ---------------------------------------------------------
# TAB 4: BACKTEST OPTIMIZER (GRID SEARCH)
# ---------------------------------------------------------
with tab4:
    st.header("🧪 Multi-Parameter Optimizer (Grid Search)")
    st.markdown("The algorithm tests combinations of **SMA 2-200** periods and your selected **Risk:Reward (R:R)** ratios.")
    st.info("ℹ️ **Method:** Fixed Stop Loss ($) is set. Take Profit (TP) is calculated dynamically based on R:R ratio. E.g., SL $500, R:R 1:2 means TP is $1000.")
    
    bc1, bc2 = st.columns(2)
    with bc1:
        selected_inst_name = st.selectbox("Instrument to Test", list(FLAT_INSTRUMENTS.keys()))
        ticker_to_test = FLAT_INSTRUMENTS[selected_inst_name]['ticker']
        contract_size_test = FLAT_INSTRUMENTS[selected_inst_name]['size']
        st.info(f"**{ticker_to_test}** selected. (Multiplier: {contract_size_test})")
        
    with bc2:
        sl_input = st.number_input("Stop Loss (Fixed SL) - USD ($)", value=500, step=100)
        rr_options = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0]
        selected_rrs = st.multiselect(
            "Risk:Reward Ratios to Test (R:R)", 
            options=rr_options,
            default=[1.0, 1.5, 2.0, 2.5, 3.0]
        )
        st.caption("E.g., 1.5 means targeting $750 profit for $500 risk.")

    if st.button("🚀 Start Grid Scan"):
        if not selected_rrs:
            st.error("Please select at least one R:R ratio.")
        else:
            with st.spinner(f"Testing all SMA x R:R combinations for {ticker_to_test}..."):
                df_backtest = get_ohlc_data(ticker_to_test, years=2)
                
                if df_backtest.empty:
                    st.error("Could not fetch data. Please try again later (API limit might be reached).")
                else:
                    sma_candidates = range(2, 201) 
                    results = []
                    total_iterations = len(sma_candidates) * len(selected_rrs)
                    progress_bar = st.progress(0)
                    counter = 0
                    
                    for sma in sma_candidates:
                        for rr in selected_rrs:
                            calculated_tp = sl_input * rr
                            net_pnl, win_rate, total_trades = run_backtest_strategy(
                                df_backtest, sma, calculated_tp, sl_input, contract_size_test
                            )
                            results.append({
                                "SMA_Period": sma, 
                                "RR_Ratio": rr,
                                "TP_Target ($)": calculated_tp,
                                "Net PnL ($)": net_pnl,
                                "Win Rate (%)": win_rate,
                                "Total Trades": total_trades
                            })
                            counter += 1
                            if counter % 10 == 0: 
                                progress_bar.progress(counter / total_iterations)
                    
                    progress_bar.progress(1.0)
                    res_df = pd.DataFrame(results)
                    sorted_df = res_df.sort_values(by="Net PnL ($)", ascending=False).reset_index(drop=True)
                    st.success("Optimization Complete!")
                    
                    # Champion
                    best_result = sorted_df.iloc[0]
                    c1, c2, c3, c4 = st.columns(4)
                    c1.metric("🏆 Best SMA", f"SMA {int(best_result['SMA_Period'])}")
                    c2.metric("💎 Best R:R", f"1 : {best_result['RR_Ratio']}")
                    c3.metric("Total Profit", f"${best_result['Net PnL ($)']:,.0f}")
                    c4.metric("Win Rate", f"%{best_result['Win Rate (%)']:.1f}")
                    st.divider()
                    
                    # Heatmap
                    st.subheader("🔥 Performance Heatmap")
                    st.markdown("Visualize which SMA and R:R zones are most profitable.")
                    
                    pivot_table = res_df.pivot(index='RR_Ratio', columns='SMA_Period', values='Net PnL ($)')
                    fig_heat = px.imshow(
                        pivot_table,
                        labels=dict(x="SMA Period", y="Risk:Reward Ratio", color="Net PnL ($)"),
                        x=pivot_table.columns,
                        y=pivot_table.index,
                        aspect="auto",
                        color_continuous_scale="RdBu"
                    )
                    fig_heat.update_yaxes(type='category')
                    st.plotly_chart(fig_heat, use_container_width=True)

                    # Table
                    st.subheader("🏅 Top 10 Profitable Combinations")
                    st.dataframe(
                        sorted_df.head(10).style.format({
                            "Net PnL ($)": "${:,.0f}",
                            "TP_Target ($)": "${:,.0f}",
                            "Win Rate (%)": "%{:.1f}",
                            "RR_Ratio": "{:.1f}"
                        }).background_gradient(subset=['Net PnL ($)'], cmap='Greens'),
                        width="stretch"
                    )

# ---------------------------------------------------------
# TAB 5: DATA CLEANER
# ---------------------------------------------------------
with tab5:
    st.header("📋 Market Data Parser & Auto-Fill")
    col_clean1, col_clean2 = st.columns([2, 1])
    with col_clean1:
        st.markdown("Paste your text below. The system will identify instruments and auto-fill the portfolio.")
        raw_text_input = st.text_area("Text Input:", height=150, placeholder="Example: EUR/USD 1.05 US30 34000...")
        if st.button("1. Analyze Data", type="primary"):
            if raw_text_input:
                df_cleaned = clean_market_data(raw_text_input)
                if not df_cleaned.empty:
                    st.session_state['cleaned_data'] = df_cleaned
                    st.success("Data parsed! View table on the right.")
                else:
                    st.error("Could not understand data format.")

    with col_clean2:
        st.subheader("Result & Import")
        if 'cleaned_data' in st.session_state:
            df_result = st.session_state['cleaned_data']
            st.dataframe(df_result, height=200, width="stretch")
            st.markdown("---")
            st.button("2. Import to Portfolio 📥", on_click=update_portfolio_from_cleaner)
