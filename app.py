import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score

# USD_TO_INR = 83.50

# -------------------------------------------------------------------
# 1. PAGE CONFIGURATION
# -------------------------------------------------------------------
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)

st.title("📈 Stock Price Prediction App")
st.write("A Machine Learning dashboard using Technical Indicators and Random Forest.")

# -------------------------------------------------------------------
# 2. SIDEBAR CONTROLS
# -------------------------------------------------------------------
st.sidebar.header("1. Stock Configuration")
ticker = st.sidebar.text_input("Stock Ticker Symbol", value="NVDA").upper()
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2020-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2026-01-01"))

# st.sidebar.header("2. Model Hyperparameters")
# n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100, step=10)
# max_depth = st.sidebar.slider("Max Tree Depth", 2, 20, 10)

# run_button = st.sidebar.button("Run Prediction")
st.sidebar.header("2. Trading Strategy Preset")
strategy = st.sidebar.selectbox(
    "Select Investor Profile",
    ["Balanced Investor (Recommended)", "Short-Term Swing Trader", "Custom Hyperparameters"]
)

# Automatically assign model parameters based on stock market profiles
if strategy == "Short-Term Swing Trader":
    n_estimators, max_depth = 50, 5
elif strategy == "Balanced Investor (Recommended)":
    n_estimators, max_depth = 100, 10
else:
    # Custom ML Controls
    n_estimators = st.sidebar.slider("Number of Trees (n_estimators)", 10, 200, 100, step=10, 
                                     help="Higher values make predictions more stable.")
    max_depth = st.sidebar.slider("Max Tree Depth", 2, 20, 10, 
                                  help="Controls complexity without memorizing noise.")

run_button = st.sidebar.button("Run Prediction")

# Inline help guide for non-technical users
with st.sidebar.expander("ℹ️ How to use this dashboard"):
    st.write("""
    1. Enter a ticker (e.g., `NVDA`, `AAPL`, or `RELIANCE.NS`).
    2. Choose an **Investor Profile**.
    3. Click **Run Prediction** to view trading signals and price forecasts.
    """)

# -------------------------------------------------------------------
# 3. HELPER FUNCTIONS
# -------------------------------------------------------------------
def get_currency_symbol(ticker_symbol):
    """Detects currency symbol using yfinance info or ticker suffix."""
    try:
        ticker_data = yf.Ticker(ticker_symbol)
        currency_code = ticker_data.info.get('currency', '').upper()
        
        currency_map = {
            'INR': '₹',
            'USD': '$',
            'EUR': '€',
            'GBP': '£',
            'JPY': '¥',
            'CAD': 'CA$',
            'AUD': 'A$'
        }
        if currency_code in currency_map:
            return currency_map[currency_code]
    except Exception:
        pass
        
    # Fallback: Infer from ticker suffix
    if ticker_symbol.endswith('.NS') or ticker_symbol.endswith('.BO'):
        return '₹'
    else:
        return '$'  # Default fallback to USD

@st.cache_data
def load_data(symbol, start, end):
    """Fetch historical stock price data from Yahoo Finance."""
    data = yf.download(symbol, start=start, end=end)
    if data.empty:
        return None
    # Flatten MultiIndex columns if returned by yfinance
    if isinstance(data.columns, pd.MultiIndex):
        data.columns = data.columns.get_level_values(0)
    df = data[['Open', 'High', 'Low', 'Close', 'Volume']].copy()
    #Convert USD price columns to INR (skips if ticker is Indian like RELIANCE.NS)
    if not (symbol.endswith(".NS") or symbol.endswith(".BO")):
        df[['Open', 'High', 'Low', 'Close']] = df[['Open', 'High', 'Low', 'Close']] 
        # * USD_TO_INR
        
    return df

def add_features(data):
    """Engineers stationary technical indicators and target returns."""
    df = data.copy()
    
    # 1. Price Ratios (Stationary)
    df['Price_to_SMA10'] = df['Close'] / df['Close'].rolling(10).mean()
    df['Price_to_SMA50'] = df['Close'] / df['Close'].rolling(50).mean()
    df['High_Low_Ratio'] = df['High'] / df['Low']
    df['Close_Open_Ratio'] = df['Close'] / df['Open']
    
    # 2. Daily Returns (Stationary)
    df['Return_1D'] = df['Close'].pct_change(1)
    df['Return_5D'] = df['Close'].pct_change(5)
    
    # 3. Volatility & Volume Ratios
    df['Volatility_5D'] = df['Return_1D'].rolling(5).std()
    df['Volume_Ratio'] = df['Volume'] / df['Volume'].rolling(10).mean()
    
    # 4. Standard RSI (Wilder's Smoothing)
    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-10)
    df['RSI'] = 100 - (100 / (1 + rs))
    
    # 5. MACD Ratio (Stationary)
    ema12 = df['Close'].ewm(span=12, adjust=False).mean()
    ema26 = df['Close'].ewm(span=26, adjust=False).mean()
    df['MACD_Ratio'] = (ema12 - ema26) / df['Close']
    
    # Target: Next-Day Percentage Return
    df['Target_Return'] = df['Close'].pct_change().shift(-1)
    
    return df


# def add_features(data):
#     """Engineers Technical Indicators and Lag features."""
#     df = data.copy()
    
#     # Moving Averages
#     df['SMA_10'] = df['Close'].rolling(window=10).mean()
#     df['SMA_50'] = df['Close'].rolling(window=50).mean()

#     # Standard Wilder's RSI (14-period exponential smoothing)
#     delta = df['Close'].diff()
#     gain = delta.clip(lower=0)
#     loss = -delta.clip(upper=0)
#     avg_gain = gain.ewm(alpha=1/14, adjust=False).mean()
#     avg_loss = loss.ewm(alpha=1/14, adjust=False).mean()
#     rs = avg_gain / avg_loss
#     df['RSI'] = 100 - (100 / (1 + rs))
    
#     # MACD
#     ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
#     ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
#     df['MACD'] = ema_12 - ema_26
#     df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
#     # Price Lags
#     df['Lag_1'] = df['Close'].shift(1)
#     df['Lag_2'] = df['Close'].shift(2)
    
#     # Target: Next-Day Close
#     df['Target'] = df['Close'].shift(-1)
    
#     return df
    
    # Relative Strength Index (RSI - 14 days)
    # delta = df['Close'].diff()
    # gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
    # loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
    # rs = gain / loss
    # df['RSI'] = 100 - (100 / (1 + rs))
    
    # # MACD
    # ema_12 = df['Close'].ewm(span=12, adjust=False).mean()
    # ema_26 = df['Close'].ewm(span=26, adjust=False).mean()
    # df['MACD'] = ema_12 - ema_26
    # df['MACD_Signal'] = df['MACD'].ewm(span=9, adjust=False).mean()
    
    # # Lags & Target (Next-Day Close)
    # df['Lag_1'] = df['Close'].shift(1)
    # df['Lag_2'] = df['Close'].shift(2)
    # df['Target'] = df['Close'].shift(-1)
    
    # df.dropna(inplace=True)
    # return df

# -------------------------------------------------------------------
# 4. MAIN APPLICATION EXECUTION
# -------------------------------------------------------------------

if run_button:
    with st.spinner("Downloading stock data and training model..."):
        # Detect currency dynamically
        currency = get_currency_symbol(ticker)

        # Step A: Load Raw Data
        raw_df = load_data(ticker, start_date, end_date)
        
        if raw_df is None or raw_df.empty:
            st.error(f"Failed to fetch data for ticker '{ticker}'. Please check the symbol and date range.")
        else:
            # Step B: Feature Engineering
            df = add_features(raw_df)
            
            # Use ONLY stationary features (No raw Close/High/Low)
            features = [
                'Price_to_SMA10', 'Price_to_SMA50', 'High_Low_Ratio', 
                'Close_Open_Ratio', 'Return_1D', 'Return_5D', 
                'Volatility_5D', 'Volume_Ratio', 'RSI', 'MACD_Ratio'
            ]
            
            # Step C: Isolate Today's Unshifted Row & Clean Training Set
            latest_feature_row = df.iloc[[-1]][features]
            latest_actual_price = raw_df['Close'].iloc[-1]
            
            df_clean = df.dropna().copy()
            X = df_clean[features]
            y_return = df_clean['Target_Return']
            
            # Step D: Chronological Split
            train_size = int(len(df_clean) * 0.8)
            X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
            y_train, y_test = y_return.iloc[:train_size], y_return.iloc[train_size:]
            
            # Step E: Model Training
            model = RandomForestRegressor(
                n_estimators=n_estimators, 
                max_depth=5,  # Constrained depth prevents overfitting
                random_state=42
            )
            model.fit(X_train, y_train)
            
            # Step F: Backtest & Tomorrow's Forecast
            predicted_returns = model.predict(X_test)
            next_day_predicted_return = model.predict(latest_feature_row)[0]
            
            # Reconstruct Historical Price Forecasts from Returns
            test_dates = df_clean.iloc[train_size:].index
            historical_actual_prices = raw_df.loc[test_dates, 'Close']
            historical_pred_prices = historical_actual_prices * (1 + predicted_returns)
            
            # Reconstruct Tomorrow's Price Forecast
            next_day_forecast = latest_actual_price * (1 + next_day_predicted_return)
            expected_change = next_day_predicted_return * 100
            
            # Step G: Performance & Directional Accuracy Metrics
            actual_return_directions = np.sign(y_test.values)
            pred_return_directions = np.sign(predicted_returns)
            
            valid_days = actual_return_directions != 0
            if np.sum(valid_days) > 0:
                directional_acc = np.mean(actual_return_directions[valid_days] == pred_return_directions[valid_days]) * 100
            else:
                directional_acc = 0.0

            rmse = np.sqrt(mean_squared_error(historical_actual_prices, historical_pred_prices))
            mae = mean_absolute_error(historical_actual_prices, historical_pred_prices)
            r2 = r2_score(y_test, predicted_returns)
            
            # Actionable Signal with Noise Filter (>0.5% Threshold)
            st.markdown("### 🚦 Model Trading Signal")
            if expected_change > 0.5:
                st.success(f"🟢 **BULLISH (BUY)** — Model forecasts a **+{expected_change:.2f}%** return for the next trading session.")
            elif expected_change < -0.5:
                st.error(f"🔴 **BEARISH (SELL)** — Model forecasts a **{expected_change:.2f}%** return for the next trading session.")
            else:
                st.warning(f"🟡 **NEUTRAL (HOLD)** — Minimal price movement expected (**{expected_change:.2f}%**).")
            
            st.markdown("---")
            
            # Metric Display
            st.markdown("### 📊 Model Performance Metrics")
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Directional Accuracy", f"{directional_acc:.1f}%", help="Correctly predicted UP/DOWN directions.")
            m2.metric("RMSE", f"{currency}{rmse:.2f}")  # <-- Updated
            m3.metric("MAE", f"{currency}{mae:.2f}")   # <-- Updated
            m4.metric("R² (Returns)", f"{r2:.4f}")
            
            st.markdown("---")

            # Split Layout Tabs
            tab_investor, tab_ml = st.tabs(["📈 Investor View", "⚙️ ML Engineering Metrics"])
            
            # --- TAB 1: INVESTOR VIEW ---
            with tab_investor:
                # st.markdown(f"### 📉 Historical Backtest: Actual vs Predicted ({ticker})")
                # fig, ax = plt.subplots(figsize=(10, 4))
                # ax.plot(test_dates, historical_actual_prices, label="Actual Price", color="#1f77b4", linewidth=1.5)
                # ax.plot(test_dates, historical_pred_prices, label="Predicted Price", color="#ff7f0e", linestyle="--", linewidth=1.5)
                # ax.set_xlabel("Date")
                # ax.set_ylabel(f"Price ({currency})")  # <-- Updated Chart Y-axis
                # ax.legend()
                # ax.grid(True, linestyle=":", alpha=0.6)
                # st.pyplot(fig)

                # --- Top Bar: Title & Refresh Button ---
                ref_col1, ref_col2 = st.columns([0.85, 0.15])
                with ref_col1:
                    st.markdown(f"### 📉 Historical Backtest & Volume ({ticker})")
                with ref_col2:
                    if st.button("🔄 Refresh Data"):
                        st.cache_data.clear()
                        st.rerun()

                # --- Dual Subplot: Price Chart + Trading Volume Bar Chart ---
                fig, (ax1, ax2) = plt.subplots(
                    2, 1, figsize=(10, 6), sharex=True, 
                    gridspec_kw={'height_ratios': [3, 1]}
                )

                # Top Subplot (Prices)
                ax1.plot(test_dates, historical_actual_prices, label="Actual Price", color="#1f77b4", linewidth=1.5)
                ax1.plot(test_dates, historical_pred_prices, label="Predicted Price", color="#ff7f0e", linestyle="--", linewidth=1.5)
                ax1.set_ylabel(f"Price ({currency})")
                ax1.legend(loc="upper left")
                ax1.grid(True, linestyle=":", alpha=0.6)

                # Bottom Subplot (Volume)
                test_volumes = raw_df.loc[test_dates, 'Volume']
                ax2.bar(test_dates, test_volumes, color="#2ca02c", alpha=0.6, width=1.0)
                ax2.set_ylabel("Volume")
                ax2.set_xlabel("Date")
                ax2.grid(True, linestyle=":", alpha=0.6)

                plt.tight_layout()
                st.pyplot(fig)
                
                col1, col2 = st.columns(2)
                col1.metric("Latest Market Close", f"{currency}{latest_actual_price:.2f}")  # <-- Updated
                col2.metric("Next Trading Session Forecast", f"{currency}{next_day_forecast:.2f}", delta=f"{expected_change:+.2f}%")  # <-- Updated

                st.markdown("### 📊 Key Stock Statistics")

                # Slice the dataset to only look at the last ~252 trading days (1 year)
                # one_year_df = raw_df.tail(252)
                latest_date = raw_df.index[-1]
                one_year_ago = latest_date - pd.DateOffset(years=1)
                one_year_df = raw_df.loc[raw_df.index >= one_year_ago]

                sc1, sc2, sc3, sc4 = st.columns(4)
                sc1.metric("Current Price", f"{currency}{latest_actual_price:.2f}")
                sc2.metric("52-Week High", f"{currency}{one_year_df['Close'].max():.2f}")
                sc3.metric("52-Week Low", f"{currency}{one_year_df['Close'].min():.2f}")
                sc4.metric("Avg Daily Volume", f"{int(raw_df['Volume'].mean()):,}")
                
                # sc1, sc2, sc3, sc4 = st.columns(4)
                # sc1.metric("Current Price", f"{currency}{latest_actual_price:.2f}")  # <-- Updated
                # sc2.metric("52-Week High", f"{currency}{raw_df['Close'].max():.2f}") # <-- Updated
                # sc3.metric("52-Week Low", f"{currency}{raw_df['Close'].min():.2f}")  # <-- Updated
                # sc4.metric("Avg Daily Volume", f"{int(raw_df['Volume'].mean()):,}")

                st.markdown("---")

                with st.expander("📄 View Raw Historical Stock Data & Download"):
                    st.dataframe(df_clean.tail(10).sort_index(ascending=False), use_container_width=True)
                    csv = df_clean.to_csv().encode('utf-8')
                    st.download_button("📥 Download Dataset as CSV", data=csv, file_name=f"{ticker}_stock_data.csv", mime="text/csv")

            # --- TAB 2: ML ENGINEERING VIEW ---
            with tab_ml:
                st.markdown("### 🔍 Feature Importance Analysis")
                importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
                fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
                importances.plot(kind='barh', ax=ax_imp, color='#2ca02c')
                ax_imp.set_xlabel("Relative Importance")
                st.pyplot(fig_imp)

                st.markdown("---")

                # 2. Model Edge & Statistical Validation
                st.markdown("### 🧪 Model Edge & Statistical Validation")

                # Naive Baseline: Directional accuracy if you guessed 'UP' every single day
                actual_return_directions = np.sign(y_test.values)
                pred_return_directions = np.sign(predicted_returns)
                
                naive_baseline_acc = np.mean(actual_return_directions == 1) * 100

                # Signal Threshold Filtering: Accuracy on high-confidence trades (> ±0.5% predicted move)
                threshold = 0.005  # 0.5% return threshold
                strong_signals_mask = np.abs(predicted_returns) >= threshold

                if np.sum(strong_signals_mask) > 0:
                    strong_directional_acc = np.mean(
                        actual_return_directions[strong_signals_mask] == pred_return_directions[strong_signals_mask]
                    ) * 100
                    strong_trade_count = np.sum(strong_signals_mask)
                else:
                    strong_directional_acc = 0.0
                    strong_trade_count = 0

                # Display Validation Metrics
                v1, v2, v3 = st.columns(3)
                v1.metric(
                    label="Overall Directional Acc.", 
                    value=f"{directional_acc:.1f}%", 
                    delta=f"{directional_acc - naive_baseline_acc:+.1f}% vs Baseline",
                    help="Target range for quantitative trading models is 53% - 58%."
                )
                v2.metric(
                    label="High-Confidence Acc. (>±0.5%)", 
                    value=f"{strong_directional_acc:.1f}%", 
                    help=f"Accuracy when the model predicts a movement larger than 0.5% ({strong_trade_count} qualified trades)."
                )
                v3.metric(
                    label="Naive Buy & Hold Baseline", 
                    value=f"{naive_baseline_acc:.1f}%", 
                    help="Directional accuracy achieved if you simply guessed 'UP' every single day."
                )

                st.markdown("---")

                # 3. Strategy Returns vs. Benchmark Simulation
                st.markdown("### 💰 Backtested Portfolio Performance")

                # Trading Strategy: Long (+1) when model predicts return > 0, Cash (0) when return <= 0
                strategy_returns = np.where(predicted_returns > 0, y_test.values, 0)

                cum_benchmark = (1 + y_test.values).cumprod() - 1
                cum_strategy = (1 + strategy_returns).cumprod() - 1

                fig_backtest, ax_bt = plt.subplots(figsize=(10, 4))
                ax_bt.plot(test_dates, cum_benchmark * 100, label="Buy & Hold Benchmark (%)", color="#7f7f7f", linestyle="--")
                ax_bt.plot(test_dates, cum_strategy * 100, label="Model Strategy (%)", color="#2ca02c", linewidth=2)
                ax_bt.set_xlabel("Date")
                ax_bt.set_ylabel("Cumulative Return (%)")
                ax_bt.legend()
                ax_bt.grid(True, linestyle=":", alpha=0.6)
                st.pyplot(fig_backtest)

                # Return Summary Metrics
                b1, b2 = st.columns(2)
                b1.metric("Benchmark Total Return", f"{cum_benchmark[-1]*100:+.2f}%")
                b2.metric("Model Strategy Return", f"{cum_strategy[-1]*100:+.2f}%")
                


else:
    st.info("👈 Use the sidebar to set your stock ticker, dates, and parameters, then click **Run Prediction**.")

# if run_button:
#     with st.spinner("Downloading stock data and training model..."):
#         # Step A: Load Raw Data
#         raw_df = load_data(ticker, start_date, end_date)
        
#         if raw_df is None:
#             st.error(f"Failed to fetch data for ticker '{ticker}'. Please check the symbol and date range.")
#         else:
#             # Step B: Feature Engineering
#             df = add_features(raw_df)
            
#             # Step B: Feature Engineering
#             df = add_features(raw_df)
            
#             features = ['Open', 'High', 'Low', 'Close', 'Volume', 
#                         'SMA_10', 'SMA_50', 'RSI', 'MACD', 'MACD_Signal', 'Lag_1', 'Lag_2']
            
#             # Step C: Isolate Out-of-Sample Row & Clean Training Set
#             # 1. Extract the unshifted latest row from df to predict TOMORROW'S price
#             latest_feature_row = df.iloc[[-1]][features]
#             latest_actual_price = raw_df['Close'].iloc[-1]
            
#             # 2. Clean feature set for historical backtesting (drops the NaN target row)
#             df_clean = df.dropna().copy()
#             X = df_clean[features]
#             y = df_clean['Target']
            
#             # Step D: Train-Test Chronological Split
#             train_size = int(len(df_clean) * 0.8)
#             X_train, X_test = X.iloc[:train_size], X.iloc[train_size:]
#             y_train, y_test = y.iloc[:train_size], y.iloc[train_size:]
            
#             # Step E: Model Training & Historical Backtest
#             model = RandomForestRegressor(
#                 n_estimators=n_estimators, 
#                 max_depth=max_depth, 
#                 random_state=42
#             )
#             model.fit(X_train, y_train)
#             predictions = model.predict(X_test)
            
#             # Step F: Out-of-Sample True Next-Day Forecast
#             next_day_forecast = model.predict(latest_feature_row)[0]
#             expected_change = ((next_day_forecast - latest_actual_price) / latest_actual_price) * 100
            
#             # Step G: Evaluation Metrics (Historical Test Set)
#             rmse = np.sqrt(mean_squared_error(y_test, predictions))
#             mae = mean_absolute_error(y_test, predictions)
#             r2 = r2_score(y_test, predictions)
            
#             # Actionable Buy/Sell Signal Card
#             st.markdown("### 🚦 Model Trading Signal")
#             if expected_change > 1.0:
#                 st.success(f"🟢 **BULLISH (BUY)** — Model forecasts a **+{expected_change:.2f}%** price increase for the next trading session.")
#             elif expected_change < -1.0:
#                 st.error(f"🔴 **BEARISH (SELL)** — Model forecasts a **{expected_change:.2f}%** drop for the next trading session.")
#             else:
#                 st.warning(f"🟡 **NEUTRAL (HOLD)** — Minimal price movement expected (**{expected_change:.2f}%**).")
            
#             st.markdown("---")
            
#             # Display Metric Cards
#             st.markdown("### 📊 Model Performance Metrics")
#             m1, m2, m3 = st.columns(3)
#             m1.metric("Root Mean Squared Error (RMSE)", f"₹{rmse:.2f}")
#             m2.metric("Mean Absolute Error (MAE)", f"₹{mae:.2f}")
#             m3.metric("R² Accuracy Score", f"{r2:.4f}")
            
#             st.markdown("---")

#             # Split layout into two user tabs
#             tab_investor, tab_ml = st.tabs(["📈 Investor View", "⚙️ ML Engineering Metrics"])
            
            # --- TAB 1: INVESTOR VIEW ---
            # with tab_investor:
            #     st.markdown(f"### 📉 Stock Price Trend & Forecast ({ticker})")
            #     fig, ax = plt.subplots(figsize=(10, 4))
            #     ax.plot(y_test.index, y_test.values, label="Actual Price", color="#1f77b4", linewidth=1.5)
            #     ax.plot(y_test.index, predictions, label="Predicted Price", color="#ff7f0e", linestyle="--", linewidth=1.5)
            #     ax.set_xlabel("Date")
            #     ax.set_ylabel("Price (₹)")
            #     ax.legend()
            #     ax.grid(True, linestyle=":", alpha=0.6)
            #     st.pyplot(fig)
                
            #     # Financial summary metrics
            #     col1, col2 = st.columns(2)
            #     col1.metric("Latest Market Price", f"₹{latest_actual_price:.2f}")
            #     # col2.metric("Model Predicted Price", f"₹{latest_predicted_price:.2f}", delta=f"{expected_change:.2f}%")
            #     col2.metric("Next Trading Session Forecast", f"₹{next_day_forecast:.2f}", delta=f"{expected_change:+.2f}%")

                # Key Stock Overview Metrics (52-Week Stats)
                # st.markdown("### 📊 Key Stock Statistics")
                # sc1, sc2, sc3, sc4 = st.columns(4)
                # sc1.metric("Current Price", f"₹{df['Close'].iloc[-1]:.2f}")
                # sc2.metric("52-Week High", f"₹{df['Close'].max():.2f}")
                # sc3.metric("52-Week Low", f"₹{df['Close'].min():.2f}")
                # sc4.metric("Avg Daily Volume", f"{int(df['Volume'].mean()):,}")

                # st.markdown("---")

                # # Expandable Raw Data Table & CSV Download Button
                # with st.expander("📄 View Raw Historical Stock Data & Download"):
                #     st.write(f"Showing recent historical price data for **{ticker}**:")
                #     st.dataframe(df.tail(10).sort_index(ascending=False), use_container_width=True)
                    
                #     csv = df.to_csv().encode('utf-8')
                #     st.download_button(
                #         label="📥 Download Dataset as CSV",
                #         data=csv,
                #         file_name=f"{ticker}_stock_data.csv",
                #         mime="text/csv"
                    # )

            # --- TAB 2: ML ENGINEERING VIEW ---
            # with tab_ml:
            #     # Calculate Directional Accuracy (Hit Rate)
            #     actual_diff = np.diff(y_test.values)
            #     pred_diff = np.diff(predictions)
                
            #     # Avoid division by zero if prices don't change
            #     valid_indices = actual_diff != 0
            #     if np.sum(valid_indices) > 0:
            #         directional_acc = np.mean(np.sign(actual_diff[valid_indices]) == np.sign(pred_diff[valid_indices])) * 100
            #     else:
            #         directional_acc = 0.0

                # st.markdown("### 📊 Regression & Directional Metrics")
                # m1, m2, m3, m4 = st.columns(4)
                # m1.metric("RMSE", f"₹{rmse:.2f}", help="Root Mean Squared Error: Penalizes large prediction errors.")
                # m2.metric("MAE", f"₹{mae:.2f}", help="Mean Absolute Error: Average price error in ₹.")
                # m3.metric("R² Score", f"{r2:.4f}", help="Variance explained (1.0 is perfect, <0 is poor).")
                # m4.metric("Directional Accuracy", f"{directional_acc:.1f}%", help="% of times the model correctly predicted UP or DOWN movement.")
                
                # st.markdown("---")
                
                # st.markdown("### 🔍 Feature Importance Analysis")
                # importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
                # fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
                # importances.plot(kind='barh', ax=ax_imp, color='#2ca02c')
                # ax_imp.set_xlabel("Relative Importance")
                # st.pyplot(fig_imp)


                # st.markdown("### 📊 Regression Metrics")
                # m1, m2, m3 = st.columns(3)
                # m1.metric("Root Mean Squared Error (RMSE)", f"₹{rmse:.2f}")
                # m2.metric("Mean Absolute Error (MAE)", f"₹{mae:.2f}")
                # m3.metric("R² Accuracy Score", f"{r2:.4f}")
                
                # st.markdown("---")
                
                # st.markdown("### 🔍 Feature Importance Analysis")
                # importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
                # fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
                # importances.plot(kind='barh', ax=ax_imp, color='#2ca02c')
                # ax_imp.set_xlabel("Relative Importance")
                # st.pyplot(fig_imp)
            
            # # Step F: Plot Predictions
            # st.markdown(f"### 📉 Actual vs Predicted Prices ({ticker})")
            # fig, ax = plt.subplots(figsize=(10, 4))
            # ax.plot(y_test.index, y_test.values, label="Actual Price", color="#1f77b4", linewidth=1.5)
            # ax.plot(y_test.index, predictions, label="Predicted Price", color="#ff7f0e", linestyle="--", linewidth=1.5)
            # ax.set_xlabel("Date")
            # ax.set_ylabel("Price (₹)")
            # ax.legend()
            # ax.grid(True, linestyle=":", alpha=0.6)
            # st.pyplot(fig)
            
            # # Step G: Feature Importances
            # st.markdown("### 🔍 Feature Importance Analysis")
            # importances = pd.Series(model.feature_importances_, index=features).sort_values(ascending=True)
            
            # fig_imp, ax_imp = plt.subplots(figsize=(8, 4))
            # importances.plot(kind='barh', ax=ax_imp, color='#2ca02c')
            # ax_imp.set_xlabel("Relative Importance")
            # st.pyplot(fig_imp)

# else:
#     st.info("👈 Use the sidebar to set your stock ticker, dates, and parameters, then click **Run Prediction**.")