import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt

# Heaviness indicator function with safe scalar conversion
def calculate_heaviness(open_price, close_price, volume, prev_day_range):
    # Force scalars to avoid ambiguous Series comparison
    try:
        volume = float(volume)
    except Exception:
        return np.nan

    try:
        prev_day_range = float(prev_day_range)
    except Exception:
        return np.nan

    if np.isnan(volume) or np.isnan(prev_day_range) or volume == 0 or prev_day_range == 0:
        return np.nan

    delta_p_per_volume = (close_price - open_price) / volume
    heaviness = (delta_p_per_volume / prev_day_range) * 100
    return max(0, min(100, 100 - abs(heaviness * 100)))

# Backtest logic
def backtest_heaviness(df, threshold=20, hold_minutes=15):
    trades = []
    position = None
    entry_time = None

    for i in range(len(df) - hold_minutes):
        row = df.iloc[i]
        if np.isnan(row['H%']):
            continue

        if position is None and row['H%'] < threshold:
            position = {
                'entry_time': row.name,
                'entry_price': row['Close']
            }
            entry_time = row.name

        elif position is not None and (row.name - entry_time) >= timedelta(minutes=hold_minutes):
            exit_price = df.iloc[i]['Close']
            trades.append({
                'entry': entry_time,
                'exit': row.name,
                'entry_price': position['entry_price'],
                'exit_price': exit_price,
                'return': (exit_price - position['entry_price']) / position['entry_price']
            })
            position = None

    return pd.DataFrame(trades)

# Streamlit UI
st.title("📉 Heaviness Indicator (H%) & Backtester")
ticker = st.text_input("Enter Stock Ticker", value="AAPL")
start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=5))
end_date = st.date_input("End Date", value=datetime.today())
threshold = st.slider("H% Buy Threshold", 0, 100, 20)
hold_period = st.slider("Hold Period (minutes)", 1, 60, 15)

if st.button("Run Analysis"):
    data = yf.download(ticker, start=start_date, end=end_date, interval='1m', progress=False)

    if data.empty:
        st.error("No data found. Try another ticker or date range.")
    else:
        # Calculate H%
        daily = yf.download(ticker, start=start_date - timedelta(days=2), end=end_date, interval='1d', progress=False)
        prev_day_range = daily['High'].shift(1) - daily['Low'].shift(1)

        intraday_df = data.copy()
        intraday_df['Date'] = intraday_df.index.date
        
        # Ensure PrevRange is scalar float
        def get_prev_range(d):
            matches = prev_day_range.loc[prev_day_range.index.date == d]
            return float(matches.values[0]) if len(matches) > 0 else np.nan
        
        intraday_df['PrevRange'] = intraday_df['Date'].map(get_prev_range)

        intraday_df['H%'] = [
            calculate_heaviness(row['Open'], row['Close'], row['Volume'], row['PrevRange'])
            for _, row in intraday_df.iterrows()
        ]

        st.write("Sample Data with H%")
        st.dataframe(intraday_df[['Open', 'Close', 'Volume', 'H%']].tail(10))

        # Run backtest
        st.subheader("Backtest Results")
        trades_df = backtest_heaviness(intraday_df, threshold=threshold, hold_minutes=hold_period)
        if trades_df.empty:
            st.warning("No trades matched your criteria.")
        else:
            st.dataframe(trades_df)
            total_return = trades_df['return'].sum()
            win_rate = (trades_df['return'] > 0).mean()
            avg_return = trades_df['return'].mean()

            st.metric("Total Return", f"{total_return * 100:.2f}%")
            st.metric("Win Rate", f"{win_rate * 100:.2f}%")
            st.metric("Avg Trade Return", f"{avg_return * 100:.2f}%")

            # Plot equity curve
            trades_df['Equity'] = (1 + trades_df['return']).cumprod()
            fig, ax = plt.subplots()
            ax.plot(trades_df['exit'], trades_df['Equity'], label="Equity Curve")
            ax.set_ylabel("Equity")
            ax.set_xlabel("Time")
            ax.legend()
            st.pyplot(fig)
