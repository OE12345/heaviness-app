import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
from typing import Optional


def to_float_safe(val):
    if isinstance(val, pd.Series):
        return float(val.iloc[0])
    return float(val)


def calculate_heaviness(
    open_price: float,
    close_price: float,
    volume: float,
    prev_day_range: float
) -> Optional[float]:
    try:
        if volume <= 0 or prev_day_range <= 0 or np.isnan(volume) or np.isnan(prev_day_range):
            return None
        delta_per_volume = (close_price - open_price) / volume
        heaviness_raw = (delta_per_volume / prev_day_range) * 100
        heaviness_score = 100 - abs(heaviness_raw * 100)
        return max(0.0, min(100.0, heaviness_score))
    except Exception:
        return None


def backtest_heaviness(
    df: pd.DataFrame,
    threshold: float = 20.0,
    hold_minutes: int = 15
) -> pd.DataFrame:
    trades = []
    position = None

    for idx in range(len(df) - hold_minutes):
        current_row = df.iloc[idx]

        h_value = current_row.get('H%')
        if pd.isna(h_value) or h_value is None:
            continue

        timestamp = pd.Timestamp(current_row.name)

        if position is None and h_value < threshold:
            position = {
                'entry_time': timestamp,
                'entry_price': current_row['Close']
            }
            continue

        if position is not None:
            elapsed = timestamp - position['entry_time']
            if isinstance(elapsed, pd.Timedelta) and elapsed >= timedelta(minutes=hold_minutes):
                exit_price = current_row['Close']
                trade_return = (exit_price - position['entry_price']) / position['entry_price']
                trades.append({
                    'entry': position['entry_time'],
                    'exit': timestamp,
                    'entry_price': position['entry_price'],
                    'exit_price': exit_price,
                    'return': trade_return
                })
                position = None

    return pd.DataFrame(trades)


def load_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.DataFrame:
    data = yf.download(
        ticker,
        start=start_date,
        end=end_date,
        interval='1m',
        progress=False,
        auto_adjust=True  # <-- Explicit to silence warning
    )
    if data.empty:
        raise ValueError(f"No intraday data found for {ticker} between {start_date} and {end_date}")
    return data


def load_daily_data(ticker: str, start_date: datetime, end_date: datetime) -> pd.Series:
    daily = yf.download(
        ticker,
        start=start_date - timedelta(days=5),
        end=end_date + timedelta(days=1),
        interval='1d',
        progress=False,
        auto_adjust=True  # <-- Explicit to silence warning
    )
    if daily.empty:
        raise ValueError(f"No daily data found for {ticker} between {start_date} and {end_date}")
    prev_day_range = (daily['High'] - daily['Low']).shift(1).dropna()
    prev_day_range.index = prev_day_range.index.date
    return prev_day_range


def main():
    st.set_page_config(page_title="Heaviness Indicator Backtester", layout="wide")

    st.title("📉 Heaviness Indicator (H%) & Backtester")

    ticker = st.text_input("Enter Stock Ticker", value="AAPL", max_chars=10).upper()
    start_date = st.date_input("Start Date", value=datetime.today() - timedelta(days=7))
    end_date = st.date_input("End Date", value=datetime.today())

    threshold = st.slider("H% Buy Threshold", min_value=0, max_value=100, value=20)
    hold_period = st.slider("Hold Period (minutes)", min_value=1, max_value=60, value=15)

    if start_date >= end_date:
        st.error("Start Date must be before End Date.")
        return

    if st.button("Run Analysis"):
        try:
            with st.spinner("Downloading data..."):
                intraday_df = load_data(ticker, start_date, end_date)
                prev_range_series = load_daily_data(ticker, start_date, end_date)

            intraday_df = intraday_df.copy()
            intraday_df['Date'] = intraday_df.index.date
            intraday_df['PrevRange'] = intraday_df['Date'].map(lambda d: prev_range_series.get(d, np.nan))

            def safe_calc(row):
                try:
                    return calculate_heaviness(
                        to_float_safe(row['Open']),
                        to_float_safe(row['Close']),
                        to_float_safe(row['Volume']),
                        to_float_safe(row['PrevRange']) if not pd.isna(row['PrevRange']) else np.nan
                    )
                except Exception:
                    return np.nan

            intraday_df['H%'] = intraday_df.apply(safe_calc, axis=1)

            st.subheader("Sample Data with H%")
            st.dataframe(intraday_df[['Open', 'Close', 'Volume', 'H%']].tail(10))

            st.subheader("Backtest Results")
            trades_df = backtest_heaviness(intraday_df, threshold=threshold, hold_minutes=hold_period)

            if trades_df.empty:
                st.warning("No trades matched your criteria.")
            else:
                st.dataframe(trades_df.style.format({
                    'entry_price': '${:,.2f}',
                    'exit_price': '${:,.2f}',
                    'return': '{:.2%}'
                }))

                total_return = trades_df['return'].sum()
                win_rate = (trades_df['return'] > 0).mean()
                avg_return = trades_df['return'].mean()

                st.metric("Total Return", f"{total_return:.2%}")
                st.metric("Win Rate", f"{win_rate:.2%}")
                st.metric("Average Trade Return", f"{avg_return:.2%}")

                trades_df['Equity'] = (1 + trades_df['return']).cumprod()

                fig, ax = plt.subplots(figsize=(10, 5))
                ax.plot(trades_df['exit'], trades_df['Equity'], label="Equity Curve", color='tab:blue', linewidth=2)
                ax.set_xlabel("Exit Time")
                ax.set_ylabel("Equity Growth")
                ax.set_title(f"Equity Curve for {ticker} Heaviness Backtest")
                ax.grid(True, linestyle='--', alpha=0.7)
                ax.legend()
                st.pyplot(fig)

        except Exception as e:
            st.error(f"Error during analysis: {e}")


if __name__ == "__main__":
    main()
