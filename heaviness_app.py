import streamlit as st
import pandas as pd
import yfinance as yf
from datetime import datetime, timedelta

def calculate_heaviness(open_price, close_price, volume, prev_day_range):
    # Handle None or NaN or zero safely
    if pd.isna(volume) or pd.isna(prev_day_range) or volume == 0 or prev_day_range == 0:
        return None
    return ((close_price - open_price) / prev_day_range) * (volume / 1_000_000)

# Function to get the previous day's range
def get_prev_range(date, daily_data):
    prev_day = date - timedelta(days=1)
    matches = daily_data[daily_data.index.date == prev_day.date()]
    if not matches.empty:
        high = matches['High'].iloc[0]
        low = matches['Low'].iloc[0]
        return float(high - low)
    return None

st.title("📈 Heaviness App")

ticker = st.text_input("Enter stock ticker:", "AAPL")
days = st.slider("Number of past days to analyze:", 1, 10, 3)

if st.button("Run Analysis"):
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    st.write(f"Fetching intraday data for {ticker}...")
    data = yf.download(ticker, start=start_date, end=end_date, interval='1m', progress=False, auto_adjust=False)

    st.write(f"Fetching daily data for {ticker}...")
    daily = yf.download(ticker, start=start_date - timedelta(days=5), end=end_date, interval='1d', progress=False, auto_adjust=False)

    if data.empty or daily.empty:
        st.error("No data found for the given ticker and date range.")
    else:
        intraday_df = data.copy()
        intraday_df['Date'] = intraday_df.index.date
        intraday_df['PrevRange'] = intraday_df['Date'].map(lambda d: get_prev_range(pd.Timestamp(d), daily))

        # Force scalar values before passing to the function
        intraday_df['H%'] = intraday_df.apply(
            lambda row: calculate_heaviness(
                float(row['Open'].iloc[0]) if isinstance(row['Open'], pd.Series) else float(row['Open']),
                float(row['Close'].iloc[0]) if isinstance(row['Close'], pd.Series) else float(row['Close']),
                float(row['Volume'].iloc[0]) if isinstance(row['Volume'], pd.Series) else float(row['Volume']),
                float(row['PrevRange'].iloc[0]) if isinstance(row['PrevRange'], pd.Series) else (float(row['PrevRange']) if pd.notna(row['PrevRange']) else None)
            ),
            axis=1
        )

        st.write("### Results")
        st.dataframe(intraday_df)

        st.line_chart(intraday_df['H%'])
